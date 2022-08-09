import pandas as pd
import torch
from tqdm import tqdm
from skmultiflow.trees import HoeffdingTreeRegressor, HoeffdingAdaptiveTreeRegressor
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error
from river import preprocessing
from sklearn.tree import DecisionTreeRegressor

from src.data_stores import PandasDataset, OnlineMASBuffer, BalancedReplayBuffer
from src.drift_signals import DriftOracle, DriftDetector
from src.torch_modules import MLP, MAS

class Learner(object):
    def __init__(self, criterion, config):
        self.isTreeBased  = False
        self.criterion    = criterion
        self.config       = config
        self.features     = config["features"]
        self.target       = [ config["target"] ]
        self.taskCol      = config["taskcol"]
        self.modelType    = config["modeltype"]
        self.hyperParams  = config["hyperparams"]
        self.oracle       = DriftOracle( self.modelType, config["tasksignal"] ) if "tasksignal" in config.keys() else None
        
        if "MLP" in self.modelType:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
            
        if self.criterion == "MSE":
            self.criterion = torch.nn.MSELoss() if "MLP" in self.modelType else mean_squared_error
        else:
            self.criterion = criterion
    
    def scaleOnline(self, data, fit=True, adaptive=False):
        data  = data.copy()
        pdata = data.copy()
        if fit:
            self.scaler.partial_fit( data[self.features] )

        data = pd.DataFrame(self.scaler.transform(data[self.features].values), columns=self.features, index=data.index)
       
        data[self.target]  = pdata[self.target]
        data[self.taskCol] = pdata[self.taskCol]
        return data

    def loadData(self, data):
        if "MLP" in self.modelType:
            data       = data.copy().reset_index(drop=True)
            dataset    = PandasDataset(data, inputs=self.features, outputs=self.target, task_id=self.taskCol, use_pd_indices=True)
            dataLoader = torch.utils.data.DataLoader( dataset, batch_size=self.config["stepsize"])
            return dataLoader
        else:
            return data[ self.features + self.target ]
        
        
class MLPLearner(Learner):    
    def __init__(self, criterion, config, randomSeed=None, modelClass=None):
        super().__init__(criterion, config)
        self.modelClass = MLP if not modelClass else modelClass
        
        self.initModel()
        
    def initModel(self):
        self.model     = self.modelClass( len(self.features), self.hyperParams["hiddenlayers"], len(self.target) )
        self.optimizer = torch.optim.Adam( params=self.model.parameters(), lr=self.hyperParams["learningrate"] )
        
        self.model.to(self.device)
        
    def setLearningRate(self, learningRate):
        for paramaterGroups in self.optimizer.param_groups:
            paramaterGroups["lr"] = learningRate
        
    def trainOffline(self, data, config):
        self.setLearningRate( config["learningrate"] )
        
        [ self.train(data) for _ in tqdm( range( config["epochs"] ) ) ]

        self.setLearningRate(self.hyperParams["learningrate"])

    def forwardLoss(self, x, y, errors=False):
        output = self.model( x.float() )
        if len(y) == 1 and len(output) == 1:
            predict = [ output.to(torch.float32).item() ]
            true    = [ y.to(torch.float32).item() ]
        else:
            predict = output.to(torch.float32).squeeze().tolist()
            true    = y.to(torch.float32).squeeze().tolist()
        if errors:
            if len(y) == 1 and len(output) == 1:
                errors = [ abs(output.to(torch.float32).item() - y.to(torch.float32).item()) ]
            else:
                errors = [ abs(i - j) for i, j in zip(output.to(torch.float32).squeeze().tolist(), y.to(torch.float32).squeeze().tolist()) ]
            return errors, self.criterion( output.to(torch.float32), y.to(torch.float32) ), predict, true
        else:
            return self.criterion( output.to(torch.float32), y.to(torch.float32) ), predict, true
        
    def train(self, data, epochs=10):
        for e in range(epochs):
            for *_ , x, y in data:
                x, y = x.to(self.device), y.to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()
                
                loss, _, _ = self.forwardLoss(x, y)
    
                loss.backward()
                self.optimizer.step()
            
    def evaluate(self, data, returnErrors=False):
        evaluationLoss = 0
        errorList = []
        predictList = []
        trueList = []
        for *_ , x, y in data:
            x, y = x.to(self.device), y.to(self.device)
            self.model.eval()
            
            if returnErrors:
                errors, loss, predict, true = self.forwardLoss(x, y, errors=True)
                errorList    = [*errorList, *errors]
            else:
                loss, predict, true = self.forwardLoss(x, y)
            
            predictList = [*predictList, *predict]
            trueList    = [*trueList, *true]
            
            evaluationLoss += loss.item() * x.size(0)
        if returnErrors:
            return errorList, evaluationLoss / len(data.dataset), predictList, trueList
        else:
            return evaluationLoss / len(data.dataset), predictList, trueList

    
class ShallowLearner(Learner):
    def __init__(self, criterion, config, randomSeed, batchLearner=False):
        super().__init__(criterion, config)
        self.randomSeed = randomSeed
        self.initModel()

    def initModel(self):
        self.batchLearner = True if "batchlearner" in self.config.keys() else False
            
        if "HAT" in self.modelType:
            self.isTreeBased = True
            self.model = HoeffdingAdaptiveTreeRegressor( grace_period=int( self.hyperParams["graceperiod"] ),
                                                         split_confidence=float( self.hyperParams["split_confidence"] ),
                                                         learning_ratio_perceptron=float( self.hyperParams["learningratio"] ),
                                                         random_state=self.randomSeed)

        elif "VFDT" in self.modelType:
            self.isTreeBased = False
            self.model = HoeffdingTreeRegressor( grace_period=int( self.hyperParams["graceperiod"] ),
                                                 split_confidence=float( self.hyperParams["split_confidence"] ),
                                                 learning_ratio_perceptron=float( self.hyperParams["learningratio"] ),
                                                 random_state=self.randomSeed)

            
        elif "DTREE" in self.modelType:
            self.isTreeBased = True
            self.model = DecisionTreeRegressor( max_depth=self.hyperParams["maxdepth"],
                                                min_samples_split=self.hyperParams["minsplit"],
                                                min_samples_leaf=self.hyperParams["minleaf"],
                                                random_state=self.randomSeed)
        
    def train(self, data):
        x = data[self.features].to_numpy()
        y = data[ self.target[0] ].ravel()

        self.model.partial_fit(x, y) if not self.batchLearner else self.model.fit(x, y)
    
    def trainOffline(self, data, config):
        self.train(data)
    
    def evaluate(self, data, returnErrors=False):
        x        = data[self.features].to_numpy()
        y        = data[ self.target[0] ].ravel()
        predicts = self.model.predict(x)
        if returnErrors:
            return [ abs(i - j) for i, j in zip(y, predicts) ], self.criterion(predicts, y), predicts.tolist(), y.tolist()
        else:
            return self.criterion(predicts, y), predicts.tolist(), y.tolist()
    
    
class ModelAgnosticLearner(Learner):
    def __init__(self, criterion, config, randomSeed, batchLearner=False):
        super().__init__(criterion, config)
        self.randomSeed   = randomSeed
        self.config       = config
        self.modelClass   = MLPLearner if "MLP" in self.modelType else ShallowLearner
        self.model        = self.initNewModel()
        self.isTreeBased  = self.model.isTreeBased
        self.batchLearner = batchLearner
        self.scaler       = self.initScaler()
        self.drifted      = False
        self.buffer       = pd.DataFrame()
        self.warningState = False
        
    def checkForDrift(self, data, minBufferSize=0):
        loadData = data.copy()

        if self.drifted or self.driftDetector.isDrifted:
            self.drifted = True
            self.buffer = pd.concat( [self.buffer, data] ).reset_index(drop=True)
            if len(self.buffer) >= minBufferSize:
                print(data.index)
                self.adapt()
                loadData          = self.buffer
                self.buffer       = pd.DataFrame()
                self.warningState = False
                self.drifted      = False
                self.driftDetector.reset()
        elif self.warningState or self.driftDetector.isWarning:
            self.warningState = True
            self.buffer = pd.concat( [self.buffer, data] ).reset_index(drop=True)

        return loadData
        
    def initScaler(self):
        if self.config["scaler"] == "standard":
            scaler = StandardScaler()
        return scaler

    def initNewModel(self):
        return self.modelClass(self.criterion, self.config, self.randomSeed)
    
    
class WindowLearner(ModelAgnosticLearner):
    def __init__(self, criterion, config, randomSeed):
        super().__init__(criterion, config, randomSeed, batchLearner=True)
        self.windowSize   = self.config["hyperparams"]["windowsize"]
        self.windowData   = pd.DataFrame()
        
    def updateWindow(self, data):
        self.windowData = pd.concat( [self.windowData, data] ).reset_index(drop=True)
        if len(self.windowData) > self.windowSize:
            self.windowData = self.windowData.iloc[-1 * self.windowSize:, :]
    
    def train(self, data):
        self.model  = self.initNewModel()
        self.scaler = StandardScaler()
        
        self.updateWindow(data)
        
        trainData  = self.scaleOnline(self.windowData) if not self.isTreeBased else self.windowData
        loadedData = self.loadData(trainData)
        if "MLP" in self.model.modelType:
            self.model.train(loadedData, epochs=10)
        else:
            self.model.train(loadedData)
        
    def trainOffline(self, data, config):
        self.model  = self.initNewModel()
        self.scaler = StandardScaler()
        
        self.updateWindow(data)
        
        trainData  = self.scaleOnline(self.windowData) if not self.isTreeBased else self.windowData
        loadedData = self.loadData(trainData)
        self.model.trainOffline(loadedData, config)
        
    def evaluate(self, data):
        evaluateData = self.scaleOnline(data, fit=False) if not self.isTreeBased else data
        loadedData   = self.loadData(evaluateData)
        return self.model.evaluate(loadedData)

    
class IncrementalLearner(ModelAgnosticLearner):
    def __init__(self, criterion, config, driftOracle, randomSeed):
        super().__init__(criterion, config, randomSeed)
        self.oracle         = driftOracle
        self.reconstruction = config["reconstruction"] if "reconstruction" in config.keys() else False
        self.driftDetector  = DriftDetector(self.hyperParams["levels"], randomSeed) if not self.oracle and self.reconstruction else None
        self.adaptScaler  = { feature: preprocessing.AdaptiveStandardScaler(0.8) for feature in self.features }
    
    def adapt(self):
        self.scaler = self.initScaler()
        self.model  = self.initNewModel()
    
    def train(self, data, minBufferSize=30):
        data = data.copy()
        if self.reconstruction and self.oracle:
            driftDetected = self.oracle.isDrifted
            if driftDetected:
                self.drifted = True
            if self.drifted:
                self.buffer = pd.concat( [self.buffer, data] ).reset_index(drop=True)
                if len(self.buffer) >= minBufferSize:
                    self.scaler = self.initScaler()
                    self.model  = self.initNewModel()
                    data = self.buffer
                    self.buffer = pd.DataFrame()
                    self.drifted = False
                
        elif self.reconstruction:
            data = self.checkForDrift(data, minBufferSize=0)
            
        if self.reconstruction:
            scaledData = self.scaleOnline(data) if not self.isTreeBased else data
        else:
            scaledData = self.scaleOnline(data, adaptive=True) if not self.isTreeBased else data
            
        loadedData = self.loadData(scaledData)
        self.model.train(loadedData)
      
    def trainOffline(self, data, config):
        scaledData = self.scaleOnline(data) if not self.isTreeBased else data
            
        loadedData = self.loadData(scaledData)
        self.model.trainOffline(loadedData, config)
        
    def evaluate(self, data):
        if self.reconstruction:
            scaledData = self.scaleOnline(data, fit=False) if not self.isTreeBased else data
        else:
            scaledData = self.scaleOnline(data, fit=False, adaptive=True) if not self.isTreeBased else data
        loadedData = self.loadData(scaledData)
        
        if not self.oracle and self.reconstruction:
            errors, loss, predicts, true = self.model.evaluate(loadedData, returnErrors=True)
            self.driftDetector.update(errors)
            return loss, predicts, true
        else:
            return self.model.evaluate(loadedData)
                
    
class ModelRepository(ModelAgnosticLearner):
    def __init__(self, criterion, config, driftOracle, randomSeed):
        super().__init__(criterion, config, randomSeed)
        self.oracle        = driftOracle
        self.driftDetector = DriftDetector(self.hyperParams["levels"], randomSeed) if not self.oracle else None
        self.cloneInit     = config["cloneinit"] if "cloneinit" in config.keys() else False
        self.modelStore    = {}
        self.scalerStore   = {}
        self.trainingMode  = True
        
        if not self.oracle:
            self.setModel()
        
    def setModel(self):
        newKey = len(self.modelStore)
        self.modelStore[newKey]  = self.initNewModel()
        self.scalerStore[newKey] = self.initScaler()

        self.activeModel = self.modelStore[newKey]
        self.scaler = self.scalerStore[newKey] 
        
    def adapt(self, threshold=1.0):
        if len(self.modelStore) > 1:
            errors = {}
            for modelKey, model in self.modelStore.items():
                if not self.isTreeBased:
                    self.scaler = self.scalerStore[modelKey]
                    data = self.scaleOnline(self.buffer.copy(), fit=False)
                else:
                    data = self.buffer.copy()
                loadedData = self.loadData(data)
                _, predicts, true = model.evaluate(loadedData)
                
                errors[modelKey] = mean_absolute_error(predicts, true)
                
            minError = min( errors.values() )
            
            if minError <= threshold:
                self.activeModel = self.modelStore[ min(errors, key=errors.get) ]
                self.scaler      = self.scalerStore[ min(errors, key=errors.get) ]
            else:
                self.setModel()
        else:
            self.setModel()
        
    def checkRepository(self):
        task = self.oracle.currentTask
        if task in self.modelStore.keys():
            self.activeModel = self.modelStore[task]
            self.scaler      = self.scalerStore[task]

    def updateRepository(self, data, minBufferSize=30):
        loadedData = data.copy()
        task = self.oracle.currentTask
        
        if task not in self.modelStore.keys():
            self.buffer = pd.concat( [self.buffer, data] ).reset_index(drop=True)
            if not self.modelStore:
                self.modelStore[task]  = self.initNewModel()
                self.scalerStore[task] = self.initScaler()
                self.buffer = pd.DataFrame()
            elif len(self.buffer) >= minBufferSize:
                loadedData  = self.buffer
                self.buffer = pd.DataFrame()
                self.trainingMode = True
                self.modelStore[task]  = self.initNewModel()
                self.scalerStore[task] = self.initScaler()
            else:
                self.trainingMode = False
            
        if self.trainingMode:
            self.activeModel = self.modelStore[task]
            self.scaler      = self.scalerStore[task]
        return loadedData

    def trainOffline(self, data, config):
        self.updateRepository(data)
        scaledData = self.scaleOnline(data) if not self.isTreeBased else data.copy()
        loadedData = self.loadData(scaledData)
        self.activeModel.trainOffline(loadedData, config)

    def train(self, data):
        data = data.copy()
        if self.oracle:
            data = self.updateRepository(data)
        else:
            data = self.checkForDrift(data, minBufferSize=60)
        
        if not self.drifted and not self.warningState and self.trainingMode:
            scaledData = self.scaleOnline(data) if not self.isTreeBased else data.copy()
            loadedData = self.loadData(scaledData)
            self.activeModel.train(loadedData)
            
    def evaluate(self, data):
        if self.oracle:
            self.checkRepository()
            
        scaledData = self.scaleOnline(data, fit=False) if not self.isTreeBased else data.copy()
        loadedData = self.loadData(scaledData)
        
        if self.oracle:
            return self.activeModel.evaluate(loadedData)
        else:
            errors, loss, predicts, true = self.activeModel.evaluate(loadedData, returnErrors=True)
            self.driftDetector.update(errors)
            return loss, predicts, true


class ContinualLearner(MLPLearner):
    def __init__(self, criterion, config, model, driftOracle):
        super().__init__(criterion, config, modelClass=model)
        self.oracle      = driftOracle
        self.cloneInit   = config["cloneinit"] if "cloneinit" in config.keys() else False
        self.multiHeaded = config["multihead"]
        self.taskHeads   = None if self.multiHeaded else {1: 0}
        self.activeHead  = 0
        self.scaler      = StandardScaler()

    def checkHeads(self):
        task = self.oracle.currentTask
        if task in self.taskHeads.keys():
            self.activeHead = self.taskHeads[task]
    
    def updateHeads(self, data):
        task = self.oracle.currentTask
        
        if not self.taskHeads:
            self.taskHeads = {task: 0}
        elif task not in self.taskHeads.keys(): 
            self.model.add_head()
        
            self.optimizer       = torch.optim.Adam( params=self.model.parameters(), lr=self.hyperParams["learningrate"] )
            self.taskHeads[task] = len(self.taskHeads)
            
            if self.cloneInit:
                errors = {}
                for task, head in self.taskHeads.items():
                    self.activeHead  = head
                    errors[task], *_ = self.evaluate(data, False, False)
                    
                newHead   = self.model.layers[ "out{}".format( self.taskHeads[task]+1 ) ]
                cloneHead = self.model.layers[ "out{}".format( self.taskHeads[ min(errors, key=errors.get) ]+1 ) ]
                
                # newHead   = self.model.layers[ "out{}".format( self.taskHeads[task]+1 ) ] 
                # cloneHead = self.model.layers[ "out{}".format(  self.taskHeads[ dict( (v,k) for k,v in self.taskHeads.items() ).get(self.activeHead)+1 ] ) ]
                
                with torch.no_grad():
                    newHead.weight.copy_(cloneHead.weight)
                    newHead.bias.copy_(cloneHead.bias)

        self.activeHead = self.taskHeads[task]
        
    def forwardLoss(self, x, y, head=None):
        head   = self.activeHead if not head else head
        output = self.model( x.float(), head )
        
        if len(y) == 1 and len(output) == 1:
            predict = [ output.to(torch.float32).item() ]
            true    = [ y.to(torch.float32).item() ]
        else:
            predict = output.to(torch.float32).squeeze().tolist()
            true    = y.to(torch.float32).squeeze().tolist()

        return self.criterion( output.to(torch.float32), y.to(torch.float32) ), predict, true
        
        
    def trainOffline(self, data, config):
        scaledData = self.scaleOnline(data)
        loadedData = self.loadData(scaledData)
        super().trainOffline(loadedData, config)
        
    def train(self, data):
        scaledData = self.scaleOnline(data)
        loadedData = self.loadData(scaledData)
        super().train(loadedData)
    
    def evaluate(self, data, check=True, load=True):
        if load:
            scaledData = self.scaleOnline(data.copy(), fit=False)
            loadedData = self.loadData(scaledData)
        else:
            loadedData = data

        if check and self.multiHeaded:
            self.checkHeads()
        return super().evaluate(loadedData)   


class MASLearner(ContinualLearner):
    def __init__(self, criterion, config, driftOracle):
        super().__init__(criterion, config, MAS, driftOracle)
        self.buffer = OnlineMASBuffer( self.hyperParams["buffersize"] )
            
    def regularize(self):
        bufferData = torch.utils.data.DataLoader( self.buffer, batch_size=len(self.buffer) )
        
        self.model.update_omega(bufferData, task_id_dict=self.taskHeads, device=self.device)
        self.model.update_theta()
        self.buffer.clear()
        
    def updateHeads(self, data):
        previousHead  = self.activeHead
        previousHeads = self.taskHeads
        
        super().updateHeads(data)
        
        if self.activeHead is not previousHead and previousHeads:
            self.regularize() if self.activeHead not in previousHeads.items() else self.buffer.clear()
        
    def train(self, data, epochs=10):
        scaledData = self.scaleOnline(data.copy())
        loadedData = self.loadData(scaledData)
        
        if self.multiHeaded:
            self.updateHeads(loadedData)

        for _ in range(epochs):
            for *_ , x, y in loadedData:
                x, y = x.to(self.device), y.to(self.device)
                
                self.model.train()
                self.optimizer.zero_grad()
                
                loss, _, _  = self.forwardLoss(x, y)
                omegaLoss   = loss + ( self.hyperParams["lamb"] * self.model.compute_omega_loss() )
    
                omegaLoss.backward()
                self.optimizer.step()
        
        self.buffer.update(loadedData)
        
        
class ReplayLearner(ContinualLearner):
    def __init__(self, criterion, config, driftOracle):
        super().__init__(criterion, config, MLP, driftOracle)
        self.buffer = BalancedReplayBuffer( self.hyperParams["buffersize"] )
       
    def train(self, data, epochs=10):
        scaledData = self.scaleOnline(data.copy())
        loadedData = self.loadData(scaledData)
        
        task = self.oracle.currentTask
        
        if self.multiHeaded:
            self.updateHeads(loadedData)
        
        for _ in range(epochs):
            for t, x, y in loadedData:
                self.model.train()
                self.optimizer.zero_grad()

                if (len(self.taskHeads) > 1 or not self.multiHeaded):
                    tB, xB, yB = self.buffer.sample(x.size(0), task)
                    t, x, y    = ( torch.cat((t, tB)), torch.cat((x, xB)), torch.cat((y, yB)) )
    
                t     = tuple( self.taskHeads[ int(sample) ] for sample in t )
                x, y  = tuple( v.to(self.device).float() for v in (x, y) )
                heads = sorted( set(t) )
                out   = self.model(x, head=heads)
                out   = torch.stack( [ out[heads.index(t_id)][i] for i, t_id in enumerate(t) ], dim=0 ) if len(out) > 1 else out[0]
                loss  = self.criterion(out, y)
                
                loss.backward()
                self.optimizer.step()
            
        self.buffer.update(loadedData, task) 