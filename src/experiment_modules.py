from sklearn.metrics import mean_absolute_error


class ExperimentModule(object):
    def __init__(self, manager):
        self.learners = manager.learners
        self.features = manager.features
        self.target   = manager.target
        self.task     = manager.task
        self.scaleAll = manager.scaleAll
        
        
class Trainer(ExperimentModule):
    def __init__(self, manager):
        super().__init__(manager)
    
    def preTrainModels(self, originalData, preTrainConfig):
        for key, learner in self.learners.items():
            learner.trainOffline(originalData, preTrainConfig)

    def updateModels(self, originalData):
        for key, learner in self.learners.items():
            learner.train(originalData)


class Evaluator(ExperimentModule):
    def __init__(self, manager, stepSize):
        super().__init__(manager)
        self.criterion        = manager.criterion
        self.learningSteps    = []
        self.experimentValues = { model: { key: [] for key in ["concept", "true_value", "predictions"] } \
                                for model in self.learners.keys() }
        self.onlineMeasures   = [f"exponential_{self.criterion}", f"running_{self.criterion}", f"cumulative_{self.criterion}"]
        self.evaluations      = { model: { key: [] for key in [self.criterion] + self.onlineMeasures } \
                                for model in self.learners.keys() }
        self.valueStore       = { model: { key: 0 for key in self.onlineMeasures } \
                                for model in self.learners.keys() }
        self.stepSize         = stepSize
        self.nSteps           = 0
       
    def evaluateModels(self, originalData):        
        self.learningSteps.append(self.nSteps)
        self.nSteps += 1
        for key, learner in self.learners.items():
            loss, predicts, true = learner.evaluate(originalData)
            predicts = [ p if p > 0 else 0 for p in predicts ]
            loss = mean_absolute_error(predicts, true)

            self.experimentValues[key]["concept"].extend(originalData["concept"].tolist())
            self.experimentValues[key]["true_value"].extend(true)
            self.experimentValues[key]["predictions"].extend(predicts)
            
            self.valueStore[key][f"exponential_{self.criterion}"] = 0.05 * loss + 0.95 * self.valueStore[key][f"exponential_{self.criterion}"] if self.nSteps != 1 else loss
            self.valueStore[key][f"running_{self.criterion}"]    += ( loss - self.valueStore[key][f"running_{self.criterion}"] ) / self.nSteps if self.nSteps != 1 else loss
            self.valueStore[key][f"cumulative_{self.criterion}"] += loss

            self.evaluations[key][self.criterion].append(loss)
            for measure in self.onlineMeasures:
                self.evaluations[key][measure].append( self.valueStore[key][measure] )
                