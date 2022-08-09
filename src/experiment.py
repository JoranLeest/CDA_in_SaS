import numpy as np 
import pandas as pd
import seaborn as sns

from src.drift_signals import *
from src.experiment_modules import *
from src.model_manager import ModelManager


class Experiment(object):
    def __init__(self, config):
        self.data         = pd.read_csv( config["data"] )
        self.randomSeed   = config["randomseed"] if "randomseed" in config.keys() else None
        self.originalData = self.data.copy()
        self.name         = config["experimentname"]
        self.criterion    = config["criterion"]
        self.oracle       = DriftOracle("Nan") if config["scenarioconfig"]["oracle"] else None
        self.modelManager = ModelManager(config, self.oracle)
        self.evaluator    = Evaluator(self.modelManager, config["stepsize"])
        self.trainer      = Trainer(self.modelManager)
        self.features     = config["scenarioconfig"]["features"]
        self.target       = config["scenarioconfig"]["target"]
        self.taskCol      = config["scenarioconfig"]["taskcol"]
        self.stepSize     = config["stepsize"]

        if not self.oracle:
            self.data["concept"] = 1
            
    
    def run(self):
        self.data = self.data[ self.features + [self.target] + [self.taskCol] ]
        
        increments = [ np.arange(step, step + self.stepSize) \
                       for step in np.arange(self.data.index.min(), self.data.index.max(), self.stepSize) ]
        for i, incrementIDs in enumerate(increments):
            incrementData = self.data[ self.data.index.isin(incrementIDs) ]
            
            if self.oracle:
                self.oracle.update( incrementData[self.taskCol].iat[0] )

            if i > 0:
                self.evaluator.evaluateModels(incrementData)
                
            self.trainer.updateModels(incrementData)

    def getResults(self):
        results = pd.DataFrame()
        for model, modelResults in self.evaluator.evaluations.items():
            df = pd.DataFrame(modelResults)
            
            df["time"]    = [ (i * self.stepSize) + (self.originalData.time.min() + self.stepSize) for i in self.evaluator.learningSteps ]
            df["model"]   = model
            df["concept"] = df.time.apply( lambda x: self.originalData.loc[self.originalData.time == x, "concept"].values[0] )
            df["segment"] = df.time.apply( lambda x: self.originalData.loc[self.originalData.time == x, "segment"].values[0] )
            
            results = pd.concat( [results, df] ).reset_index(drop=True)
        return pd.DataFrame( results )

    def getExperimentValues(self):
        results = pd.DataFrame()
        for model, modelResults in self.evaluator.experimentValues.items():
            df = pd.DataFrame(modelResults)
            df["model"] = model
            df["time"]  = [ i + (2*self.stepSize) for i in range(len(df)) ]
            results = pd.concat( [results, df] ).reset_index(drop=True)
        return pd.DataFrame( results )

    def getCumulativePerformance(self):
        return { model: modelResults[f"cumulative_{self.criterion}"][-1] for model, modelResults in self.evaluator.evaluations.items() }


class ExperimentEvaluator(object):
    def __init__(self, files=None, config=None):
        self.files   = files
        self.config  = config
        self.results = pd.DataFrame()
        self.values  = pd.DataFrame()
        
    def runNIterations(self, n, saveFileName=None):            
        for file in self.files: 
            for i in range(n):
                self.config["data"] = file
                self.config["randomseed"] += i
                experiment = Experiment(self.config)
                self.name = experiment.name
                experiment.run()
                experimentResults = experiment.getResults()
                experimentValues  = experiment.getExperimentValues()
                experimentValues["run"]  = i
                experimentResults["run"] = i
                self.results = pd.concat( [self.results, experimentResults] ).reset_index(drop=True)
                self.values  = pd.concat( [self.values, experimentValues] ).reset_index(drop=True)
                
        if saveFileName:
            self.values.to_csv(f"results/{saveFileName}.csv", index=False)
        else:
            return self.results
    
    def setData(self, data, name):
        self.name    = name
        self.results = data