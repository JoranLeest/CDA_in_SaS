import numpy as np 
import random
import torch

from src.learners import *


class ModelManager(object):
    def __init__(self, config, driftOracle):
        self.criterion     = config["criterion"]
        self.scenario      = config["scenarioconfig"]
        self.learnerConfig = config["learnerconfig"]
        self.features      = self.scenario["features"]
        self.target        = self.scenario["target"]
        self.task          = self.scenario["taskcol"]
        self.scaleAll      = config["scaleall"]
        self.learners      = {}
        self.randomSeed    = config["randomseed"] if "randomseed" in config.keys() else random.randint(0, 1000)

        for learner, hyperParams in self.learnerConfig["learners"].items():
            self.setSeed()
            learnerConfig = { **{"modeltype": learner}, **self.scenario, **{"hyperparams": hyperParams} }
            if learner == "MASMLP":
                self.learners["MASMLP"] = MASLearner(self.criterion, learnerConfig, driftOracle)
            elif learner == "replayMLP":
                self.learners["replayMLP"] = ReplayLearner(self.criterion, learnerConfig, driftOracle)
            else:
                self.learners[learner] = ShallowLearner(self.criterion, learnerConfig, self.randomSeed)
        
        for learner, modelConfig in self.learnerConfig["agnosticlearners"].items():
            for model, hyperParams in modelConfig.items():
                self.setSeed()
                learnerConfig = { **{"modeltype": model}, **self.scenario, **{"hyperparams": hyperParams} }
                if learner == "repository":
                    self.learners[f"{learner}-{model}"] = ModelRepository(self.criterion, learnerConfig, driftOracle, self.randomSeed)
                elif learner == "window":
                    learnerConfig["batchlearner"] = True
                    self.learners[f"{learner}-{model}"] = WindowLearner(self.criterion, learnerConfig, self.randomSeed)
                else:
                    if learner == "reconstruction":
                        learnerConfig = { **{"reconstruction": True}, **learnerConfig }
                    self.learners[f"{learner}-{model}"] = IncrementalLearner(self.criterion, learnerConfig, driftOracle, self.randomSeed)
    
    def setSeed(self):
        np.random.seed(self.randomSeed)
        random.seed(self.randomSeed)
        torch.manual_seed(self.randomSeed)
        torch.cuda.manual_seed(self.randomSeed)
        
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark     = False
    
    def getlearner(self, idx=None):
        if idx:
            return self.learner[idx]
        else:
            return self.learners if len(self.learners) > 1 else self.learners[0]