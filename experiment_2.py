import json

from src.experiment import ExperimentEvaluator


if __name__ == '__main__':
    with open("config/mlp_config_exp2.json") as json_file:
        mlpConfig = json.load(json_file)
    
    with open("config/tree_config_exp2.json") as json_file:
        treeConfig = json.load(json_file)
    
    def runExp(n, configs, dataFiles, saveFiles):
        for i, (key, config) in enumerate(configs.items()):
            experimentConfig = {
                "data":           None,
                "experimentname": "experiment-2: Software Fault",
                "randomseed": 42,
                "scenarioconfig": {
                    "features":   ["interArrival", "dimmer", "servers", "queue"],
                    "target":     "response",
                    "scaler":     "standard",
                    "taskcol":    "concept",
                    "multihead":  True,
                    "cloneinit":  True,
                    "oracle":     True,
                    "stepsize":   10
                    },
                "learnerconfig":  config,
                "criterion":      "MSE",
                "stepsize":       10,
                "scaleall":       False
                }
            
            evaluator = ExperimentEvaluator([dataFiles[i]], experimentConfig)
            evaluator.runNIterations(n, saveFileName=saveFiles[i])
        
    
    runExp(n         = 30,
           configs   = {"mlp": mlpConfig, "tree": treeConfig},
           dataFiles = ["base-t1.csv",
                        "base-t2.csv",
                        "base-t3.csv"],
           saveFiles = ["e2-base-t1.csv",
                        "e2-base-t2.csv",
                        "e2-base-t3.csv"])
    
    runExp(n         = 30,
           configs   = {"tree": treeConfig},
           dataFiles = ["base-low-t1.csv",
                        "base-low-t2.csv",
                        "base-low-t3.csv"],
           saveFiles = ["e2-base-low-t1.csv",
                        "e2-base-low-t2.csv",
                        "e2-base-low-t3.csv"])
        
    runExp(n         = 30,
           configs   = {"mlp": mlpConfig},
           dataFiles = ["seq-t1.csv",
                        "seq-t2.csv",
                        "seq-t3.csv"],
           saveFiles = ["e2-seq-t1.csv",
                        "e2-seq-t2.csv",
                        "e2-seq-t3.csv"])

    