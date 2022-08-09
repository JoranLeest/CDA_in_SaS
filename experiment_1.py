import json

from src.experiment import ExperimentEvaluator


if __name__ == '__main__':
    # with open("config/mlp_config_exp1.json") as json_file:
    #     mlpConfig = json.load(json_file)
    
    # with open("config/tree_config_exp1.json") as json_file:
    #     treeConfig = json.load(json_file)
    
    # def runExp(n, configs, dataFiles, saveFiles):
    #     for i, (key, config) in enumerate(configs.items()):
    #         experimentConfig = {
    #             "data":           None,
    #             "experimentname": "experiment-1: Cloud Interference",
    #             "randomseed": 42,
    #             "scenarioconfig": {
    #                 "features":   ["interArrival", "dimmer", "servers", "queue"],
    #                 "target":     "response",
    #                 "scaler":     "standard",
    #                 "taskcol":    "concept",
    #                 "multihead":  True,
    #                 "cloneinit":  True,
    #                 "oracle":     False,
    #                 "stepsize":   10
    #                 },
    #             "learnerconfig":  config,
    #             "criterion":      "MSE",
    #             "stepsize":       10,
    #             "scaleall":       False
    #             }
            
    #         evaluator = ExperimentEvaluator([dataFiles[i]], experimentConfig)
    #         evaluator.runNIterations(n, saveFileName=saveFiles[i])
        
    
    # runExp(n         = 30,
    #        configs   = {"mlp": mlpConfig, "tree": treeConfig},
    #        dataFiles = ["base-t1.csv",
    #                     "base-t2.csv",
    #                     "base-t3.csv"],
    #        saveFiles = ["e1-base-t1.csv",
    #                     "e1-base-t2.csv",
    #                     "e1-base-t3.csv"])
    
    # runExp(n         = 30,
    #        configs   = {"mlp": mlpConfig, "tree": treeConfig},
    #        dataFiles = ["trans-t1.csv",
    #                     "trans-t2.csv",
    #                     "trans-t3.csv"],
    #        saveFiles = ["e1-trans-t1.csv",
    #                     "e1-trans-t2.csv",
    #                     "e1-trans-t3.csv"])
    
    
    # d = pd.read_csv("results/e1-base-mlp.csv")
    
    import pandas as pd
    def clean(file):
        data = pd.read_csv(f"results/{file}.csv")
        # data = data[["model", "time", "trace", "run", "MAE"]]
        
        data["trace"].replace({"Bursts": 1, "Periodic": 2, "Volatile": 3}, inplace=True)
        data.to_csv(f"results/{file}.csv", index=False)
        return data
    
    files = [
        "e1-base-mlp",
        "e1-base-tree",
        "e1-trans-mlp",
        "e1-trans-tree",
        "e2-base-low-tree",
        "e2-base-mlp",
        "e2-base-tree",
        "e2-seq-mlp"
        ]
    
    for f in files:
        d = clean(f)

    
    
    
    
    
    