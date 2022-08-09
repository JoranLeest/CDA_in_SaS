import pandas as pd
import numpy as np
from math import sqrt
from sklearn.metrics import mean_absolute_error


def process(learners, file, saveFile=True):
    mainString = f"data/raw/{file}"
    subStrings = ["1", "2", "3"]
    for learner in learners:
        data = pd.DataFrame()
        for subString in subStrings:
            subFile = pd.read_csv(mainString + "-" + learner + "t" + subString + ".csv")
            subFile["trace"] = subString
            data = pd.concat([data, subFile]).reset_index(drop=True)

        data["MAE"] = [ mean_absolute_error([i], [j]) for i, j in zip(data["true_value"], data["predictions"]) ]
        
        if saveFile:
            data.to_csv(f"data/{file}-{learner}.csv", index=False)
        else:
            return data
    
    
def lateXTable(file, learners, segments, saveFile=False, y="MAE"):
    data = { learner: pd.read_csv(f"results/{file}-{learner}.csv") for learner in learners }
    
    learnersNumbers = {}
    for i, (learner, learnerData) in enumerate(data.items()): 
        learnerNumbers = { i: [] for i in ["period", "model", "trace", "run", y] }
        for j, segment in enumerate(segments):
            segmentData = learnerData[(learnerData.time >= segment[0]) & (learnerData.time < segment[1])]
            for model in segmentData.model.unique():
                for trace in segmentData.trace.unique():
                    for run in segmentData.run.unique():
                        runData = segmentData[(segmentData.model == model) & (segmentData.run == run) & (segmentData.trace == trace)]

                        learnerNumbers["period"].append(f"{segment[0]}-{segment[1]}")
                        learnerNumbers["model"].append(model)
                        learnerNumbers["trace"].append(trace)
                        learnerNumbers["run"].append(run)
                        learnerNumbers[y].append(runData[y].mean())
        
        learnersNumbers[learner] = pd.DataFrame(learnerNumbers)
    
    learnerTables = {}
    for learner, learnerData in learnersNumbers.items():
        metaData     = {"period": [], "trace": []}
        learnerTable = { **metaData, **{ model: [] for model in learnerData.model.unique() } }
        for period in learnerData.period.unique():
            for trace in learnerData.trace.unique():
                learnerTable["period"].append(period)
                learnerTable["trace"].append(trace)
                for model in learnerData.model.unique():
                    runMeans = []
                    for run in learnerData.run.unique():
                        runData = learnerData[(learnerData.model == model) & (learnerData.run == run) & (learnerData.trace == trace) & (learnerData.period == period)]
                        runMeans.append(runData[y].mean())
                        
                    median   = round(np.median(np.array(runMeans)), 2)
                    q75, q25 = np.percentile(np.array(runMeans), [75, 25])
                    iqr      = round(q75 - q25, 2)
                    
                    learnerTable[model].append(f"{median}({iqr})")
        
        learnerTables[learner] = pd.DataFrame(learnerTable)
        
    for learner, table in learnerTables.items():
        print(f"-----{learner}------")
        print(table.to_latex(index=False))
        print("--------------------")
        
    if saveFile:
        for learner, table in learnerTables.items():
            table.to_csv(f"results/tables/{file}-{learner}-table.csv", index=False)
    else:
        return learnerTables
    
    
if __name__ == '__main__':
    process(learners=["tree", "mlp"], file="e1-base")
    process(learners=["tree", "mlp"], file="e1-trans")
    process(learners=["tree", "mlp"], file="e2-base")
    process(learners=["tree"], file="e2-base-low")
    process(learners=["mlp"], file="e2-seq")
    
    
    file     = "e1-base"
    learners = ["tree", "mlp"]
    segments = [ [1200, 1800], [1800, 2400], [2400, 3000], [3000, 3600] ]
    tables = lateXTable(file=file,
                        learners=learners, 
                        segments=segments,
                        saveFile=True)
    
    file     = "e1-trans"
    learners = ["tree", "mlp"]
    segments = [ [1200, 1800], [1800, 2400], [2400, 3000], [3000, 3600] ]
    tables = lateXTable(file=file,
                        learners=learners, 
                        segments=segments,
                        saveFile=True)

    file     = "e2-base"
    learners = ["tree", "mlp"]
    segments = [ [1200, 1800], [1800, 2400], [2400, 3000], [3000, 3600] ]
    tables = lateXTable(file=file,
                        learners=learners, 
                        segments=segments,
                        saveFile=True)
    
    file     = "e2-base-low"
    learners = ["tree"]
    segments = [ [1200, 1800], [1800, 2400], [2400, 3000], [3000, 3600] ]
    tables = lateXTable(file=file,
                        learners=learners, 
                        segments=segments,
                        saveFile=True)
    
    file     = "e2-seq"
    learners = ["mlp"]
    segments = [ [1200, 2400], [2400, 3000], [3000, 4200], [4800, 6000], [6000, 6600] ]
    tables = lateXTable(file=file,
                        learners=learners, 
                        segments=segments,
                        saveFile=True)
    
    
    