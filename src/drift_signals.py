from skmultiflow.drift_detection import ADWIN


class DriftDetector(object):
    def __init__(self, levels, randomSeed):
        self.isDrifted = False
        self.isWarning = False
        
        self.warningMonitor = ADWIN(levels["warning"])
        self.driftMonitor   = ADWIN(levels["drift"])

    def update(self, errors):
        self.isWarning = False
        self.isDrifted = False
        for error in errors:
            self.warningMonitor.add_element(error)
            self.driftMonitor.add_element(error)
            if self.warningMonitor.detected_change():
                self.isWarning = True
            if self.driftMonitor.detected_change():
                self.isDrifted = True
            
    def reset(self):
        self.warningMonitor.reset()
        self.driftMonitor.reset()


class DriftOracle(object):
    def __init__(self, learnerType):
        self.learnerType  = learnerType
        self.isDrifted    = False
        self.currentTask  = None
    
    def update(self, task):
        if task != self.currentTask:
            self.isDrifted   = True
            self.currentTask = task
        else:
            self.isDrifted = False
    
    def oracle(self, data):
        if "MLP" in self.learnerType:
            task, *_ = next( iter(data) )
            task     = int( task.data[0] )
        else:
            task = data["concept"].iat[0]
        return task
        
    def checkForDrift(self, data, labelSignal=True):
        if self.taskLabel == "oracle":
            task = self.oracle(data)
            if not labelSignal:
                driftDetected = (task != self.lastTask)
                self.lastTask = task
                return driftDetected
            else: 
                return task