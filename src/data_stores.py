import torch
import numpy as np
import random
from collections.abc import Iterable

class PandasDataset(torch.utils.data.Dataset):
    def __init__(self, data, inputs: Iterable, outputs: Iterable, task_id=None, use_pd_indices=False):
        super(PandasDataset, self).__init__()

        self.use_pd_indices = use_pd_indices
        self.indices = list(data.index)
        self.samples = []

        for _, row in data.iterrows():
            x = torch.tensor([row[i] for i in inputs])
            y = torch.tensor([row[i] for i in outputs])

            if(task_id is not None):
                self.samples.append( (row[task_id], x, y) )
            else:
                self.samples.append( (x,y) )
        
    def __len__(self):        
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[ self.indices.index(idx) ] if(self.use_pd_indices) else self.samples[idx]


class Buffer(torch.utils.data.Dataset):
    def __init__(self, bufferSize):
        super().__init__()
        self.bufferSize = bufferSize
        self.samples    = []
        
    def __len__(self):        
        return len(self.samples)

    def __getitem__(self, idx):
        return self.samples[idx]
    
    
class OnlineMASBuffer(Buffer):
    '''
    Online buffer for MAS, updated based on data within a sliding window.
    Buffer is cleared everytime knowledge is consolidated.
    '''
    
    def __init__(self, bufferSize):
        super().__init__(bufferSize)
    
    def update(self, data):
        for t, x, y in data:
            for i in range( len(x) ): # iterate through samples in batch
                if self.bufferSize < len(self):
                    self.samples.pop(0)
                self.samples.append( ( t[i], x[i], y[i] ) )
    
    def clear(self):
        self.samples = []

class BalancedReplayBuffer(Buffer):
    def __init__(self, bufferSize):
        super().__init__(bufferSize)
        self.taskBuffers   = {}
        self.subBufferSize = self.bufferSize
        
    def update(self, data, task):
        if task not in self.taskBuffers:
            self.subBufferSize = int(self.bufferSize / (len(self.taskBuffers) + 1))
            for taskKey in self.taskBuffers.keys():
                self.taskBuffers[taskKey].setSize(self.subBufferSize)
                if len(self.taskBuffers[taskKey]) > self.subBufferSize:
                    self.taskBuffers[taskKey].subset(self.subBufferSize)
                    
            self.taskBuffers[task] = ReplayBuffer(self.subBufferSize)
        
        self.taskBuffers[task].update(data, task)
    
    def sample(self, size, task):
        sampleDictionary = { k: v for k, v in self.taskBuffers.items() if k != task }
        randomDataSet    = random.choice(list(sampleDictionary.values()))
        return randomDataSet.sample(size, task)

    
class ReplayBuffer(Buffer):
    '''
    Replay buffer, updated through reservoir sampling.
    '''
    
    def __init__(self, bufferSize):
        super().__init__(bufferSize)
        self.observedSamples = 0

    def setSize(self, size):
        self.bufferSize = size

    def sample(self, size, task):
        self.randomSamples = [ self.samples[i] for i in np.random.choice(len(self), size, replace=False) ]
        return next( iter( torch.utils.data.DataLoader(self, batch_size=size) ) )

    def update(self, data, task):
        j = 0
        for t, x, y in data:
            for i in range( len(x) ): # iterate through samples in batch
                sample = ( t[i], x[i], y[i] )
                if self.bufferSize > len(self):
                    self.samples.append(sample)
                else:
                    rand = np.random.randint(0, self.observedSamples + j)
                    if rand < self.bufferSize:
                        self.samples[rand] = sample
                j += 1
        self.observedSamples += j
        
    def subset(self, size):
        self.samples = [ self.samples[i] for i in np.random.choice(len(self), size, replace=False) ]
        
    def __getitem__(self, idx):
        return self.randomSamples[idx]
    
    
    
    
    