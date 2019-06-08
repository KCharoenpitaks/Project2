from collections import deque
import numpy as np

class Memory():
    def __init__(self, max_size=10):
        self.buffer = deque(maxlen=max_size)
        self.count = 0

    
    def add(self, experience):
        self.buffer.append(experience)
        self.count += 1 

            
    def sample(self, batch_size=5):
        if self.count<batch_size:
            return None#[self.buffer[0]]
        else:
            idx = np.random.choice(np.arange(len(self.buffer)), 
                                   size=batch_size, 
                                   replace=False)
            return [self.buffer[ii] for ii in idx]
    
    def length(self):
        # Return number of epsiodes saved in memory
        return self.count #len(self.buffer)
    
