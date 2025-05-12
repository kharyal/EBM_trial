'''
base class for all buffers, uses 
'''
import random

class Buffer:
    def __init__(self, capacity):
        self.replay = []
        self.capacity = capacity
        self.position = 0

    def push(self, experience):
        if len(self.replay) < self.capacity:
            self.replay.append(experience)
        else:
            self.replay[self.position] = experience
        self.position = (self.position + 1) % self.capacity

    def sample(self, batch_size):
        return random.sample(self.replay, batch_size)
    
    def __len__(self):
        return len(self.replay)
    