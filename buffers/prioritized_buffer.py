'''
    Prioritized Buffer
'''

from buffers.sum_tree import SumTree
from buffers.buffer import Buffer
import random
import numpy as np

class PrioritizedBuffer(Buffer):
    def __init__(self, capacity):
        super().__init__(capacity)
        self.tree = SumTree(capacity)
        self.epsilon = 0.01
        self.alpha = 0.6
        self.beta = 0.4
        self.beta_increment_per_sampling = 0.00001
        self.max_priority = self.epsilon
        self.max_td_error = 10.0
        
    def push(self, experience):
        position = self.position
        assert self.tree.index == position, "index must be equal to position"
        self.tree.add(self.max_priority, position)
        super().push(experience)

    def sample(self, batch_size):
        batch_indices = []
        priorities = np.zeros((batch_size,), dtype=np.float32)
        batch_experiences = []
        segment = self.tree.total / batch_size
        self.beta = min(1.0, self.beta + self.beta_increment_per_sampling)
        
        for i in range(batch_size):
            start = i * segment
            end = (i + 1) * segment
            value = random.uniform(start, end)
            data_indx, priority, buffer_indx = self.tree.get(value)
            batch_indices.append(data_indx)
            batch_experiences.append(self.replay[buffer_indx])
            priorities[i] = priority

        sampling_probabilities = priorities / self.tree.total
        is_weight = np.power(self.tree.size * sampling_probabilities, -self.beta)
        is_weight /= is_weight.max()
        is_weight = is_weight.tolist()

        return batch_experiences, batch_indices, is_weight
    
    def update_priority(self, batch_indices, priorities):
        # print(batch_indices)
        # print(priorities)
        for i, priority in zip(batch_indices, priorities):
            if priority > self.max_td_error:
                priority = self.max_td_error
            priority += self.epsilon
            self.tree.update_priority_from_data_index(i, priority)
            self.max_priority = max(self.max_priority, priority)