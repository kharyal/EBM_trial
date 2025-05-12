'''
sum tree for prioritized experience replay
'''

class SumTree:
    def __init__(self, capacity):
        self.capacity = capacity
        self.tree = [0] * (2 * capacity - 1)
        self.data = [0] * capacity
        self.index = 0
        self.num_elements = 0

    def add(self, priority, data):
        priority_index = self.index + self.capacity - 1
        self.data[self.index] = data
        self.update(priority, priority_index)
        self.index = (self.index + 1) % self.capacity
        self.num_elements = min(self.num_elements + 1, self.capacity)
    
    def update(self, priority, priority_index):
        diff = priority - self.tree[priority_index]
        self.tree[priority_index] = priority
        while priority_index != 0:
            priority_index = (priority_index - 1) // 2
            self.tree[priority_index] += diff

    def update_priority_from_data_index(self, data_index, priority):
        priority_index = data_index + self.capacity - 1
        self.update(priority, priority_index)

    def get(self, value):
        '''
            gets the data point where the cumulative priority is greater than value
        '''
        assert value >= 0, "value must be greater than or equal to 0"
        assert value <= self.tree[0], "value must be less than or equal to the sum of all priorities"

        tree_ind = 0

        left = 2 * tree_ind + 1
        right = 2 * tree_ind + 2
        while left < len(self.tree):
            if value <= self.tree[left]:
                tree_ind = left
            else:
                value -= self.tree[left]
                tree_ind = right
            left = 2 * tree_ind + 1
            right = 2 * tree_ind + 2

        data_index = tree_ind - self.capacity + 1
        return data_index, self.tree[tree_ind], self.data[data_index]
    
    @property
    def total(self):
        return self.tree[0]
    
    @property
    def size(self):
        return self.num_elements
    
    def render_tree(self):
        '''
            print tree with different levels
        '''
        levels = []
        level = 0
        while True:
            start = 2 ** level - 1
            end = 2 ** (level + 1) - 1
            if start >= len(self.tree):
                break
            levels.append(self.tree[start:end])
            level += 1
        
        for i, level in enumerate(levels):
            print(f"Level {i}: ", end="")
            for j, node in enumerate(level):
                print(f"{node:.2f}", end=" ")
            print()
            
    
    