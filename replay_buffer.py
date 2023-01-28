from collections import deque, namedtuple
import random
import torch 
import numpy as np
import random

Transition = namedtuple('Transition', ('state', 'action', 'next_state', 'reward'))


class Memory(object):
    def __init__(self, capacity):
        self.capacity = capacity

class ReplayMemory(Memory):
    def __init__(self, capacity):
        super(ReplayMemory, self).__init__(capacity)
        self.memory = deque([], maxlen=capacity)

    def push(self, *args):
        self.memory.append(Transition(*args))

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)

class PrioritizedReplayMemory(Memory):
    def __init__(self, capacity, state_size=25, action_size=5):
        super(PrioritizedReplayMemory, self).__init__(capacity)
        self.tree = SumTree(capacity)
        self.memory = deque([], maxlen=capacity)
        self.eps = 1e-2
        self.alpha = 0.7
        self.beta = 0.4
        self.max_priority = self.eps

        self.count = 0
        self.real_size = 0
        self.size = capacity

    def push(self, *args):
        self.memory.append(Transition(*args))
        self.tree.add(self.max_priority, self.count)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def sample(self, batch_size):
        tree_idxs = []
        batch = []
        P = torch.empty(batch_size, 1, dtype=torch.float)

        segment = self.tree.total / batch_size
        for i in range(batch_size):
            a, b = segment * i, segment * (i + 1)
            cumsum = random.uniform(a, b)
            tree_idx, priority, sample_idx = self.tree.get(cumsum)
            P[i] = priority
            tree_idxs.append(tree_idx)
            batch.append(self.memory[sample_idx])

        probs = P / self.tree.total

        weights = (self.real_size * probs) ** -self.beta
        weights = weights / weights.max()

        return batch, weights, tree_idxs

    
    def update(self, idxs, td_errors):
        if isinstance(td_errors, torch.Tensor):
            td_errors = td_errors.detach().cpu().numpy()

        for idx, td in zip(idxs, td_errors):
            td = (td[0] + self.eps) ** self.alpha

            self.tree.update(idx, td)
            self.max_priority = max(self.max_priority, td)

    def __len__(self):
        return len(self.memory)


class SumTree:
    def __init__(self, size):
        self.count = 0
        self.real_size = 0
        self.nodes = [0] * (2 * size - 1)
        self.data = [None] * size
        self.size = size

    @property
    def total(self):
        return self.nodes[0]

    def update(self, data_idx, value):
        idx = data_idx + self.size - 1 
        change = value - self.nodes[idx]

        self.nodes[idx] = value

        parent = (idx - 1) // 2
        while parent >= 0:
            self.nodes[parent] += change
            parent = (parent - 1) // 2

    def add(self, value, data):
        self.data[self.count] = data
        self.update(self.count, value)
        self.count = (self.count + 1) % self.size
        self.real_size = min(self.size, self.real_size + 1)

    def get(self, cumsum):
        assert cumsum <= self.total

        idx = 0
        while 2 * idx + 1 < len(self.nodes):
            left, right = 2*idx + 1, 2*idx + 2

            if cumsum <= self.nodes[left]:
                idx = left
            else:
                idx = right
                cumsum = cumsum - self.nodes[left]

        data_idx = idx - self.size + 1

        return data_idx, self.nodes[idx], self.data[data_idx]