import random
from collections import namedtuple

Transition = namedtuple('Transition',
                        ('state', 'reward', 'done', 'action', 'next_state'))


class Memory:
    def __init__(self, memory_size):
        self.memory_size = memory_size
        self.memory = []

    def add(self, *transition):
        self.memory.append(Transition(*transition))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        assert len(self.memory) <= self.memory_size

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)
