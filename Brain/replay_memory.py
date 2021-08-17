import random
from collections import namedtuple


class Memory:
    def __init__(self, memory_size, seed):
        self.memory_size = memory_size
        self.memory = []
        self.Transition = namedtuple('Transition', ('state', 'z', 'done', 'action', 'next_state'))
        self.seed = seed
        random.seed(self.seed)

    def add(self, *transition):
        self.memory.append(self.Transition(*transition))
        if len(self.memory) > self.memory_size:
            self.memory.pop(0)
        assert len(self.memory) <= self.memory_size

    def sample(self, size):
        return random.sample(self.memory, size)

    def __len__(self):
        return len(self.memory)

    @staticmethod
    def get_rng_state():
        return random.getstate()

    @staticmethod
    def set_rng_state(random_rng_state):
        random.setstate(random_rng_state)
