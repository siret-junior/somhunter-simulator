import numpy as np


class TopNDisplay:

    def __init__(self, dsize = 64):
        self._dsize = dsize

    def generate(self, scores):
        return np.argsort(scores)[::-1][:self._dsize]

