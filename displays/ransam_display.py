import numpy as np


class RanSamDisplay:

    def __init__(self, dsize = 64):
        self._dsize = dsize

    def generate(self, scores):
        return np.random.choice(scores.shape[0], self._dsize, p=scores, replace=False)

