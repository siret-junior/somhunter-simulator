import numpy as np


class IdealUser:

    def __init__(self, features, target, count = 1):
        self._features = features
        self._target = target
        self._count = count


    def decision(self, display):
        dist_to_target = 1 - np.dot(self._features[display], self._features[self._target])
        return display[dist_to_target.argsort()[:self._count]]
