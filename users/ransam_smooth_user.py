import numpy as np


class RanSamSmoothUser:

    def __init__(self, features, target, power, up, right, count = 1):
        self._features = features
        self._target = target
        self._count = count
        self._power = power
        self._up = up
        self._right = right


    def decision(self, display):
        dist_to_target = (1 + np.dot(self._features[display], self._features[self._target])) / 2
        dist_to_target += self._right
        dist_to_target = dist_to_target ** self._power + self._up
        dist_to_target = dist_to_target / np.sum(dist_to_target)
        return display[np.random.choice(dist_to_target.shape[0], self._count, p=dist_to_target, replace=False)]


    def decision_ids(self, display):
        dist_to_target = (1 + np.dot(self._features[display], self._features[self._target])) / 2
        dist_to_target += self._right
        dist_to_target = dist_to_target ** self._power + self._up
        dist_to_target = dist_to_target / np.sum(dist_to_target)
        return np.random.choice(dist_to_target.shape[0], self._count, p=dist_to_target, replace=False)

