import numpy as np


class RanSamMultiplePriorUser:

    def __init__(self, features, target, power, prior_mask, count = 1):
        self._features = features
        self._target = target
        self._count = count
        self._power = power
        self._prior_mask = prior_mask


    def decision(self, disp, disp_type):
        dist_to_target = (1 + np.dot(self._features[disp], self._features[self._target])) / 2
        dist_to_target = dist_to_target ** self._power
        dist_to_target = dist_to_target * self._prior_mask[disp_type]
        dist_to_target = dist_to_target / np.sum(dist_to_target)
        return disp[np.random.choice(dist_to_target.shape[0], self._count, p=dist_to_target, replace=False)]


    def decision_ids(self, disp, disp_type):
        dist_to_target = (1 + np.dot(self._features[disp], self._features[self._target])) / 2
        dist_to_target = dist_to_target ** self._power
        dist_to_target = dist_to_target * self._prior_mask[disp_type]
        dist_to_target = dist_to_target / np.sum(dist_to_target)
        return np.random.choice(dist_to_target.shape[0], self._count, p=dist_to_target, replace=False)

