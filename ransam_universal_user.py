import numpy as np


class RanSamUniversalUser:
    
    def decision(self, display, weights, power, count):
        weights = weights ** power
        weights = weights / np.sum(weights)
        return display[np.random.choice(weights.shape[0], count, p=weights, replace=False)]


    def decision_ids(self, display, weights, power, count):
        weights = weights ** power
        weights = weights / np.sum(weights)
        return np.random.choice(weights.shape[0], count, p=weights, replace=False)

