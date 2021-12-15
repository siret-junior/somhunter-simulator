import numpy as np

class BayesianAvgRanker:

    def __init__(self, features, size, alpha = 0.9, sigma = 0.1):
        self._features = features
        self._scores = np.ones(size)
        self._sigma = sigma
        self._alpha = alpha
        self._size = size

    @property
    def features(self):
        return self._features

    @property
    def scores(self):
        return self._scores

    @property
    def sigma(self):
        return self._sigma

    @property
    def size(self):
        return self._size

    def probabilities(self):
        return self._scores / np.sum(self._scores)

    def apply_feedback(self, likes, display):
        display = list(filter(lambda r: r not in likes, display))
        updates = []
        for lik in likes:
            lik_exp = np.exp(- (1 - np.dot(self._features, self._features[lik])) / self._sigma)
            disp_feats = np.array([self._features[di] for di in display])
            disp_exps = np.exp( - (1 - self._features @ disp_feats.T) / self._sigma)
            updates.append(lik_exp / (lik_exp + np.sum(disp_exps, axis = 1)))
            
        self._scores = np.exp( np.log(self._scores) * (2 - self._alpha) + self._alpha * np.log(np.prod(updates, axis=0)))
        #self._scores = np.exp( np.log(self._scores) + np.log(np.prod(updates, axis=0)))
        self._scores /= np.max(self._scores)

