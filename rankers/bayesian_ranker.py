import numpy as np

class BayesianRanker:

    
    def __init__(self, features, size, sigma = 0.1):
        self._features = features
        self._scores = np.ones(size, dtype=np.float32)
        self._sigma = sigma
        self._size = size
        self.MIN_SCORE = 1e-12

    def reset(self):
        self._scores[:] = 1
        
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
        for lik in likes:
            lik_exp = np.exp(- (1 - np.dot(self._features, self._features[lik])) / self._sigma)
            disp_feats = np.array([self._features[di] for di in display])
            disp_exps = np.exp( - (1 - self._features @ disp_feats.T) / self._sigma)
            self._scores *= lik_exp / (lik_exp + np.sum(disp_exps, axis = 1))
            
        self.normalize()
            
    def normalize(self):
        smax = np.max(self._scores)
        if smax < self.MIN_SCORE:
            smax = self.MIN_SCORE
        self._scores /= smax
        self._scores[self._scores < self.MIN_SCORE] = self.MIN_SCORE

