import numpy as np
import pandas as pd
import statsmodels.iolib.smpickle as smpickle

class LogitUser:

    def __init__(self, features, target, pickle_file = "../pickle/smf.all.full.pickle", count = 1):
        self._features = features
        self._target = target
        self._count = count
        self._model = smpickle.load_pickle(pickle_file)
        # Static display features
        self._border = np.ones([8,8], dtype=bool)
        self._border[1:7,1:7] = False
        self._border = self._border.reshape(-1).tolist()
        self._first = [1] + 63 * [0]
        self._row1 = [max((8-pos)/8,0) for pos in range(64)]


    def decision(self, display):
        dists = 1 - np.dot(self._features[display], self._features[self._target])
        dranks = np.sum(np.reshape(dists, (-1, 1)) > np.reshape(dists, (1, -1)), axis=-1)

        feats = {'D_distance_to_target': dists.tolist(),
                'D_rank': dranks.tolist(),
                'first': self._first,
                'border': self._border,
                'row1': self._row1}
        feats = pd.DataFrame(feats)

        probabs = self._model.predict(feats)
        probabs /= np.sum(probabs)
        return display[np.random.choice(probabs.shape[0], self._count, p=probabs, replace=False)]