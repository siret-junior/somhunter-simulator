import numpy as np


class NullUser:

    def __init__(self, count = 1):
        self._count = count


    def decision(self, display):
        return display[np.random.randint(low=0, high=display.shape[0], size=self._count)]
