import numpy as np
import som_py.SOMDisplay as SOM

class SOMDisplay:

    def __init__(self, features, width = 8, height = 8, seed = 42):
        self._features = features
        self._width = width
        self._height = height
        self._seed = seed

    def generate(self, scores):
        return SOM.create_som_display(self._features, scores, self._width, self._height, self._seed)

