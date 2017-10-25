import numpy as np
import sys




class Sigmoid():
    def __init__(self): pass

    def function(self, x):
        return 1 / (1 + np.exp(-x))

    def gradient(self, x):
        return self.function(x) * (1 - self.function(x))


class Softmax():
    def __init__(self): pass

    def function(self, x):
        e_x = np.exp(x - np.max(x, axis=1, keepdims=True))
        return e_x / np.sum(e_x, axis=1, keepdims=True)

    def gradient(self, x):
        p = self.function(x)
        return p * (1 - p)

class ReLU():
    def __init__(self): pass

    def function(self, x):
        return np.where(x >= 0, x, 0)

    def gradient(self, x):
        return np.where(x >= 0, 1, 0)

