import numpy as np


class SigmoidAct():
    @staticmethod
    def func(x):
        funcValue =  1.0 / (1.0 + np.exp(-x))
        return funcValue;

    @staticmethod
    def derivative(x):
        funcValue = SigmoidAct.func(x)
        derValue = funcValue * (1.0 - funcValue)
        return derValue

class SoftmaxAct():

    @staticmethod
    def func(x):
        numtr = np.exp(x - np.max(x, axis=1, keepdims=True));
        dentr = np.sum( numtr , axis=1, keepdims=True)
        funcValue = numtr/dentr;
        return funcValue

    @staticmethod
    def derivative(x):
        funcValue = SoftmaxAct.func(x);
        derValue = funcValue * (1.0 - funcValue)
        return derValue

class ReluACT():

    @staticmethod
    def func(x):
        funcValue = np.where(x >= 0.0, x, 0.0)
        return funcValue;

    @staticmethod
    def derivative(x):
        derValue = np.where(x >= 0.0, 1.0, 0.0)
        return derValue

