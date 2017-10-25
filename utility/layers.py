from __future__ import print_function, division
import sys
import os
import math
import numpy as np
import copy
from activations import Sigmoid, ReLU ,Softmax

class Layer(object):
    def set_input_shape(self, shape):
        self.input_shape = shape

    def layer_name(self):
        return self.__class__.__name__

    def parameters(self):
        return 0

    def forward_pass(self, X, training):
        raise NotImplementedError()

    def backward_pass(self, acc_grad):
        raise NotImplementedError()

    def output_shape(self):
        raise NotImplementedError()


class Dense(Layer):

    def __init__(self, n_units, input_shape=None):
        self.layer_input = None
        self.initialized = False

        self.input_shape, self.n_units = input_shape, n_units
        self.W = None
        self.wb = None

        self.trainable = True

    def initialize(self, optimizer):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])
        self.W = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_units))
        self.wb = np.zeros((1, self.n_units))
        # Weight optimizers
        self.W_opt = copy.deepcopy(optimizer)
        self.wb_opt = copy.deepcopy(optimizer)

    def parameters(self):
        return np.prod(self.W.shape) + np.prod(self.wb.shape)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return X.dot(self.W) + self.wb

    def backward_pass(self, acc_grad):
        # Save weights used during forwards pass
        W = self.W

        if self.trainable:
            # Calculate gradient w.r.t layer weights
            grad_w = self.layer_input.T.dot(acc_grad)
            grad_wb = np.sum(acc_grad, axis=0, keepdims=True)

            # Update the layer weights
            opt = self.W_opt;

            self.W = opt.update(self.W, grad_w)
            self.wb = self.wb_opt.update(self.wb, grad_wb)

        # Return accumulated gradient for next layer
        # Calculated based on the weights used during the forward pass
        acc_grad = acc_grad.dot(W.T)
        return acc_grad

    def output_shape(self):
        return (self.n_units,)



class Activation(Layer):
    """A layer that applies an activation operation to the input.

    Parameters:
    -----------
    name: string
        The name of the activation function that will be used.
    """
    activation_functions = {
        'relu': ReLU,

        'sigmoid': Sigmoid,

        'softmax': Softmax,

    }

    def __init__(self, name):
        self.activation_name = name
        self.activation = self.activation_functions[name]()
        self.trainable = True

    def initialize(self , optimizer=None):
        print("Empty Init")

    def layer_name(self):
        return "%s (%s)" % (self.__class__.__name__, self.activation.__class__.__name__)

    def forward_pass(self, X, training=True):
        self.layer_input = X
        return self.activation.function(X)

    def backward_pass(self, acc_grad):
        return acc_grad * self.activation.gradient(self.layer_input)

    def output_shape(self):
        return self.input_shape
