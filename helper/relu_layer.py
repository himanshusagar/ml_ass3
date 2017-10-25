

import math

import numpy as np

from helper import adam_opt
from helper.activation_funcs import ReluACT as ACTIVATION

class DenseReluLayer():

    def __init__(self, n_units, input_shape=None):
        self.input_stack = []
        self.activation_input_stack = []
        self.input_shape, self.n_neurons = input_shape, n_units
        self.weights = None
        self.bias = None

        ## for gradient descent
        self.weights_mean = np.array([])
        self.weights_variance = np.array([])

        self.bias_mean = np.array([])
        self.bias_variance = np.array([])
        ##end


    def initialize(self):
        # Initialize the weights
        limit = 1 / math.sqrt(self.input_shape[0])

        self.weights = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_neurons))
        self.bias = np.zeros((1, self.n_neurons))



    def feed_forward(self, X):
        self.input_stack.append( X )
        act_inpt = X.dot(self.weights) + self.bias;
        self.activation_input_stack.append(act_inpt )

        return ACTIVATION.func(act_inpt)


    def _update_weights(self, grad_weights):

        if not self.weights_mean.any():
            self.weights_mean = np.zeros_like(grad_weights)
            self.weights_variance = np.zeros_like(grad_weights)

        self.weights, self.weights_mean, self.weights_variance \
            = adam_opt.optimizer_update(self.weights, grad_weights,
                                        self.weights_mean, self.weights_variance)

    def _update_biases(self, grad_biases):

        if not self.bias_mean.any():
            self.bias_mean = np.zeros_like(grad_biases)
            self.bias_variance = np.zeros_like(grad_biases)

        self.bias , self.bias_mean , self.bias_variance \
            = adam_opt.optimizer_update(self.bias, grad_biases , self.bias_mean , self.bias_variance)


    def feed_backward(self, summed_grad):

        act_input = self.activation_input_stack.pop()
        summed_grad = np.multiply(summed_grad, ACTIVATION.derivative(act_input))

        W = self.weights
        l_inp = self.input_stack.pop()

        grad_weights = np.dot(np.transpose(l_inp), summed_grad)
        grad_bias = np.sum(summed_grad, axis=0, keepdims=True)

        self._update_weights(grad_weights)
        self._update_biases(grad_bias)

        summed_grad = np.dot(summed_grad , np.transpose(W ) )

        return  summed_grad


    def output_shape(self):
        return (self.n_neurons,)


