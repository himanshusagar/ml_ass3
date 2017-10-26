import math

import numpy as np

from helper import adam_opt

class DenseSigmoidLayer():


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

#############################################
    ######### Activations

    def active_func(self, x):
        funcValue = 1.0 / (1.0 + np.exp(-x))
        return funcValue;

    def active_derivative(self, x):
        funcValue = self.active_func(x)
        derValue = funcValue * (1.0 - funcValue)
        return derValue

    #########################################
    def _grad_weight_bias_pair(self , l_inp , grad):
        grad_weights = np.dot(np.transpose(l_inp), grad)
        grad_bias = np.sum(grad, axis=0, keepdims=True)
        return grad_weights , grad_bias;

    def _get_W_t_plus_B(self , X):
        return np.dot(X , self.weights ) + self.bias;

    def feed_forward(self, X):
        self.input_stack.append( X )
        act_inpt = self._get_W_t_plus_B(X);
        self.activation_input_stack.append(act_inpt )
        act_out =  self.active_func(act_inpt)
        return act_out

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


    def feed_backward(self, grad_act):

        act_input = self.activation_input_stack.pop()
        grad_total = np.multiply(grad_act, self.active_derivative(act_input))

        W = self.weights
        l_inp = self.input_stack.pop()

        grad_weights , grad_bias = \
            self._grad_weight_bias_pair(l_inp, grad_total)

        self._update_weights(grad_weights)
        self._update_biases(grad_bias)

        grad_total = np.dot(grad_total, np.transpose(W))

        return  grad_total


    def output_shape(self):
        return (self.n_neurons,)


    def initialize_weights(self):

        limit = 1 / np.sqrt(self.input_shape[0])

        self.weights = np.random.uniform(-limit, limit, (self.input_shape[0], self.n_neurons))

        self.bias = np.zeros((1, self.n_neurons))


