

import math

import numpy as np

from helper import adam_opt
from helper.activation_funcs import SoftmaxAct as ACTIVATION
from helper.dense_layer import DenseSigmoidLayer


class DenseSoftmaxLayer(DenseSigmoidLayer):

    #  Softmax Derivative is Sigmoid Derivative

    def active_func(self, x):
        numtr = np.exp(x - np.max(x, axis=1, keepdims=True));
        dentr = np.sum(numtr, axis=1, keepdims=True)
        funcValue = numtr / dentr;
        return funcValue


    def feed_forward(self, X):
        self.input_stack.append(X)
        act_inpt = self._get_W_t_plus_B(X);
        self.activation_input_stack.append(act_inpt)
        act_out = self.active_func(act_inpt)
        return act_out


    def feed_backward(self, grad_act):

        act_input = self.activation_input_stack.pop()
        grad_total = np.multiply(grad_act, self.active_derivative(act_input))

        W = self.weights
        l_inp = self.input_stack.pop()

        grad_weights, grad_bias = \
            self._grad_weight_bias_pair(l_inp, grad_total)

        self._update_weights(grad_weights)
        self._update_biases(grad_bias)

        grad_total = np.dot(grad_total, np.transpose(W))

        return grad_total
