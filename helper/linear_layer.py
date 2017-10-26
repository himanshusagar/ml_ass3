import numpy as np

from helper.dense_layer import DenseSigmoidLayer

POWER = 1   ;

class DenseLinearLayer(DenseSigmoidLayer):

    def active_func(self, x):
        if(POWER == 2):
            return np.square(x)
        else:
            return np.power(x, POWER);

    def active_derivative(self, x):
        return POWER;


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


