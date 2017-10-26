import numpy as np

from helper.dense_layer import DenseSigmoidLayer


class DenseReluLayer(DenseSigmoidLayer):

    def active_derivative(self, x):
        derValue = np.where(x >= 0.0, 1.0, 0.0)
        return derValue

    def active_func(self, x):
        funcValue = np.where(x >= 0.0, x, 0.0)
        return funcValue;


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


