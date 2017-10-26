from helper.dense_layer import DenseSigmoidLayer
import numpy as np

from helper.linear_layer import DenseLinearLayer
from utility.data_manipulation import convert_one_hot


class DenseMaxOutLayer(DenseSigmoidLayer):
    # Source : http://www.simon-hohberg.de/2015/07/19/maxout.html


    def __init__(self, n_units, input_shape=None, k=2):
        DenseSigmoidLayer.__init__(self, n_units, input_shape)
        self.internal_layers = []
        self.k = k;

        for i_internal_layer in range(k):
            self.internal_layers.append((i_internal_layer
                                         , DenseLinearLayer(n_units, input_shape)))

    def active_derivative(self, x):
        mask_iLayer = []

        for index, iLayer in self.internal_layers:
            value = np.asarray(np.where(self.act_mask_index_wise == index , 1, 0), dtype=bool)
            mask_iLayer.append(value);

        actual = np.sum(np.logical_or(mask_iLayer[0] , mask_iLayer[1]))
        expected = np.sum(np.ones_like(mask_iLayer[0] , dtype=bool) )

        if(actual != expected):
            raise ValueError("Check Derivative of MaxOut");

        return mask_iLayer;


    def active_func(self, x):
        #internal_x = [iLayer.active_func(x) for id, iLayer in self.internal_layers]

        funcValue = np.max(x , axis=0)

        #funcValue = np.where( x[0] > x[1] , x[0] , x[1] )

        # if one pos has 0 it means you have to pick whatever value of 0th layer in it's position
        self.act_mask_index_wise = np.argmax(x , axis=0)
        #np.asarray(np.where(x[0] > x[1]))

        return funcValue;

    def _get_W_t_plus_B(self, X):
        raise NotImplementedError();

    def feed_forward(self, X):

        arrr = [ ]
        for index , iLayer in self.internal_layers:
            arrr.append(iLayer.feed_forward(X));

        #NO NEED
        #self.input_stack.append(X)

        act_inpt = [iLayer._get_W_t_plus_B(X) for index , iLayer in self.internal_layers]

        self.activation_input_stack.append(act_inpt)

        act_out = self.active_func( act_inpt  )
        return act_out


    def feed_backward(self, grad_act):

        act_input = self.activation_input_stack.pop()

        der_list = self.active_derivative(act_input);

        grad_list = []

        for index , iLayer in self.internal_layers:
            grad_list.append(
                iLayer.feed_backward( np.multiply(grad_act , der_list[index] ) ) );

        #row , col = np.shape(grad_act)

        # grad_via_mask =\
        #     np.asarray( np.where( self.act_mask_index_wise , grad_list[0] , grad_list[1]  ))

        grad_via_mask = grad_list[0] + grad_list[1]

        return grad_via_mask;

    def initialize_weights(self):
        self.weights = None;
        self.bias = None;

        for index, iLayer in self.internal_layers:
            iLayer.input_shape = self.input_shape;
            iLayer.initialize_weights();

