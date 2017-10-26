from __future__ import print_function
import numpy as np

from helper import cross_entropy

IN_SHAPE = 28 * 28



from utility import scores


class AlphaNeuralNetwork():
    def __init__(self,  class_count , internal_layers='sigmoid', output_layer='softmax'):
        n_hidden_one = 100
        n_hidden_two = 50
        self.layers = []

        self.epoch_outputs = []

        if(class_count <= 0):
            raise ValueError("Class Count should be Non-Neg ")

        if(internal_layers == "sigmoid"):
            from helper.dense_layer import DenseSigmoidLayer
            self.add(DenseSigmoidLayer(n_hidden_one, input_shape=(IN_SHAPE,)) )
            self.add(DenseSigmoidLayer(n_hidden_two) )
        elif(internal_layers == 'relu'):
            from helper.relu_layer import DenseReluLayer
            self.add(DenseReluLayer(n_hidden_one, input_shape=(IN_SHAPE,)))
            self.add(DenseReluLayer(n_hidden_two))
        elif(internal_layers == 'maxout'):
            from helper.maxout_layer import DenseMaxOutLayer
            self.add(DenseMaxOutLayer(n_hidden_one, input_shape=(IN_SHAPE,)))
            self.add(DenseMaxOutLayer(n_hidden_two))

        if(output_layer == 'softmax'):
            from helper.softmax_layer import DenseSoftmaxLayer
            self.add(DenseSoftmaxLayer(class_count) );
        elif(output_layer == 'sigmoid' ):
            from helper.dense_layer import DenseSigmoidLayer
            self.add(DenseSigmoidLayer(class_count));

        if( len(self.layers) > 3):
            raise ValueError("UNexpected Case {0} {1} Size :".format(internal_layers , output_layer , len(self.layers)))

    def add(self, layer):
        if self.layers:
            layer.input_shape = self.layers[-1].output_shape()
            #last layer's output is your input

        elif (layer.input_shape[0] == -1):
            raise ValueError("Make Sure to set input size of first layer")

        layer.initialize_weights()
        self.layers.append(layer)

    def train_on_minilbatch(self, X, y):

        #neural networks and deep learnnig chapter 3
        '''

        Feed Forward
        z_l =  w_l * a_l-1 + b_l
        '''
        z_l = self._feed_forward(X)

        '''
        Compute Loss
        '''
        loss_l , loss_l_grad = cross_entropy.cross_entropy_loss_n_gradient(y , z_l)

        self._feed_backward(loss_grad=loss_l_grad)

        return loss_l


    def fit(self, X, y, n_epochs, batch_size):

        from utility.data_manipulation import convert_one_hot
        y = convert_one_hot(y)

        n_samples = np.shape(X)[0]
        n_batches = int(n_samples / batch_size)

        self.layer_stats(batch_size)

        for i_epoch in range(n_epochs):
            #SOURCE : Mini Batch Shuffle Strategy from StackOverflow.com

            indices = range(n_samples)
            np.random.shuffle(indices)

            cummulative_batch_error = 0

            for i in range(n_batches):
                #mini batch
                X_batch = X[indices[i * batch_size:(i + 1) * batch_size]]
                y_batch = y[indices[i * batch_size:(i + 1) * batch_size]]
                iLoss  = self.train_on_minilbatch(X_batch, y_batch)
                cummulative_batch_error += iLoss

            print("Epoch = {0}/{1}  = {2} ".format(i_epoch, n_epochs, cummulative_batch_error))
            from utility.scores import accuracy_score
            try:
                self.epoch_outputs.append( accuracy_score( self.validY , self.predict(self.validX) ) );
            except AttributeError:
                doNothin_cant_leave_empty = True


    def _feed_forward(self, X):
        l_fwd_inp = X
        for layer in self.layers:
            l_fwd_inp = layer.feed_forward(l_fwd_inp)
        return l_fwd_inp

    def _feed_backward(self, loss_grad):
        l_bwd_grad = loss_grad
        for layer in self.layers[::-1]:
            l_bwd_grad = layer.feed_backward(l_bwd_grad)


    def layer_stats(self , batch_sizes):
        print(self.__class__.__name__)
        print("Batch Size :  " + str(batch_sizes))
        for index , i_layer in  enumerate(self.layers):
            print("Layer {0} : {1} & Shape : {2} ".format(index , i_layer.__class__.__name__ ,
                                                           i_layer.output_shape() ) )

    def predict(self, X):
        return np.argmax(self._feed_forward(X) , axis=1 )

    def set_valid_data(self ,validX , validY):
        self.validX = validX;
        self.validY = validY;


if __name__ == "__main__":
    print("Main")
