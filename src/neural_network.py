from __future__ import print_function
from terminaltables import AsciiTable
import numpy as np
# import progressbar
#
#
# bar_widgets = [
#     'Training: ', progressbar.Percentage(), ' ', progressbar.Bar(marker="-", left="[", right="]"),
#     ' ', progressbar.ETA()
# ]
#

class NeuralNetwork():


    def __init__(self, optimizer, loss):
        self.optimizer = optimizer
        self.layers = []
        self.errors = {"training": [], "validation": []}
        self.loss_function = loss()


    # Method which enables freezing of the weights of the network's layers.
    def set_trainable(self, trainable):
        for layer in self.layers:
            layer.trainable = trainable

    # Method which adds a layer to the neural network
    def add(self, layer):
        # If the first layer has been added set the input shape
        # as the output shape of the previous layer
        if self.layers:
            layer.set_input_shape(shape=self.layers[-1].output_shape())

        # If the layer has weights that needs to be initialized
        if hasattr(layer, 'initialize'):
            layer.initialize(optimizer=self.optimizer)

        # Add layer to network
        self.layers.append(layer)

    def train_on_batch(self, X, y):
        # Calculate output
        y_pred = self._forward_pass(X)
        # Calculate the training loss
        loss = np.mean(self.loss_function.loss(y, y_pred))
        # Calculate the gradient of the loss function wrt y_pred
        loss_grad = self.loss_function.derivative(y, y_pred)
        # Calculate the accuracy of the prediction
        acc = self.loss_function.acc(y, y_pred)
        # Backprop. Update weights
        self._backward_pass(loss_grad=loss_grad)

        return loss, acc


    def fit(self, X, y, n_epochs, batch_size):

        # if(np.shape(y)[1] <= 1):
        #     raise ValueError("Y should be One Hot encoded")

        # Convert to one-hot encoding

        from utility.data_manipulation import convert_one_hot
        y = convert_one_hot(y.astype("int"))


        n_samples = np.shape(X)[0]
        n_batches = int(n_samples / batch_size)

     ##   bar = progressbar.ProgressBar(widgets=bar_widgets)

        for i_epoch in range(n_epochs):
            idx = range(n_samples)
            ##print("DISABLED RANDOM")
            np.random.shuffle(idx)

            batch_t_error = 0  # Mean batch training error
            for i in range(n_batches):
                X_batch = X[idx[i * batch_size:(i + 1) * batch_size]]
                y_batch = y[idx[i * batch_size:(i + 1) * batch_size]]
                loss, _ = self.train_on_batch(X_batch, y_batch)
                batch_t_error += loss

            print("Epoch = {0}/{1}  = {2} ".format( i_epoch , n_epochs ,batch_t_error))
            # Save the epoch mean error
            self.errors["training"].append(batch_t_error / n_batches)

        return self.errors["training"], self.errors["validation"]

    def _forward_pass(self, X, training=True):
        # Calculate the output of the NN. The output of layer l1 becomes the
        # input of the following layer l2
        layer_output = X
        for layer in self.layers:
            layer_output = layer.feed_forward(layer_output, training)

        return layer_output

    def _backward_pass(self, loss_grad):
        # Propogate the gradient 'backwards' and update the weights
        # in each layer
        acc_grad = loss_grad
        for layer in reversed(self.layers):
            acc_grad = layer.feed_backward(acc_grad)

    def summary(self, name="Model Summary"):

        # Print model name
        print(AsciiTable([[name]]).table)

        print("Input Shape: %s" % str(self.layers[0].input_shape))

        # Print the each layer's configuration
        table_data = [["Layer Type", "Parameters", "Output Shape"]]
        tot_params = 0
        for layer in self.layers:
            layer_name = layer.layer_name()
            params = layer.parameters()
            out_shape = layer.output_shape()
            table_data.append([layer_name, str(params), str(out_shape)])

            tot_params += params
        print(AsciiTable(table_data).table)

        print("Total Parameters: %d\n" % tot_params)

    # Use the trained model to predict labels of X
    def predict(self, X):
        return self._forward_pass(X, training=False)



if __name__ == "__main__":
    print("Main")