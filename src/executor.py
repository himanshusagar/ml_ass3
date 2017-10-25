import joblib
import numpy as np
from sklearn import datasets

from src.neural_network import NeuralNetwork
from utility import mnist_loader, dataset_creater, lossNoptimizer, scores
from utility.layers import Dense, Activation


CLF_NAME = "NeuralNet"
IN_SHAPE = 28*28

BOOL_ARG = True

def main(arg):
    data = datasets.load_digits()
    X = data.data
    y = data.target

    X_train, y_train = dataset_creater.loadIT("train")
    X_test, y_test = dataset_creater.loadIT("test")



    if arg:

        n_hidden_one = 100
        n_hidden_two = 50

        clf = NeuralNetwork(optimizer=lossNoptimizer.Adam(),
                            loss= lossNoptimizer.CrossEntropy,
                            )

        clf.add(Dense(n_hidden_one, input_shape=( IN_SHAPE ,)))
        clf.add(Activation('sigmoid'))
        clf.add(Dense(n_hidden_two))
        clf.add(Activation('sigmoid'))

        # clf.add(Dropout(0.25))
        # clf.add(Dense(n_hidden))
        # clf.add(Activation('softmax'))
        # clf.add(Dropout(0.25))
        # clf.add(Dense(n_hidden))
        # clf.add(Activation('leaky_relu'))
        # clf.add(Dropout(0.25))

        clf.add(Dense(10))
        clf.add(Activation('softmax'))

        print ()
        clf.summary(name="MLP")

        clf.fit(X_train, y_train, n_epochs=150, batch_size=len(X_train/10000))
        ##clf.plot_errors()
        joblib.dump(clf , CLF_NAME)
    else:
        clf = joblib.load(CLF_NAME)

    y_pred = np.argmax(clf.predict(X_test), axis=1)

    y_test_mod = []
    for i in y_test:
        y_test_mod.append(i[0].tolist())

    #y_test = [i[0] for i in y_test]

    accuracy = scores.accuracy_score(y_test, y_pred)
    print ("Accuracy: %f", accuracy)


if __name__ == '__main__':
    main(BOOL_ARG);