import h5py
import numpy as np

import mnist_loader
from utility.data_manipulation import convert_non_one_hot


def apply(data):
    import mnist_loader

    X_train = np.transpose(data[0][0])  # np.ones( (0  , 28 * 28))
    y_train = np.transpose(data[0][1])  # np.ones(( len(data) , 1))

    flag = False
    if (np.size(data[0][1]) > 1):
        # y_train = np.ones((0 , 10))
        flag = True

    for i in xrange(1, len(data)):
        X_train = np.row_stack((X_train, np.transpose(data[i][0] ) ))
        y_train = np.row_stack((y_train, np.transpose(data[i][1]) ))
        if(i%1000 == 0):
            print(i)

    # if flag:
    #     for i in xrange(1 , len(data)):
    #         X_train = np.row_stack( (X_train , data[i][0]) )
    #         y_train = np.row_stack( (y_train , data[i][1]) )
    # else:
    #     for i in xrange(0 , len(data)):
    #         X_train = np.row_stack((X_train  , data[i][0]) )
    #         y_train[i] = data[i][1]
    #

    print(np.shape(X_train))
    print(np.shape(y_train))
    return X_train, y_train


def saveIT(name, data):
    with h5py.File("../dataset/" + name, 'w') as hf:
        hf.create_dataset(name=name, data=data)

def loadIT(partial_name ):
    if(partial_name == "val"):
        partial_name = "valid";

    with h5py.File("../dataset/" + partial_name + "X" , 'r') as hf:
        dataX = hf[partial_name + "X"][:]

    with h5py.File("../dataset/" + partial_name + "Y", 'r') as hf:
        dataY = hf[partial_name + "Y"][:]
        if( (np.shape(dataY)[1] )  <= 1):
            dataY = dataY.flatten()

    if(partial_name == 'train'):
        dataY = convert_non_one_hot(dataY)

    return dataX , dataY


def applyANDSave(data, name):
    X, y = apply(data)
    saveIT(name + "X", X);
    saveIT(name + "Y", y);


# Load the test data
def load_h5py(filename):
    with h5py.File(filename, 'r') as hf:
        X = hf['X'][:]
        Y = hf['Y'][:]
    return X, Y


def loadSmallbset():
    return load_h5py("../data/" + "dataset_partA.h5")

if __name__ == '__main__':
    trainData, testData, validData = mnist_loader.load_data_wrapper()

   # applyANDSave(testData, "test")
    print("Test Done")
    #
    # applyANDSave(validData, "valid")


    print("Valid Done")
    applyANDSave(trainData, "train")
    print("Train Done")
