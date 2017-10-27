
import numpy as np
from sklearn.datasets import fetch_mldata
from sklearn.model_selection import train_test_split, StratifiedKFold
from sklearn.neural_network import MLPClassifier
from sklearn.preprocessing import OneHotEncoder, LabelBinarizer

from utility import dataset_creater
from utility.scores import accuracy_score


IS_TRAIN_MODE = True

# mnist = fetch_mldata("MNIST original")

def gen_plot_res(X, y, X_valid, y_valid  ,n_epochs):

    mlp = MLPClassifier(hidden_layer_sizes=(50, 100,), max_iter=n_epochs, alpha=1e-4,
                            solver='adam', verbose=10, tol=1e-4, random_state=1,
                            activation='logistic'
                            )

    mlp.fit(X, y )

    y_pred = mlp.predict(X_valid);
    accuracy = accuracy_score(y_valid, y_pred)

    print("Epoch" , n_epochs)
    print("Acc"  , accuracy)
    return accuracy;


def k_fold_compute(X, y, X_valid, y_valid, activaton=None, prefix=None):

    if(activaton == "sigmoid"):
        activaton = 'logistic'

    skf = StratifiedKFold(n_splits=3)
    best_accuracy = 1e-15;
    best_model = None

    for train_index , test_index in skf.split(X , y):


        from plots.large_src import MAX_EPOCH
        mlp = MLPClassifier(hidden_layer_sizes=(50, 100), max_iter=100, alpha=1e-4,
                            solver='adam', verbose=1, tol=1e-4, random_state=42,
                            activation=activaton
                            )
        print(mlp)

        X_train  , X_test = X[train_index] , X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mlp.fit(X_train, y_train )#, classes=np.unique(y_train))

        y_pred = mlp.predict(X_test);
        iter_accuracy = accuracy_score( y_test ,  y_pred)
        if(iter_accuracy >best_accuracy ):
            best_accuracy = iter_accuracy
            best_model = mlp
        print("iFold Accuracy : ", iter_accuracy, " Best:", best_accuracy)
        break

    y_pred = best_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid , y_pred)
    print("...................")
    print("Validation Accuracy : ", valid_accuracy)

    best_model.partial_fit(X_valid , y_valid)

    from sklearn.externals import joblib
    joblib.dump(best_model, "../models/sklearn_" + prefix + "_" + activaton + "_model_pkl");



def small_main(activation):
    X_raw, y = dataset_creater.loadSmallbset();

    X = np.zeros( (X_raw.shape[0] , 784) )
    for i in xrange(np.shape(X)[0]):
        X[i] = X_raw[i].flatten()


    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.1, random_state = 42)

    k_fold_compute(X_train, y_train, X_test, y_test, activation, "small")


def large_main(activation):
    X_train, y_train = dataset_creater.loadIT("train")
    X_valid, y_valid = dataset_creater.loadIT("valid")

    k_fold_compute(X_train, y_train, X_valid, y_valid, activation, "large")



if __name__ == '__main__':
    #small_main('sigmoid')
    small_main('sigmoid')