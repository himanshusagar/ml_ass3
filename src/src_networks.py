
import numpy as np
from sklearn.externals import joblib
from sklearn.model_selection import StratifiedKFold, train_test_split

from utility.scores import accuracy_score
from src.alpha_neural_net import AlphaNeuralNetwork, IN_SHAPE
from utility import scores, dataset_creater

n_hidden_one = 100
n_hidden_two = 50

CLASS_COUNT = 10;
#
# X_train, y_train = dataset_creater.loadIT("train")
# X_test, y_test = dataset_creater.loadIT("test")
#

def main():
    large_main("sigmoid" , "softmax" )
    #    large_main('ques1ca');

def k_fold_compute(X, y, X_valid, y_valid, internal_layers=None , output_layer=None, prefix=None):


    skf = StratifiedKFold(n_splits=3)
    best_accuracy = 0.0;

    all_models = []
    uniq_clas = np.unique(y);
    n_classes =  np.size(uniq_clas)

    for train_index , test_index in skf.split(X , y):
        X_train  , X_test = X[train_index] , X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        if(prefix == 'small' ):
            y_test[ np.where(y_test == np.min(uniq_clas) ) ] = 0
            y_test[np.where(y_test == np.max(uniq_clas))] = 1

        clf = AlphaNeuralNetwork(n_classes  , internal_layers , output_layer )

        n_samples = np.shape(X_train)[0]

        batch_size = n_samples;
        while (batch_size > 500):
            batch_size = batch_size * 0.1;

        clf.set_valid_data(X_test , y_test)

        from plots.large_src import MAX_EPOCH
        clf.fit(X_train, y_train, n_epochs= MAX_EPOCH , batch_size=int(batch_size))

        all_models.append(clf)

        y_pred = clf.predict(X_test);


        iter_accuracy = accuracy_score( y_test ,  y_pred)
        if(iter_accuracy >best_accuracy ):
            best_accuracy = iter_accuracy
            best_model = clf
        print("iFold Accuracy : ", iter_accuracy, " Best:", best_accuracy)



    y_pred = best_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid , y_pred)
    print("...................")
    print("Validation Accuracy : ", valid_accuracy)

    #best_model.partial_fit(X_valid , y_valid)

    joblib.dump(best_model, "../models/" +"custom_" + prefix + "_" + internal_layers + "_" +  output_layer + "_model_pkl");

    all_models.append(best_model);
    return all_models;


def small_main(internal , output):
    X_raw, y = dataset_creater.loadSmallbset();
    X = np.zeros( (X_raw.shape[0] , 784) )

    for i in xrange(np.shape(X)[0]):
        X[i] = X_raw[i].flatten()

    X_train, X_test, y_train, y_test = train_test_split( X, y, test_size = 0.1, random_state = 42)


    return k_fold_compute(X_train, y_train, X_test, y_test, internal , output , 'small')


def large_main( internal  , output ):
    X_train, y_train = dataset_creater.loadIT("train")
    X_valid, y_valid = dataset_creater.loadIT("test")


    return k_fold_compute(X_train, y_train, X_valid, y_valid,internal , output , 'large')


if __name__ == '__main__':
    main();