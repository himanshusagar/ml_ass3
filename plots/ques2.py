import joblib
import numpy as np
from sklearn.model_selection import train_test_split

from utility import dataset_creater
import matplotlib.pyplot as plt


def main(X_train , y_train , X_valid , y_valid , prefix):



    y_axis = []
    x_axis_epoch = [ 2, 4 , 10 , 20 , 30 , 40 , 50 , 60 ];

    from impl.impl_networks import gen_plot_res
    for iEpoch in x_axis_epoch:
        y_axis.append( gen_plot_res(X_train , y_train , X_valid , y_valid,  iEpoch ))


    x_axis = x_axis_epoch[:len(y_axis)]
    fig = plt.figure( figsize=(11,8))
    ax1 = fig.add_subplot(111)
    ax1.plot( x_axis , y_axis , label ="Accuracy Graph" , color='c' , marker='o')


    plt.ylabel("Accuracy ")
    plt.xticks(x_axis)
    plt.xlabel("Epoch Count")

    handles , labels = ax1.get_legend_handles_labels()
    lgd = ax1.legend(handles , labels , loc = 'upper center' , bbox_to_anchor = (1.5 , 1) )
    ax1.grid('on')


    plt.savefig("../figures/"+ prefix + "_sigmoid")
    plt.show()




if __name__ == '__main__':

#     X_train , y_train = dataset_creater.loadIT("train")
#
#     X_valid, y_valid = dataset_creater.loadIT("test")
#
#     main(X_train , y_train  , X_valid ,  y_valid , 'large');
#
# #    exit(0)

    X_raw, y = dataset_creater.loadSmallbset();

    X = np.zeros((X_raw.shape[0], 784))
    for i in xrange(np.shape(X)[0]):
        X[i] = X_raw[i].flatten()

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)

    main(X_train, y_train, X_test, y_test , "small")



'''
def gen_epoch_res(X, y, X_valid, y_valid, activaton , prefix , epochs_list):

    if(activaton == "sigmoid"):
        activaton = 'logistic'

    skf = StratifiedKFold(n_splits=3)
    best_accuracy = 1e-15
    best_model = None

    for train_index , test_index in skf.split(X , y):

        mlp = MLPClassifier(hidden_layer_sizes=(50, 100), max_iter = 1, alpha=0.0,
                            solver='adam', verbose=10, tol=1e-4, random_state=1,
                            activation=activaton,
                            )

        X_train  , X_test = X[train_index] , X[test_index]
        y_train, y_test = y[train_index], y[test_index]

        mlp.fit(X_train, y_train )

        y_pred = mlp.predict(X_test);
        iter_accuracy = accuracy_score( y_test ,  y_pred)

        if(iter_accuracy >best_accuracy ):
            best_accuracy = iter_accuracy
            best_model = mlp

        print("iFold Accuracy : ", iter_accuracy, " Best:", best_accuracy)

    y_pred = best_model.predict(X_valid)
    valid_accuracy = accuracy_score(y_valid , y_pred)

    print("######################");
    print("Validation Accuracy : ", valid_accuracy)

    joblib.dump(best_model, "../models/" + prefix + "_" + activaton + "_model_pkl");

    return valid_accuracy
'''
