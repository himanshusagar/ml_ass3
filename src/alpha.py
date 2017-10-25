import numpy as np

from src.alpha_neural_net import AlphaNeuralNetwork
from utility import dataset_creater
from utility.data_manipulation import convert_non_one_hot
from utility.scores import accuracy_score

X_train, y_train = dataset_creater.loadIT("train")
X_test, y_test = dataset_creater.loadIT("valid")


sdf = np.unique(y_train);
n_classes =  np.size(sdf)


clf = AlphaNeuralNetwork(n_classes  , 'sigmoid' , 'sigmoid' )

n_samples = np.shape(X_train)[0]

batch_size = n_samples;
while (batch_size > 500):
    batch_size = batch_size * 0.1;

clf.fit(X_train, y_train, n_epochs=100 , batch_size=int(batch_size))

y_pred = clf.predict(X_test);
iter_accuracy = accuracy_score( y_test ,  y_pred)
print("Acc" , iter_accuracy)