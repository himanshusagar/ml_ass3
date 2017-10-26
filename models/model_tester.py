from sklearn.externals import joblib

from utility import dataset_creater
from utility.scores import accuracy_score

X_test, y_test = dataset_creater.loadIT("test")
X_valid , y_valid = dataset_creater.loadIT("valid")

clf = joblib.load("sklearn_large_logistic_model_pkl")

y_pred = clf.predict(X_test);
iter_accuracy = accuracy_score( y_test ,  y_pred)
print("Acc Test" , iter_accuracy)

##########

y_pred = clf.predict( X_valid )
iter_accuracy = accuracy_score( y_valid  ,  y_pred )
print("Acc Valid" , iter_accuracy)

