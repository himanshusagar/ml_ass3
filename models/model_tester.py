import numpy as np
from sklearn.externals import joblib
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

from utility import dataset_creater


X_test, y_test = dataset_creater.loadIT("test")
X_valid , y_valid = dataset_creater.loadIT("valid")

# X_raw, y = dataset_creater.loadSmallbset();
#
# X = np.zeros((X_raw.shape[0], 784))
# for i in xrange(np.shape(X)[0]):
#     X[i] = X_raw[i].flatten()
#
# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.1, random_state=42)
#

clf = joblib.load("sklearn_large_logistic_model_pkl")

y_pred = clf.predict(X_test);
iter_accuracy = accuracy_score(y_test, y_pred)
print("Acc Test", iter_accuracy)

##########

y_pred = clf.predict( X_valid )
iter_accuracy = accuracy_score( y_valid  ,  y_pred )
print("Acc Valid" , iter_accuracy)
