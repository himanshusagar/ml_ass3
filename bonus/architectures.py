from sklearn.neural_network import MLPClassifier

from src.alpha import X_train
from utility import dataset_creater
from utility.scores import accuracy_score

X_train, y_train = dataset_creater.loadIT("train")
X_valid, y_valid = dataset_creater.loadIT("valid")

mlp = MLPClassifier(hidden_layer_sizes=(50, 100,), max_iter=50, alpha=1e-4,
                    solver='adam', verbose=10, tol=1e-4, random_state=1,
                    activation='logistic'
                    )

mlp.fit(X_train, y_train)

y_pred = mlp.predict(X_valid);
accuracy = accuracy_score(y_valid, y_pred)
print("Acc", accuracy)
