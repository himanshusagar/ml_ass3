import numpy as np


#Remember to clip loss

def clip_me(value):
    value = np.clip(value, 1e-15, 1 - 1e-15)
    return value


def accuracy_score(y_true, y_pred):
    acc = 0.0;
    limit =np.shape(y_pred)[0]
    for i in xrange(limit):
        if(y_pred[i] == y_true[i]):
            acc = acc + 1.0;

    return acc  / limit


def accuracy_2d_score(y_true, y_pred):
    return accuracy_score(np.argmax(y_true, axis=1), np.argmax(y_pred, axis=1))
