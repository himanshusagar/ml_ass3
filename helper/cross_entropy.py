import numpy as np


def cross_entropy_loss(y_true, y_pred):
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - y_true * np.log(y_pred) - (1 - y_true) * np.log(1 - y_pred);



def cross_entropy_gradient(y_true , y_pred):
    # Avoid division by zero
    y_pred = np.clip(y_pred, 1e-15, 1 - 1e-15)
    return - (y_true / y_pred) + (1 - y_true) / (1 - y_pred)



def cross_entropy_loss_n_gradient(y_true , y_pred):
    loss = np.mean(cross_entropy_loss(y_true, y_pred))
    loss_grad = cross_entropy_gradient(y_true, y_pred)
    return loss , loss_grad