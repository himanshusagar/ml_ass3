import numpy as np

from utility.scores import clip_me


def cross_entropy_loss(y_true, y_pred):
    y_pred = clip_me(y_pred)

    c_e =  -1.0 * ( y_true * np.log(y_pred) +
                    (1 - y_true) * np.log(1 - y_pred) );

    return c_e


def cross_entropy_gradient(y_true , y_pred):
    y_pred = clip_me(y_pred)

    return - (y_true / y_pred) + \
           (1 - y_true) / (1 - y_pred)



def cross_entropy_loss_n_gradient(y_true , y_pred):
    loss = np.mean(cross_entropy_loss(y_true, y_pred))
    loss_grad = cross_entropy_gradient(y_true, y_pred)
    return loss , loss_grad