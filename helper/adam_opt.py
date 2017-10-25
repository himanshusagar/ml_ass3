import numpy as np

#default and permanent params
learning_rate = 0.001
b1 = 0.9
b2 = 0.999
eps = 1e-8


def optimizer_update(w, grad_wrt_w, mean, variance):

    mean = b1 * mean + (1 - b1) * grad_wrt_w
    variance = b2 * variance + (1 - b2) * np.power(grad_wrt_w, 2)

    estmiated_mean = mean / (1 - b1)
    estmiated_Var = variance / (1 - b2)

    w_updt = learning_rate / (np.sqrt(estmiated_Var) + eps) * estmiated_mean

    upValue = w - w_updt
    return upValue , mean , variance

