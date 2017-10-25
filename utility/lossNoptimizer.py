import numpy as np


class CrossEntropy():
    def __init__(self): pass

    def loss(self, y, p):

        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - y * np.log(p) - (1 - y) * np.log(1 - p)

    def acc(self, y, p):
        from scores import accuracy_score
        return accuracy_score(np.argmax(y, axis=1), np.argmax(p, axis=1))


    def gradient(self, y, p):
        # Avoid division by zero
        p = np.clip(p, 1e-15, 1 - 1e-15)
        return - (y / p) + (1 - y) / (1 - p)


class Adam():
    def __init__(self, learning_rate=0.001, b1=0.9, b2=0.999):
        self.learning_rate = learning_rate
        self.eps = 1e-8
        self.m = np.array([])
        self.v = np.array([])
        # Decay rates
        self.b1 = b1
        self.b2 = b2


    def update(self, w, grad_wrt_w):
        # If not initialized
        if not self.m.any():
            self.m = np.zeros(np.shape(grad_wrt_w))
            self.v = np.zeros(np.shape(grad_wrt_w))

        self.m = self.b1 * self.m + (1 - self.b1) * grad_wrt_w
        self.v = self.b2 * self.v + (1 - self.b2) * np.power(grad_wrt_w, 2)

        m_hat = self.m / (1 - self.b1)
        v_hat = self.v / (1 - self.b2)

        self.w_updt = self.learning_rate / (np.sqrt(v_hat) + self.eps) * m_hat

        return w - self.w_updt
