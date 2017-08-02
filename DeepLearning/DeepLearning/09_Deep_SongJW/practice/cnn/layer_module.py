import numpy as np

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        x = x.T - np.max(x, axis=1)
        x = np.exp(x)
        sum_exp_x = np.sum(x, axis=0)
        self.y = (x / sum_exp_x).T
        self.t = t
        return -np.sum(t * np.log(self.y + 1e-8)) / self.y.shape[0]

    def backward(self, dout):
        return dout * (self.y - self.t) / self.y.shape[0]

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None

    def forward(self, x):
        self.x = x
        return np.dot(x, self.W)

    def backward(self, dout):
        self.W = np.dot(self.x.T, dout)
        self.b = np.sum(dout, axis=0)
        return np.dot(dout, self.W.T)




if __name__ == '__main__':
    pass