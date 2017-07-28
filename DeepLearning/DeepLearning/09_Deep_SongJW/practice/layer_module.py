import numpy as np

class Sigmoid:
    def __init__(self):
        self.y = None

    def forward(self, x):
        self.y = 1 / (1 + np.exp(-x))
        return self.y

    def backward(self, dout):
        y = self.y
        return dout * y * (1 - y)

class ReLU:
    def __init__(self):
        self.x_bool = None

    def forward(self, x):
        self.x_bool = (x <= 0)
        return np.maximum(0, x)

    def backward(self, dout):
        dout[self.x_bool] = 0
        return dout

class SoftmaxWithLoss:
    def __init__(self):
        self.y = None
        self.t = None

    def forward(self, x, t):
        x = x.T
        x = x - np.max(x, axis = 0)
        x_exp = np.exp(x)
        x_exp_sum = np.sum(x_exp, axis=0)
        self.y = (x_exp / x_exp_sum).T
        self.t = t
        loss = - np.sum(t * np.log(self.y + 1e-8)) / t.shape[0]
        return loss
        # x = x - np.max(x, axis = 1).reshape(x.shape[0],1)
        # x_exp = np.exp(x)
        # x_exp_sum = np.sum(x_exp, axis = 1).reshape(x.shape[0],1)
        # self.y = x_exp / x_exp_sum
        # self.t = t
        # loss = - np.sum(t*np.log(self.y + 1e-7))
        # return loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size

class Affine:
    def __init__(self, W, b, lr):
        self.lr = lr
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
        self.dx = np.dot(dout, self.W.T)
        return self.dx

    def gradient_descent(self):
        self.W -= self.dW * self.lr
        self.b -= self.db * self.lr

if __name__ == '__main__':
    x = np.array([1,2], ndmin=2)
    w = np.array([[1,3,5],[2,4,6]])
    b = np.array([1,2,3])
    out = np.array([6, 13, 20], ndmin=2)

    a = Affine(w, b, 0.1)
    print(a.forward(x))
    print(a.backward(out))