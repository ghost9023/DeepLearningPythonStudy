import numpy as np
from practice.cnn.layer_module import *
from practice.cnn.function_module import *

class network:
    def __init__(self, lr=.1, std_scale=.01):
        # def get_HE_std_scale(n):
        #     return (2 / n) ** .5
        self.temp_loss = None
        self.layers = []
        self.lr = lr
        self.layers.append(Conv(std_scale * np.random.randn(20, 1, 5, 5), np.zeros((20)), stride=1, pad=0))
        self.layers.append(ReLU())
        self.layers.append(Pooling())
        self.layers.append(Conv(std_scale * np.random.randn(50, 20, 5, 5), np.zeros((50)), stride=1, pad=0))
        self.layers.append(ReLU())
        self.layers.append(Pooling())
        self.layers.append(Conv(std_scale * np.random.randn(500, 50, 4, 4), np.zeros((500)), stride=1, pad=0))
        self.layers.append(ReLU())
        self.layers.append(Affine(std_scale * np.random.randn(500, 50), np.zeros(50)))
        self.layers.append(ReLU())
        self.layers.append(Affine(std_scale * np.random.randn(50, 10), np.zeros(10)))
        self.layers.append(SoftmaxWithLoss())
        for i in self.layers:
            print(i.__class__.__name__)

    def predict(self, x):
        for i in self.layers[:-1]:
            x = i.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.layers[-1].forward(y, t)
        return loss

    def gradient_descent(self, x, t):
        self.temp_loss = self.loss(x, t)
        layer_lst = self.layers.copy()
        layer_lst.reverse()
        dout = 1
        for i in layer_lst:
            dout = i.backward(dout)

        for i in self.layers:
            if i.__class__.__name__ in ['Affine', 'Conv']:
                i.W -= self.lr * i.dW
                i.b -= self.lr * i.db

    def accuracy(self, x, t):
        y = self.predict(x)
        acc = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / float(y.shape[0])
        return acc