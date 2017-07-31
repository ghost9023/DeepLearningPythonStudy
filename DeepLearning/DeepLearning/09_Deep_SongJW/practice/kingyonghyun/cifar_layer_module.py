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
        loss = - np.sum(np.log(self.y + 1e-8)) / t.shape[0]

        return loss

    def backward(self, dout):
        batch_size = self.t.shape[0]
        return (self.y - self.t) / batch_size

class Affine:
    def __init__(self, W, b, lr):
        self.lr = lr
        self.params = {'W' : W, 'b' : b}
        self.x = None
        self.dx = None
        self.grad = {}
        self.v = {}
        self.a = .9
        self.h = {}
        self.delta = 1e-8
        self.beta1, self.beta2 = 0.9, 0.999
        self.adam_params = {}
        self.lst = ['W', 'b']
        self.iter = 1


    def forward(self, x):
        self.x = x
        return np.dot(x, self.params['W']) + self.params['b']

    def backward(self, dout):
        self.grad['W'] = np.dot(self.x.T, dout)
        self.grad['b'] = np.sum(dout, axis=0)
        self.dx = np.dot(dout, self.params['W'].T)
        return self.dx

    def gradient_descent(self, method):
        if method == 'SGD':
            for i in self.lst:
                self.params[i] -= self.grad[i] * self.lr
        elif method == 'Momentum':
            if self.v == {}:
                for i in self.lst:
                    self.v[i] = np.zeros_like(self.params[i])
            for i in self.lst:
                self.v[i] = self.a * self.v[i] - self.lr * self.grad[i]
                self.params[i] += self.v[i]
        elif method == 'AdaGrad':
            if self.h == {}:
                for i in self.lst:
                    self.h[i] = np.zeros_like(self.params[i])
            for i in self.lst:
                self.h[i] += self.grad[i] * self.grad[i]
                self.params[i] -= self.lr / np.sqrt(self.h[i] + self.delta) * self.grad[i]
        elif method == 'Adam':
            beta1, beta2 = self.beta1, self.beta2
            if self.adam_params == {}:
                for i in self.lst:
                    self.adam_params[i] = {'m':0, 'v':0}
            for i in self.lst:
                self.adam_params[i]['m'] = beta1 * self.adam_params[i]['m'] + (1 - beta1) * self.grad[i]
                self.adam_params[i]['v'] = beta2 * self.adam_params[i]['v'] + (1 - beta2) * (self.grad[i]**2)
                m_t = self.adam_params[i]['m'] / (1 - beta1 ** self.iter)
                v_t = self.adam_params[i]['v'] / (1 - beta2 ** self.iter)
                self.params[i] -= (self.lr / np.sqrt(v_t + self.delta)) * m_t
            self.iter += 1

if __name__ == '__main__':
    t = np.array([0,0,1]+[0]*7, ndmin=2)
    x = np.array([.01]*6+[.05, .3, .1, .5], ndmin=2)
    x2 = np.array([.01, .01, .9]+[.01]*6+[.02], ndmin=2)
    soft = SoftmaxWithLoss()
    print(soft.forward(x, t))
    print(soft.backward(1))
    soft2 = SoftmaxWithLoss()
    print(soft2.forward(x2, t))
    print(soft2.backward(1))