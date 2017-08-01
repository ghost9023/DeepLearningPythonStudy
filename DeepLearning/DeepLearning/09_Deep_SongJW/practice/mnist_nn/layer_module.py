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

class BatchNormalizaition:
    def __init__(self):
        self.gamma = 1
        self.beta = 0
        self.dgamma = None
        self.dbeta = None
        self.xhat = None
        self.ivar = None
        self.xmu = None
        self.sqrtvar = None
        self.var = None

    def forward(self, x):
        x = x.T
        mu = np.sum(x, axis=0) / x.shape[0]
        self.xmu = x - mu
        sq = self.xmu**2
        self.var = np.sum(sq, axis=0) / sq.shape[0]
        self.sqrtvar = np.sqrt(self.var)
        self.ivar = 1 / self.sqrtvar
        self.xhat = self.xmu * self.ivar
        gxhat = self.gamma * self.xhat
        out = gxhat + self.beta

        return out.T

    def backward(self, dout):
        dout = dout.T
        D, N = dout.shape
        self.dbeta = np.sum(dout, axis = 0)
        dgxhat = dout
        self.dgamma = np.sum(self.xhat * dgxhat, axis=0)
        dxhat = dgxhat * self.gamma
        dxmu1 = dxhat * self.ivar
        divar = np.sum(dxhat * self.xmu, axis=0)
        dsqrtvar = -1 * divar / (self.sqrtvar**2)
        dvar = dsqrtvar / (2 * np.sqrt(self.var))
        dsq = np.ones((D, N)) * dvar / D
        dxmu2 = 2 * self.xmu * dsq
        dx1 = dxmu1 + dxmu2
        dmu = -1 * np.sum(dxmu1 + dxmu2, axis=0)
        dx2 = np.ones((D, N)) * dmu / D
        dx = dx1 + dx2

        return dx.T

if __name__ == '__main__':
    x = np.array([[1,2,3,4], [4,5,6,7], [7,8,9,10]])
    n = BatchNormalizaition()
    output = n.forward(x)
    print(output)
    print(n.backward(output))

