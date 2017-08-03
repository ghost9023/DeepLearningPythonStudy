import numpy as np
from practice.cnn.function_module import img2col

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


class conv:
    def __init__(self, f, b, stride=1, pad=0):
        self.f = f
        self.b = b
        self.stride = stride
        self.pad = pad

    def forward(self, x):
        FN, C, FH, FW = self.f.shape
        f = self.f.flatten().reshape(FN, -1)
        col_x = img2col(x, FH, FW, self.stride, self.pad)
        N, C, H, W = x.shape
        OH = int((H + 2*self.pad - FH) / self.stride + 1)
        OW = int((W + 2 * self.pad - FW) / self.stride + 1)
        x = np.dot(col_x, f.T) + self.b
        return x.reshape(N, OH, OW, -1).transpose(0,3,1,2)

    def backward(self, dout):
        pass

class pooling:
    def __init__(self, ph, pw, stride):
        self.ph = ph
        self.pw = pw
        self.stride = stride

    def forward(self, x):
        N, C, H, W = x.shape
        y = img2col(x, fh=self.ph, fw=self.pw, stride=self.stride)
        y = y.T.reshape(N, C, H, W).transpose(0, 1, 3, 2)
        max_y = np.max(y, axis=3)
        result = max_y.reshape(N, C, int(H/2), int(W/2))
        return result


if __name__ == '__main__':
    pass