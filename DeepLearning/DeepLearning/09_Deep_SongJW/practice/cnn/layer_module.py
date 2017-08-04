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
    def __init__(self, W, b, stride, pad):
        self.W = W.flatten().reshape(W.shape[0], -1).T
        self.b = b
        self.x = None
        self.stride = stride
        self.pad = pad
        self.dW = None
        self.db = None
        self.os_W = W.shape
        self.os_x = None

    def forward(self, x):
        self.os_x = x.shape
        N, C, H, W = x.shape
        FN, C, FH, FW = self.os_W
        OH, OW = int((H + 2 * self.pad - FH) / self.stride + 1), int((W + 2 * self.pad - FW) / self.stride + 1)
        self.x = img2col(x, FH, FW, self.stride, self.pad)
        xW = np.dot(self.x, self.W) + self.b
        return xW.reshape(N, OH, OW, FN).transpose(0, 3, 1, 2)

    def backward(self, dout):
        N, C, H, W = self.os_x
        FN, C, FH, FW = self.os_W
        OH, OW = int((H + 2 * self.pad - FH) / self.stride + 1), int((W + 2 * self.pad - FW) / self.stride + 1)
        dout = dout.transpose(0, 2, 3, 1).reshape(-1, FN)
        self.db = np.sum(dout, axis=0)
        self.dW = np.dot(self.x.T, dout)
        dx = np.dot(dout, self.W.T).reshape(-1, C, FH, FW)
        base = np.zeros((N, C, H, W))
        n = 0
        for i in range(N):
            for h in range(0, OH, self.stride):
                for w in range(0, OW, self.stride):
                    base[i, :, h:h+FH, w:w+FW] += dx[n]
                    n += 1
        return base

class pooling:

    def __init__(self):
        pass

    def forward(self, x):
        N, C, H, W = x.shape
        x = x.reshape(1, 1, -1, 4)
        x = img2col(x, 2, 2, 2)
        max_ind = np.argmax(x, axis=1)
        x = x[[i for i in range(x.shape[0])], max_ind].reshape(N, C, int(H / 2), int(W / 2))
        return x

if __name__ == '__main__':
    pass
    # x1 = np.arange(32).reshape(1, 2, 4, 4)
    # w1 = np.arange(18).reshape(1, 2, 3, 3)
    # b1 = 1
    # con = conv(w1, b1, 1, 0)
    # dout = con.forward(x1)
    # print(dout)
    # print(con.backward(dout))
    # print(con.W.shape)
    # print(con.dW.shape)
    # print(con.db)

    p = pooling()
    x = np.arange(48 * 2).reshape(2, 3, 4, 4)
    print(p.forward(x))