import numpy as np
from practice.cnn.function_module import img2col

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.os_x = None
        self.dW = None
        self.db = None

    def forward(self, x):
        self.os_x = x.shape
        if x.ndim == 4 :
            x = x.reshape(x.shape[0], -1)
        self.x = x
        return np.dot(x, self.W) + self.b

    def backward(self, dout):
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = np.dot(dout, self.W.T)
        return dx.reshape(self.os_x)

class ReLU:
    def __init__(self):
        self.map = None

    def forward(self, x):
        # print(x.shape)
        self.map = (x <= 0)
        x[self.map] = 0
        return x

    def backward(self, dout):
        # print('a', dout.shape)
        # print('b', self.map.shape)
        dout[self.map] = 0
        return dout

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


class Conv:
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

class Pooling:

    def __init__(self):
        self.map = None
        self.max_ind = None
        self.x_shape = None
        self.max_val = None

    def forward(self, x):
        self.x_shape = x.shape
        N, C, H, W = x.shape
        self.map = np.zeros_like(x)
        x = x.reshape(1, 1, -1, 4)
        x = img2col(x, 2, 2, 2)
        max_ind = np.argmax(x, axis=1)
        self.max_ind = max_ind
        self.max_val = x[np.array([i for i in range(x.shape[0])]), max_ind]
        x = x[np.array([i for i in range(x.shape[0])]), max_ind].reshape(N, C, int(H / 2), int(W / 2))
        return x

    def backward(self, dout):
        N, C, H, W = self.x_shape
        max_ind_iter = iter(self.max_ind)
        ind_list = [[], [], [], []]
        for n in range(N):
            for c in range(C):
                for h in range(int(H/2)):
                    for w in range(int(W/2)):
                        row, col = divmod(max_ind_iter.__next__(), 2)
                        ind_list[0].append(n)
                        ind_list[1].append(c)
                        ind_list[2].append(2 * h + row)
                        ind_list[3].append(2 * w + col)
        temp = np.zeros((N, C, H, W))
        temp[ind_list[0], ind_list[1], ind_list[2], ind_list[3]] = dout.flatten()

        return temp


if __name__ == '__main__':
    pass