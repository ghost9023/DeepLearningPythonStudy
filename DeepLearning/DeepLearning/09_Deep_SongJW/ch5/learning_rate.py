import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

class Optimizer :

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        self.dict_point = defaultdict(lambda: np.array(point))
        self.dict_trace = defaultdict(lambda: {'x': [point[0]], 'y': [point[1]]})
        self.iter_num = iter_num
        self.lr_tup = (large_lr, proper_lr, small_lr)

    def grad(self, point):
        '''
        f(x,y) = (x**2)/20 + y**2 함수의 기울기를 반환하는 함수
        :param x: np.array : 점의 위치 
        :return: np.array : 점의 위치에서의 기울기
        '''
        return np.array([1 / 10, 2]) * point

    def method(self, point, lr):
        pass

    def optimize(self):
        for i in range(self.iter_num):
            for lr in self.lr_tup:
                self.dict_point[lr] = self.method(self.dict_point[lr], lr)
                self.dict_trace[lr]['x'].append(self.dict_point[lr][0])
                self.dict_trace[lr]['y'].append(self.dict_point[lr][1])


class SGD(Optimizer):

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)

    def method(self, point, lr):
        return point - lr * self.grad(point)

class Momentum(Optimizer):

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)
        self.dict_v = defaultdict(int)
        self.alpha = .9

    def method(self, point, lr):
        self.dict_v[lr] *= self.alpha
        self.dict_v[lr] -= lr * self.grad(point)
        return point + self.dict_v[lr]

x = np.array([-7., 2.])

sgd = SGD(point=x, iter_num=50, small_lr=.4, proper_lr=.9, large_lr=1.003)
sgd.optimize()

mmt = Momentum(point=x, iter_num=25, small_lr=.03, proper_lr=.091, large_lr=.24)
mmt.optimize()

opt_lst = [sgd, mmt]

for opt in opt_lst:

    # 함수 Z = (X **2)/20 + Y **2 를 그림 - 등고선 형태
    x = np.linspace(-8, 8, 100)  # x, y 의 범위
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X ** 2) / 20 + Y ** 2  # 그리고자 하는 함수
    plt.figure(figsize=(8, 3))
    levels = np.arange(0, 40, 1)  # 등고선의 범위와 등고선간 간격
    CS = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)  # 등고선의 값 표시
    plt.grid()
    plt.xlim(-8, 8)
    plt.ylim(-3, 3)

    for lr, format in zip(opt.lr_tup, ['.-', 'v-', 'o-']):
        plt.plot(opt.dict_trace[lr]['x'], opt.dict_trace[lr]['y'], format, label=lr)
        plt.title(opt.__class__.__name__+' - iteration :'+str(opt.iter_num))
        plt.legend()
plt.show()