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
        self.dict_v = defaultdict(lambda : np.zeros_like(point))
        self.alpha = .9

    def method(self, point, lr):
        self.dict_v[lr] *= self.alpha
        self.dict_v[lr] -= lr * self.grad(point)
        return point + self.dict_v[lr]

class AdaGrad(Optimizer):

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)
        self.dict_h = defaultdict(lambda : np.zeros_like(point))
        self.epsilon = 1e-8

    def method(self, point, lr):
        gradient = self.grad(point)
        self.dict_h[lr] += gradient ** 2
        return point - lr * (1/np.sqrt(self.dict_h[lr] + self.epsilon)) * gradient

class Adam(Optimizer):

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr, beta1=.9, beta2=.999):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)
        self.epsilon = 1e-8
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = defaultdict(lambda : np.zeros_like(point))
        self.v = defaultdict(lambda : np.zeros_like(point))

    def method(self, point, lr, iter):
        gradient = self.grad(point)
        beta1, beta2 = self.beta1, self.beta2
        self.m[lr] = beta1 * self.m[lr] + (1 - beta1) * gradient
        self.v[lr] = beta2 * self.v[lr] + (1 - beta2) * (gradient ** 2)
        m_t = self.m[lr] / (1 - beta1 ** iter)
        v_t = self.v[lr] / (1 - beta2 ** iter)
        return point - (lr / np.sqrt(v_t + self.epsilon)) * m_t

    def optimize(self):
        for i in range(self.iter_num):
            for lr in self.lr_tup:
                self.dict_point[lr] = self.method(self.dict_point[lr], lr, i+1)
                self.dict_trace[lr]['x'].append(self.dict_point[lr][0])
                self.dict_trace[lr]['y'].append(self.dict_point[lr][1])

x = np.array([-7., 2.])

sgd = SGD(point=x, iter_num=50, small_lr=.4, proper_lr=.95, large_lr=1.003)
sgd.optimize()

mmt = Momentum(point=x, iter_num=25, small_lr=.03, proper_lr=.091, large_lr=.24)
mmt.optimize()

adg = AdaGrad(point=x, iter_num=30, small_lr=.5, proper_lr=1, large_lr=1.5)
adg.optimize()

# ad = Adam(point=x, iter_num=50, small_lr=.1, proper_lr=.2, large_lr=.4)
ad = Adam(point=x, iter_num=30, small_lr=.2, proper_lr=.3, large_lr=.35)
ad.optimize()

opt_lst = [sgd, mmt, adg, ad]

plt.figure(figsize=(16, 6))
for opt, i in zip(opt_lst, range(1,5)):
    plt.subplot(2, 2, i)
    # 함수 Z = (X **2)/20 + Y **2 를 그림 - 등고선 형태
    x = np.linspace(-8, 8, 100)  # x, y 의 범위
    y = np.linspace(-3, 3, 100)
    X, Y = np.meshgrid(x, y)
    Z = (X ** 2) / 20 + Y ** 2  # 그리고자 하는 함수
    levels = np.arange(0, 40, 1)  # 등고선의 범위와 등고선간 간격
    CS = plt.contour(X, Y, Z, levels=levels)
    plt.clabel(CS, inline=1, fontsize=10)  # 등고선의 값 표시
    plt.grid()
    plt.xlim(-8, 8)
    plt.ylim(-3, 3)

    for lr, format in zip(opt.lr_tup, ['.-', 'v-', 'o-']):
        plt.plot(opt.dict_trace[lr]['x'], opt.dict_trace[lr]['y'], format,ms=5 ,label=lr)
        plt.title(opt.__class__.__name__+' - iteration :'+str(opt.iter_num))
        plt.legend(title='learning rate')
plt.show()