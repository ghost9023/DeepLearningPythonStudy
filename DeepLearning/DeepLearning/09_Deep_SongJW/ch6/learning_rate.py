import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

# 부모 클래스 - 자식 클래스가 가질 공통적인 속성, 메소드 선언
class Optimizer :

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        '''
        초기화
        :param point: 현재 위치 : numpy array
        :param iter_num: 최적화 횟수 : int
        :param small_lr: 학습률1 : float
        :param proper_lr: 학습률2 : float
        :param large_lr: 학습률3 : float
        '''
        self.dict_point = defaultdict(lambda: np.array(point))  # 학습률별로 현재 점의 위치를 저장 - 딕셔너리 기본값 설정
        self.dict_trace = defaultdict(lambda: {'x': [point[0]], 'y': [point[1]]}) # 점의 궤적 저장
        self.iter_num = iter_num
        self.lr_tup = (large_lr, proper_lr, small_lr)   # 학습률 튜플

    def grad(self, point):
        '''
        f(x,y) = (x**2)/20 + y**2 함수의 기울기를 반환하는 함수
        :param x: 점의 위치 : numpy array
        :return: 점의 위치에서의 기울기 : numpy array
        '''
        return np.array([1 / 10, 2]) * point

    def method(self, point, lr):
        '''
        최적화 방법
        :param point: 점의 위치 : numpy array 
        :param lr: 학습률 : float
        :return: 점의 위치 : numpy array
        '''
        pass

    def optimize(self):
        '''
        최적화 실행
        :return: -
        '''
        for i in range(self.iter_num):  # 설정한 최적화 횟수만큼 수행
            for lr in self.lr_tup:  # 학습률 리스트에 속한 학습률별로 최적화 수행, 과정 저장
                self.dict_point[lr] = self.method(self.dict_point[lr], lr)  # 점 위치 갱신
                self.dict_trace[lr]['x'].append(self.dict_point[lr][0]) # 점의 궤적 저장
                self.dict_trace[lr]['y'].append(self.dict_point[lr][1])

# 최적화 방법을 클래스로 선언, Optimizer 클래스 상속
class SGD(Optimizer):
    '''
    경사감소법
    '''
    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)

    def method(self, point, lr):
        '''
        경사감소법 - 현재 점의 위치에서 학습률 * 미분만큼 이동
        :param point: 점의 위치 : numpy array 
        :param lr: 학습률 : float
        :return: 새로운 점의 위치 : numpy array
        '''
        return point - lr * self.grad(point)

class Momentum(Optimizer):
    '''
    모멘텀
    '''
    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)
        self.dict_v = defaultdict(lambda : np.zeros_like(point))    # 학습률별로 이전 모멘텀을 유지하기 위한 딕셔너리
        self.alpha = .9 # 모멘텀의 유지비율

    def method(self, point, lr):
        '''
        모멘텀 - 모멘텀을 학습률 * 미분만큼 가속하고 모멘텀만큼 이동.
        :param point: 점의 위치 : numpy array
        :param lr: 학습률 : flaot
        :return: 갱신된 점의 위치 : numpy array
        '''
        self.dict_v[lr] *= self.alpha
        self.dict_v[lr] -= lr * self.grad(point)
        return point + self.dict_v[lr]

class AdaGrad(Optimizer):
    '''
    AdaGrad
    '''
    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)
        self.dict_h = defaultdict(lambda : np.zeros_like(point))    # 학습률을 조정할 변수 h 를 담음
        self.epsilon = 1e-8

    def method(self, point, lr):
        '''
        AdaGrad - 점의 이동 방향별로 학습률을 조절함
        :param point: 점의 위치 : numpy array 
        :param lr: 학습률 : float
        :return: 갱신된 점의 위치 : numpy array
        '''
        gradient = self.grad(point)
        self.dict_h[lr] += gradient ** 2    # 현재 점에서의 미분의 각 원소별 제곱에 해당하는 값을 저장
        return point - lr * (1/np.sqrt(self.dict_h[lr] + self.epsilon)) * gradient  # 학습률에 h**(-0.5) 를 곱해 학습률을 감소시킴

class Adam(Optimizer):

    def __init__(self, point, iter_num, small_lr, proper_lr, large_lr, beta1=.9, beta2=.999):
        super().__init__(point, iter_num, small_lr, proper_lr, large_lr)
        self.epsilon = 1e-8
        self.beta1 = beta1
        self.beta2 = beta2
        self.m = defaultdict(lambda : np.zeros_like(point))
        self.v = defaultdict(lambda : np.zeros_like(point))

    def method(self, point, lr, iter):
        '''
        Adam - 모름
        :param point: 점의 위치 : numpy array 
        :param lr: 학습률 : float
        :param iter: 최적화횟수 : int
        :return: 갱신된 점의 위치 : numpy array
        '''
        gradient = self.grad(point)
        beta1, beta2 = self.beta1, self.beta2
        self.m[lr] = beta1 * self.m[lr] + (1 - beta1) * gradient
        self.v[lr] = beta2 * self.v[lr] + (1 - beta2) * (gradient ** 2)
        m_t = self.m[lr] / (1 - beta1 ** iter)
        v_t = self.v[lr] / (1 - beta2 ** iter)
        return point - (lr / np.sqrt(v_t + self.epsilon)) * m_t

    def optimize(self):
        '''
        최적화 실행 - 최적화 횟수를 인수로 요구하므로 오버라이딩
        :return: -
        '''
        for i in range(self.iter_num):
            for lr in self.lr_tup:
                self.dict_point[lr] = self.method(self.dict_point[lr], lr, i+1)
                self.dict_trace[lr]['x'].append(self.dict_point[lr][0])
                self.dict_trace[lr]['y'].append(self.dict_point[lr][1])

# 점의 시작 위치
x = np.array([-7., 2.])

# 최적화 방법별로 객체 생성, 설정한 최적화 횟수별로 최적화 수행
# small_lr, proper_lr, large_lr = 학습률 설정
sgd = SGD(point=x, iter_num=50, small_lr=.4, proper_lr=.95, large_lr=1.003)
sgd.optimize()

mmt = Momentum(point=x, iter_num=25, small_lr=.03, proper_lr=.091, large_lr=.24)
mmt.optimize()

adg = AdaGrad(point=x, iter_num=30, small_lr=.5, proper_lr=1, large_lr=1.5)
adg.optimize()

# ad = Adam(point=x, iter_num=50, small_lr=.1, proper_lr=.2, large_lr=.4)
ad = Adam(point=x, iter_num=30, small_lr=.2, proper_lr=.3, large_lr=.35)
ad.optimize()

# 각 최적화 방법별로 생성한 객체를 담음
opt_lst = [sgd, mmt, adg, ad]

# 최적화 방법별로 plot 생성
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