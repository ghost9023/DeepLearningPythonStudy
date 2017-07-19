import numpy as np
import matplotlib.pyplot as plt
from collections import defaultdict

def grad(x) :
    '''
    f(x,y) = (x**2)/20 + y**2 함수의 기울기를 반환하는 함수
    :param x: np.array : 점의 위치 
    :return: np.array : 점의 위치에서의 기울기
    '''
    return np.array([1/10, 2]) * x


# 최적화 함수
def SGD(x, lr) :
    '''
    확률적 경사 하강법
    학습률 추천 : .4 / .9 / 1.004
    :param x: np.array : 점의 위치
    :param lr: float : 학습률
    :return: np.array : 변경된 점의 위치
    '''
    return x-lr*grad(x)


# 함수 Z = (X **2)/20 + Y **2 를 그림 - 등고선 형태
x = np.linspace(-7.5, 7.5, 1000)    # x, y 의 범위
y = np.linspace(-2.5, 2.5, 1000)
X, Y = np.meshgrid(x, y)
Z = (X **2)/20 + Y **2  # 그리고자 하는 함수
plt.figure(figsize=(18,6))
levels = np.arange(0, 40, 1)    # 등고선의 범위와 등고선간 간격
CS = plt.contour(X, Y, Z, levels = levels)
plt.clabel(CS, inline=1, fontsize=10)   # 등고선의 값 표시
plt.grid()


# 초기 시작위치 설정
point = np.array([-7., 2.])


# 학습률, 학습 횟수 설정
small_lr = .4
proper_lr = .9
large_lr = 1.004
lr_tup = (large_lr, proper_lr, small_lr)
iter_num = 30   # 학습 횟수

# 점의 현재 위치와 점의 자취를 담을 딕셔너리 준비 - 시작 지점을 미리 넣어놓는다.
dict_point = defaultdict(lambda : np.array(point))
dict_trace = defaultdict(lambda : {'x':[point[0]], 'y':[point[1]]})

# 함수의 최소값의 위치인 (0, 0) 을 찾아가도록 경사감소법 수행
for i in range(iter_num):
    for lr in lr_tup:
        # dict_point[lr] -= lr * grad(dict_point[lr])
        dict_point[lr] = SGD(dict_point[lr], lr)
        dict_trace[lr]['x'].append(dict_point[lr][0])
        dict_trace[lr]['y'].append(dict_point[lr][1])

# 학습률별로 점의 이동 자취를 그림
for i, j in zip(lr_tup,['b.-', 'rv-', 'go-']):
    plt.plot(dict_trace[i]['x'], dict_trace[i]['y'], j,label=i)
plt.legend()
plt.show()