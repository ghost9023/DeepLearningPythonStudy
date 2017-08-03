# CHAPTER 4 신경망 학습
# 데이터 주도학습
# 특징을 추출하고 그 특징의 패턴을 기계학습으로 학습하는 방법
# 신경망은 이미지를 있는 그대로 학습한다. 즉, 특징을 사람이 설계하지 않는다.
# 신경망(딥러닝)은 종단간(end-to-end) 기계학습이라고 한다. 여기서 종단간은 처음부터 끝까지라는 의미이다.
# 신경망의 이점은 모든 문제를 같은 맥락에서 풀 수 있다는 점에 있다.
# 예를 들어 숫자를 인식하든, 개를 인식하든, 사람 얼굴을 인식하든 세부사항에 관계없이 주어진 데이터를 온전히 학습하고,
# 주어진 문제의 패턴을 발견하려고 시도한다.

# 훈련데이터와 시험데이터
# 기계학습 문제는 데이터를 훈련데이터와 시험데이터로 나눠 학습과 실험을 수행하는 것이 일반적이다.
# 우리가 원하는 것은 범용적으로 사용할 수 있는 모델을 만드는 것이기 때문에 과적합(오버피팅)을 확인하기 위해 시험데이터가 필요하다.
# 시험데이터는 훈련데이터에 포함되어 있지 않은 데이터

# 손실함수
# 정답이 아닌 나머지 모두는 t가 0이기 때문에 손실함수에 영향을 주지 않는다.
# 손실함수는 현재의 상태를 측정하는 하나의 지표이다.
# 일반적으로 평균제곱오차와 교차엔트로피오차를 사용한다.
# 평균제곱오차
# E = 1/2 * sum((y-t)**2)
# 여기서 y는 신경망의 출력 값(0.284783), t는 정답레이블(1), k는 차원 수를 나타낸다.
# 예시 y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.0,0.0,0.0]  |  t = [0,0,0,0,0,0,1,0,0,0,0]
# 위의 t같이 정답을 표시하는 것을 원-핫 인코딩이라고 한다.
# 평균제곱오차는 각 원소의 출력값들과 정답레이블들의 차를 제곱한 후 그 총합을 구한다.
# 파이썬으로 구현
import numpy as np
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])   # 소프트맥스 통과해서 나온 값 y
mean_squared_error(y, t)   # 결과값이 맞을때

t = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
mean_squared_error(y, t)   # 결과값이 틀릴때

# 교차엔트로피오차
# E = -sum(t*np.log(y))
# 여기에서 log는 밑이 e인 자연로그
# 0에 가까울 수록 정답에 가깝다.
# 교차엔트로피함수 구현
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))   # 각 원소끼리 계산해서 합
# 여기에서 y와 t는 넘파이 배열이다. 왜냐하면 np.log()함수에 0이 들어가면(혹은 컴퓨터 특성상 0에 가까운값) 무한대가 되기 때문이다.
# 즉, 무한대가 발생하지 않도록 delta를 삽입
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
cross_entropy_error(y,t)



# 미니배치학습(손실함수)
# 기계학습은 훈련데이터에 대한 손실함수의 값을 구하고 그 값을 최대한 줄여주는 매개변수를 찾아내는 것이다.
# 이렇게 하려면 모든 데이터를 대상으로 손실함수의 값을 구해야 한다.
# 하나씩 처리하면 오래 걸리기 때문에 이럴 경우 훈련데이터로부터 일부만 골라 학습을 진행한다. (표본추출)
# 여러개의 데이터의 손실함수에 대한 평균을 구하는 것
# 이때 그래프의 y축이 E(손실함수)의 평균이기 때문에 각 y(x축)의 값에 따라 극소값이 만들어진다.
# 여기서 이 일부를 mini_batch라고 한다.
# 아래는 무작위로 지정한 수의 데이터를 골라내는 식
import sys, os
sys.path.append(os.pardir)   # 부모 디렉토리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)
train_size = x_train.shape[0]   # 60000개
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # 60000개 중에 10개 무작위로 선택, 나중에 인덱스로 사용
# array([8013, 14666, 58452, 4582, 7895, 35245....])
x_batch = x_train[batch_mask]   # 이건 왜 따로 뽑는거야?
t_batch = t_train[batch_mask]   # 인덱스로 사용해서 10개 선택하고 그것과 비교

def cross_entropy_error(y,t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)   # 2차원으로 바꿔주기
        y = y.reshape(1, y.size)   # 2차원으로 바꿔주기
    batch_size = y.shape[0]        # 들어온 행의 수대로 배치사이즈 할당
    return -np.sum(t * np.log(y)) / batch_size   # 다 더해서 배치사이즈로 나눠주기, 즉 평균 구하는 것

# 레이블이 원핫 인코딩이 아니더라도 동작하는 메서드
def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size



# 원-핫 인코딩일 때 t가 0인 원소는 교차엔트로피 오차도 0이므로, 그 계산은 무시해도 좋다.

# 왜 손실함수를 설정할까?
# 신경망 학습에서는 최적의 매개변수를 탐색할 때 손실함수의 값을 가능한 한 작게 하는 매개변수값을 찾는다.
# 이때 매개변수의 미분(기울기)를 계산하고, 그 미분값을 단서로 매개변수의 값을 서서히 갱신하는 과정을 반복한다.
# 신경망을 학습할 때 정확도를 지표로 삼아서는 안된다.
# 일단 미분을 할 때 0.0001을 사용하여 하기 때문에 정확도를 기준으로 하면 계단함수가 만들어지게 된다.(기울기 0)
# 즉 y값이 변해도 정확도 변화가 없는 경우가 있다.

# 수치미분
# 수치미분은 정확도를 위해
# 중앙차분(중심차분)을 구하고 반올림오차문제를 예방하기 위해 미분하는 값으로 0.0001을 사용한다.
def numerical_diff(f, x):
    h = 1e-4   # 0.0001
    return (f(x+h) - f(x-h)) / (2*h)
# 값 하나의 미분값 구하는 함수

# y = (0.01*x)**2 + 0.1*x 를 파이썬으로 구현해보고 이 식을 미분하는 함수를 만들어보자 (예로 든 이차함수)
import numpy as np
import matplotlib.pylab as plt
def function_1(x):
    return 0.01*x**2 + 0.1*x
# 위의 함수 그리기
x = np.arange(0.0, 20.0, 0.1)
y = function_1(x)
plt.xlabel('x')
plt.ylabel('f(x)')
plt.plot(x, y)
plt.show()

numerical_diff(function_1, 5)   # 위의 식에서 x가 5일 때 기울기
numerical_diff(function_1, 10)   # 위의 식에서 x가 10일 때 기울기
# 이렇게 계산한 미분값이 x에 대한 f(x)의 변화량, 즉 함수의 기울기에 해당
# 0.01x^2 + 0.1x의 해석적 미분은 0.02x + 0.1  <- 진정한 미분

# 편미분
# 앞의 예와 달리 변수가 2개 이상이라는 점에 주의해야 한다.
# f(x0, x1) = x0^2 + x1^2을 파이썬으로 구현
def function_2(x):
    return x[0]**2 + x[1]**2
# 그릇을 엎어놓은 모양의 3차원함수가 만들어진다.
# 이와 같이 변수가 여러개인 상황에서 변수 하나에 대한 미분을 편미분이라고 한다.
# 일단 미분을 구하는데에는 상수(y절편)이 필요 없기 때문에 무시해도 괜찮아요~
def function_tmp1(x0):
    return x0*x0 + 4.0**2
numerical_diff(function_tmp1, 3.0)

def function_tmp1(x0):
    return x0*x0
numerical_diff(function_tmp1, 3.0)
# 기울기가 거의 같다.(6.000000000000378), 차이는 수치미분에 의한 차이로 해석됨

def function_tmp2(x1):
    return 3.0**2 + x1*x1
numerical_diff(function_tmp2, 4.0)
# 7.999999999999119

# 기울기
# 앞절의 예에서는 x0과 x1의 편미분을 변수별로 따로 계산했다.
# 두 변수를 동시에 계산하고 싶다면 어떻게 해야할까? (가령 x0=3, x1=4일 때)
# (x0, x1, x2, x3)의 기울기를 (x0미분, x1미분, x2미분, x3미분)과 같이 벡터의 형태로 정리한 것을 기울기(gradient)라고 한다.
# 기울기를 구하는 함수구현
def numerical_gradient(f, x):
    h = 1e-4   # 0.0001
    grad = np.zeros_like(x)    # x와 형상(shape)이 같은 배열을 생성
    for idx in range(x.size):  # .size는 배열 안에 있는 원소의 수
        tmp_val = x[idx]   # 원래값을 저장해놓을 변수
        x[idx] = tmp_val + h
        fxh1 = f(x)   # y값 구하기
        x[idx] = tmp_val - h
        fxh2 = f(x)   # y값 구하기
        grad[idx]= (fxh1-fxh2) / (2*h)   # 기울기 구하기
        x[idx] = tmp_val   # 값 복원
    return grad

def function_2(x):
    return x[0]**2 + x[1]**2

numerical_gradient(function_2, np.array([3.0, 4.0]))   # array([ 6.,  8.])
numerical_gradient(function_2, np.array([0.0, 2.0]))   # array([ 0.,  4.])
numerical_gradient(function_2, np.array([3.0, 0.0]))   # array([ 6.,  0.])

# 경사법(경사하강법)
# 손실함수가 최소값이 될때의 매개변수값을 찾아야한다.
# 기울기가 가리키는 쪽은 각 장소에서 함수의 출력값을 가장 크게 줄이는 방향
# 각 지점에서 함수의 값을 낮추는 방안을 제시하는 지표가 기울기이지만
# 기울기가 가리키는 곳에 실제로 함수의 최소값이 있는지 보장할 수 없다.
# 함수가 최소값, 극소값, 안장점이 되는 장소의 기울기가 모두 0이기 때문에 기울기가 0이라고 해서 반드시 최소값이라고 볼 수 없다.
# 경사법을 함수로 구현하기
def gradient_descent(f, init_x, lr=0.01, step_num=100):
    x = init_x
    for i in range(step_num):
        grad = numerical_gradient(f, x)   # 기울기(미분값) 벡터 반환
        x -= lr * grad   # 기울기를 구한 후 그를 이용해 x값을 갱신해서 재귀
        # 기울기(미분)값에 learning_rate를 곱해서 입력값(x)에서 빼주는 과정을 반복한다.
        # 기울기는 경사감소의 방향과 정도를 결정한다. 기울기가 작아지면 이동하는 거리도 그만큼 줄어든다.
    return x

# 경사법으로 f(x0, x1) = x0^2 + x1^2의 최소값을 구하여라
def function_2(x):
    return x[0]**2 + x[1]**2
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.1, step_num=100)
# array([ -1.25592487e-19,   1.66263303e-19]), (0, 0)에서 기울기가 가장 작다. 0에 가깝다.

# 학습률이 너무 크거나 너무 작지 않아야 한다.
# 학습률이 너무 클 경우 발산해버릴 위험이 있고 학습률이 너무 작은 경우 로컬 미니마에 빠지거나 이동값이 아예 소멸할 수 있다.
init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=10.0, step_num=100)
# array([ -2.60465215e+13,   2.57982892e+13]), x값이 발산해버린다.

init_x = np.array([-3.0, 4.0])
gradient_descent(function_2, init_x=init_x, lr=0.001, step_num=100)
# array([-2.45570041,  3.27426722]), 값이 갱신되지 않고 끝나버린다.



# 신경망에서의 기울기!!!!!!
# 신경망 학습에서도 기울기를 구해야 한다.
# 가중치 매개변수에 대한 손실함수의 기울기를 말한다.
# 간단한 신경망을 예로 들어 실제로 기울기를 구하는 코드 구현
import sys, os
sys.path.append(os.pardir)
import numpy as np
# from common.functions import softmax, cross_entropy_error  # 밑에 직접 적어놓음
# from common.gradient import numerical_gradient             # 밑에 직접 적어놓음

def softmax(a):                 # 소프트맥스 @@@@
    c = np.max(a)
    exp_a = np.exp(a-c)   # 제일 큰놈 빼주기, 오버플로 빼주기
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def cross_entropy_error(y,t):   # 크로스 엔트로피 에러 @@@@
    if y.ndim == 1:
        t = t.reshape(1, t.size)   # 2차원으로 바꿔주기
        y = y.reshape(1, y.size)   # 2차원으로 바꿔주기
    batch_size = y.shape[0]        # 들어온 행의 수대로 배치사이즈 할당
    return -np.sum(t * np.log(y)) / batch_size   # 다 더해서 배치사이즈로 나눠주기, 즉 평균 구하는 것

def numerical_gradient(f, x):   # 특정값에서 손실함수의 기울기를 벡터로 출력 @@@@, 독립변수가 여러개, 편미분 여러개 한번에 하기
    # 여기에는 x값에 net.W가 들어가기 때문에 h값을 더해주고 뺄때 net.W값이 갱신된다.
    h = 1e-4   # 0.0001
    grad = np.zeros_like(x)     # x와 형상(shape)이 같은 배열을 생성
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # .size는 배열 안에 있는 원소의 수
        idx = it.multi_index
        tmp_val = x[idx]    # 원래값을 저장해놓을 변수
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)   # y값 구하기, f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)   # y값 구하기, f(x-h)
        grad[idx]= (fxh1-fxh2) / (2*h)   # 기울기 구하기
        x[idx] = tmp_val   # 값 복원
        it.iternext()
        return grad

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2, 3)   # 정규분포로 초기화, 즉 0에 가까울 수록 빈도가 많아진다는 뜻이다. 가중치 래덤
    def predict(self, x):
        return np.dot(x, self.W)
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)   # 손실함수까지의 과정, 값은 하나가 나오지만 x에 따라 변하는 함수가 나온다고 이해하면 편하다.
        return loss

net = simpleNet()
print('net.W', net.W)   # 가중치 매개변수
x = np.array([0.6, 0.9])
p = net.predict(x)   # 랜덤가중치와 x값의 연산, 확인용!
print('p', p)
np.argmax(p)   # 최대값의 인덱스 출력, 0
t = np.array([0,0,1])
net.loss(x, t)

# 이어서 기울기를 구해보자!!!!@#!@#@!#@!#@!#@!#!@#!@#!@#!@
# numerical_gradient(f, x)를 사용해서 구하면 된다.
def f(W):
    return net.loss(x, t)   # x는 입력(x = np.array([0.6, 0.9])), t는 레이블(t = np.array([0,0,1])), w는 더미(형식상)
dW = numerical_gradient(f, net.W)
print(dW)

# lambda()를 활용하여 위의 함수 구현
f = lambda w: net.loss(x, t)
dW = numerical_gradient(f, net.W)   # 손실함수에서 x축이 W일때의 기울기


## 학습알고리즘 구현@@@@@@@@@ 하기
# 전제 : 신경망에는 적응 가능한 가중치와 편향이 있고, 이 가중치와 편향을 훈련데이터에 적응하도록 조정하는 과정을 '학습'이라고 한다.
# 1단계-미니배치 : 훈련데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실함수 값을 줄이는 것이 목표이다.
# 2단계-기울기 산출 : 미니배치의 손실 값을 줄이기 위해 각 가중치 매개변수의 기울기를 구합니다. 기울기는 손실함수의 값을 가장 작게하는 방향을 제시한다.
# 3단계-매개변수 갱신 : 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
# 4단계-반복 : 1~3단계를 반복한다.
# 손실함수의 모양은 알 수 없다.@@@@
# 2층 신경망 클래스 구현하기
# coding: utf-8
import numpy as np
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# from common.functions import *
# from common.gradient import numerical_gradient

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(a):  # 소프트맥스 @@@@
    c = np.max(a)
    exp_a = np.exp(a - c)  # 제일 큰놈 빼주기, 오버플로 빼주기
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):  # 크로스 엔트로피 에러 @@@@
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 2차원으로 바꿔주기
        y = y.reshape(1, y.size)  # 2차원으로 바꿔주기
    batch_size = y.shape[0]  # 들어온 행의 수대로 배치사이즈 할당
    return -np.sum(t * np.log(y)) / batch_size  # 다 더해서 배치사이즈로 나눠주기, 즉 평균 구하는 것


def numerical_gradient(f, x):  # 특정값에서 손실함수의 기울기를 벡터로 출력 @@@@, 독립변수가 여러개, 편미분 여러개 한번에 하기
    # 여기에는 x값에 net.W가 들어가기 때문에 h값을 더해주고 뺄때 net.W값이 갱신된다.
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상(shape)이 같은 배열을 생성
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # .size는 배열 안에 있는 원소의 수
        idx = it.multi_index
        tmp_val = x[idx]  # 원래값을 저장해놓을 변수
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # y값 구하기, f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # y값 구하기, f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 기울기 구하기
        x[idx] = tmp_val  # 값 복원
        it.iternext()
        return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):   # predict()의 결과와 정답레이블을 바탕으로 교차엔트로피 오차를 구하도록 구현
        y = self.predict(x)

        return cross_entropy_error(y, t)   # 손실함수 출력

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블, W : dummy
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)   # 손실함수를 반환한다.

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])   # 이건 미리 만들어놓은 함수를 말한다. 위의 메서드와 이름은 같지만 다른 함수
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])   # x와 t에 의해 만들어진 손실함수에서 W나 b에 대한 미분값을 구한다. (수치미분사용)
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])   # numerical_gradient는 w값을 갱신하기 위해 사용된다.
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

# 위의 코드에서 사용하는 변수는 p.139 참조

net = TwoLayerNet(784, 100, 10)   # 입력 784열, 은닉 100노드, 출력 10개
net.params['W1'].shape  # (784, 100)
net.params['b1'].shape  # (100,)   # 은닉층에서 노드가 100개
net.params['W2'].shape  # (100,10)
net.params['b2'].shape  # (10,)    # 출력 10개

x = np.random.rand(100, 784)   # 더미입력데이터, 정규분포 난수
y = net.predict(x)
# grads 변수에는 params(weight, bias) 변수에 대응하는 각 매개변수의 기울기가 저장된다.
# numerical_gradient() 메서드를 사용한 손실함수에 대한 각 매개변수의 기울기(수치미분사용) grads 변수에 저장된다.

x = np.random.rand(100, 784)  # 0~1 사이의 숫자를 100*784의 행열로 만들기 (더미 입력 레이블)
t = np.random.rand(100, 10)   # 0~1 사이의 숫자를 100*10의 행열로 만들기 (더미 정답 레이블)
grads = net.numerical_gradient(x, t)  # 다시한번 정리하면 입력값x와 정답t가 반환하는 손실함수를 이용해 w와 b값을 수치미분한 값이 대응하는 값이 key에 담긴다.

grads['W1'].shape   # (784, 100)
grads['b1'].shape   # (100,)
grads['W2'].shape   # (100, 10)
grads['b2'].shape   # (10,)
# grads변수에는 대응하는 매개변수의 미분값이 들어가 있다.
# numerical gradient()는 수치미분방식으로 매개변수의 기울기를 계산한다. 다음 장에서는 오차역전파를 이용하여 이 기울기 계산을 고속으로 수행하는 방법을 설명한다.

#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
# 미니배치 학습 구현하기
# 미니배치 학습이란 훈련데이터 중 일부를 무작위로 꺼내고(미니배치), 그 미니배치에 대해서 경사법으로 매개변수를 갱신하는 것을 말한다.
# 학습 수행
import numpy as np
from dataset.mnist import load_mnist
# from two_layer_net import TwoLayerNet
import numpy as np
import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
# from common.functions import *
# from common.gradient import numerical_gradient

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(a):  # 소프트맥스 @@@@
    c = np.max(a)
    exp_a = np.exp(a - c)  # 제일 큰놈 빼주기, 오버플로 빼주기
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y


def cross_entropy_error(y, t):  # 크로스 엔트로피 에러 @@@@
    if y.ndim == 1:
        t = t.reshape(1, t.size)  # 2차원으로 바꿔주기
        y = y.reshape(1, y.size)  # 2차원으로 바꿔주기
    batch_size = y.shape[0]  # 들어온 행의 수대로 배치사이즈 할당
    return -np.sum(t * np.log(y)) / batch_size  # 다 더해서 배치사이즈로 나눠주기, 즉 평균 구하는 것


def numerical_gradient(f, x):  # 특정값에서 손실함수의 기울기를 벡터로 출력 @@@@, 독립변수가 여러개, 편미분 여러개 한번에 하기
    # 여기에는 x값에 net.W가 들어가기 때문에 h값을 더해주고 뺄때 net.W값이 갱신된다.
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상(shape)이 같은 배열을 생성
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # .size는 배열 안에 있는 원소의 수
        idx = it.multi_index
        tmp_val = x[idx]  # 원래값을 저장해놓을 변수
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # y값 구하기, f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # y값 구하기, f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 기울기 구하기
        x[idx] = tmp_val  # 값 복원
        it.iternext()
        return grad

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)   # (784, 100)
        self.params['b1'] = np.zeros(hidden_size)  # (100,)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)  # (100, 10)
        self.params['b2'] = np.zeros(output_size)  # (10,)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):   # predict()의 결과와 정답레이블을 바탕으로 교차엔트로피 오차를 구하도록 구현
        y = self.predict(x)

        return cross_entropy_error(y, t)   # '전체' 데이터를 이용해서 '손실함수' 출력

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블, W : dummy
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)   # 손실함수를 반환한다.

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])   # 이건 미리 만들어놓은 함수를 말한다. 위의 메서드와 이름은 같지만 다른 함수
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])   # x와 t에 의해 만들어진 손실함수에서 W나 b에 대한 미분값을 구한다. (수치미분사용)
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])   # numerical_gradient는 w값을 갱신하기 위해 사용된다.
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
train_loss_list = []

# 하이퍼 파라미터
iters_num = 10000   # 반복횟수
train_size = x_train.shape[0]  # 60000
batch_size = 100    # 미니배치 크기를 100으로
learning_rate = 0.1

network = TwoLayerNet(784, 50, 10)  # 인풋, 히든, 아웃풋

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)   # 60000까지에서 숫자 100개 선
    x_batch = x_train[batch_mask]   # 인덱스로 활용
    t_batch = t_train[batch_mask]   # 인덱스로 활용
    # 60000개 중에서 100개씩 선택해서 그것들의 손실함수를 구하고 그것들의 미분값을 적용해 갱신한다.
    # (데이터가 몇개(행, 건)이 들어오든 w나 b의 형태는 변하지 않는다.) 매개변수의 형태(shape)는 은닉층과 출력층에 의해 변한다.

    # 기울기 계산
    grad = network.numerical_gradient(x_batch, t_batch)  # --> loss --> predict --> cross_entropy_error --> numerical_gradient --> (수치미분해서 각 매개변수의 기울기-미분값 구하고!!!)

    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key] # (위에서 찾아낸 기울기를 사용하여 매개변수 갱신)

    # 학습경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################
#############################################################

## 시험데이터로 평가하기
# 학습을 반복함으로써 손실함수의 값이 서서리 내려가는 것을 확인할 수 있다.
# 정확히는 '훈련데이터의 미니배치에 대한 손실함수'의 값이다.
# 시험데이터로 평가가 이루어질 수 있도록 앞의 구현에서 조금 수정을 한다.
import numpy as np
from dataset.mnist import load_mnist
# from two_layer_net import TwoLayerNet
import sys, os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정

# from common.functions import *
# from common.gradient import numerical_gradient


######### 필요한 메서드 @@@@@@@@@@
def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def softmax(x):
    if x.ndim == 2:
        x = x.T
        x = x - np.max(x, axis=0)
        y = np.exp(x) / np.sum(np.exp(x), axis=0)
        return y.T

    x = x - np.max(x) # 오버플로 대책
    return np.exp(x) / np.sum(np.exp(x))


def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # 훈련 데이터가 원-핫 벡터라면 정답 레이블의 인덱스로 반환
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size


def numerical_gradient(f, x):  # 특정값에서 손실함수의 기울기를 벡터로 출력 @@@@, 독립변수가 여러개, 편미분 여러개 한번에 하기
    # 여기에는 x값에 net.W가 들어가기 때문에 h값을 더해주고 뺄때 net.W값이 갱신된다.
    h = 1e-4  # 0.0001
    grad = np.zeros_like(x)  # x와 형상(shape)이 같은 배열을 생성
    it = np.nditer(x, flags=['multi_index'], op_flags=['readwrite'])
    while not it.finished:  # .size는 배열 안에 있는 원소의 수
        idx = it.multi_index
        tmp_val = x[idx]  # 원래값을 저장해놓을 변수
        x[idx] = float(tmp_val) + h
        fxh1 = f(x)  # y값 구하기, f(x+h)
        x[idx] = tmp_val - h
        fxh2 = f(x)  # y값 구하기, f(x-h)
        grad[idx] = (fxh1 - fxh2) / (2 * h)  # 기울기 구하기
        x[idx] = tmp_val  # 값 복원
        it.iternext()
        return grad


######## 클래스 @@@@@@@@@@
class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

    def predict(self, x):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        return y

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)

        return cross_entropy_error(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        t = np.argmax(t, axis=1)

        accuracy = np.sum(y == t) / float(x.shape[0])
        return accuracy

    # x : 입력 데이터, t : 정답 레이블
    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])

        return grads

    def gradient(self, x, t):
        W1, W2 = self.params['W1'], self.params['W2']
        b1, b2 = self.params['b1'], self.params['b2']
        grads = {}

        batch_num = x.shape[0]

        # forward
        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        y = softmax(a2)

        # backward
        dy = (y - t) / batch_num
        grads['W2'] = np.dot(z1.T, dy)
        grads['b2'] = np.sum(dy, axis=0)

        da1 = np.dot(dy, W2.T)
        dz1 = sigmoid_grad(a1) * da1
        grads['W1'] = np.dot(x.T, dz1)
        grads['b1'] = np.sum(dz1, axis=0)

        return grads


######### 실행절 @@@@@@@@@@
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

# 하이퍼파라미터
iters_num = 10000  # 반복 횟수를 적절히 설정한다.
train_size = x_train.shape[0]
batch_size = 100  # 미니배치 크기
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

# 1에폭당 반복 수
iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    # 미니배치 획득
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch)
    grad = network.gradient(x_batch, t_batch)

    # 매개변수 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    # 학습 경과 기록
    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    # 1에폭당 정확도 계산
    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print("train acc, test acc | " + str(train_acc) + ", " + str(test_acc))

# 오버피팅 없음!!!