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
    if y.dim == 1:
        t = t.reshape(1, t.size)   # 2차원으로 바꿔주기
        y = y.reshape(1, y.size)   # 2차원으로 바꿔주기
    batch_size = y.shape[0]        # 들어온 행의 수대로 배치사이즈 할당
    return -np.sum(t * np.log(y)) / batch_size   # 다 더해서 배치사이즈로 나눠주기, 즉 평균 구하는 것
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
        grad = numerical_gradient(f, x)   # 기울기 벡터 반환
        x -= lr * grad
        # 기울기(미분)값에 learning_rate를 곱해서 입력값(x)에서 빼주는 과정을 반복한다.
        # 기울기는 경사감소의 방향과 정도를 결정한다. 기울기가 작아지면 이동하는 거리도 그만큼 줄어든다.
    return x

# 경사법으로 f(x0, x1) = x0^2 + x1^2의 최소값을 구하여라
def function_2(x):
    return x[0]**2 + x[1]**2
init_x = np.array([-3.0, 4.0])
