# 신경망
# 퍼셉트론에서 신경망으로
# 입력층, 은닉층, 출력층
# 가중치는 각 신호의 영향력 제어
# 편향은 얼마나 쉽게 뉴런이 활성화되는지 결정

# y = h(b + w1*x1 + w2*x2)
# h(x) = 0 (x <= 0)
# h(x) = 1 (x > 1)

# 활성화 함수의 등장
# 입력신호의 총합을 출력신호로 변환하는 함수를 일반적으로 활성화함수라고 한다.
# 활성화함수는 입력신호의 총 합이 활성화를 일으키는지 정하는 역할을 한다.

# y = h(b + w1*x1 + w2*x2) 를 풀어서 써보면
# a = b + w1*x1 + w2*x2
# y = h(a)
# 이렇게 두단계로 생각할 수 있다.

# 활성화 함수
# 퍼셉트론에서는 활성화함수로 계단함수를 이용한다.
# 시그모이드 함수, 계단함수, relu함수
# 시그모이드 함수
# h(x) = 1 / (1+exp(-x))
# e는 자연상수를 뜻한다. 2.7828182818

# 계단함수 구현하기
def step_function(x):
    if x > 0:
        return 1
    else:
        return 0
# 하지만 위의 함수는 매개변수로 실수만 받을 수 있기 때문에
# 매개변수로 배열을 받을 수 있는 함수를 만들어야 한다.

# 계단함수 구현하기
import numpy as np
def step_function(x):
    y = x > 0               # (F, T, T, F, F, T) 이렇세 부울(bool)의 형태로 반환한다.
    return y.astype(np.int) # 부울을 정수의 형태로 바꿔준다.
x = np.array([-1.0, 1.0, 2.0])
x
y = x > 0
y
y = y.astype(np.int)
y
# 이렇게도 구현 가능
def step_function(x):
    return np.array(x>0, dtype=np.int)
a = np.array([1,2,3,4,5,6,0,-1,-2])
step_function(a)

## 계단함수의 그래프
import numpy as np
import matplotlib.pylab as plt
from pprint import pprint
def step_function(x):
    return np.array(x>0, dtype=np.int)
x = np.arange(-5.0, 5.0, 0.1)
y = step_function(x)
pprint(y)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)   # y축의 범위지정
plt.show()

# 시그모이드의 함수 구현하기
def sigmoid(x):
    return 1 / (1+np.exp(-x))
x = np.array([-1.0, 1.0, 2.0])
sigmoid(x)
# array([ 0.26894142,  0.73105858,  0.88079708])

## 시그모이드 함수의 그래프
x = np.arange(-5.0, 5.0, 0.1)
y = sigmoid(x)
plt.plot(x, y)
plt.ylim(-0.1, 1.1)   # y범위 지정
plt.show()

## 시그모이드 함수와 계단 함수 비교
# 계단함수와 시그모이드 함수의 매끄러움 차이
def step_funtion(x):
    return np.array(x>0, dtype=np.int)
def sigmoid(x):
    return 1 / (1+np.exp(-x))
x = np.arange(-5.0, 5.0, 0.1)
y1 = step_function(x)
y2 = sigmoid(x)
plt.plot(x, y1)
plt.plot(x, y2)
plt.ylim(-0.1, 1.1)
plt.show()

# 비선형함수
# 신경망에서는 활성화 함수로 비선형함수를 사용해야 한다. 왜 선형함수는 안되는 것일까?
# 층을 쌓는 혜택을 얻기 위해서.. (나도 잘 모르겠음)

# ReLU 함수
# 최근 ReLU 함수를 많이 이용하는 추세임
# ReLU 함수는 입력이 0을 넘으면 그 입력을 그대로 출력하고 넘지 않으면 0을 출력함
# ReLU 함수 구현하기
def relu(x):
    return np.maximum(0, x)   # 두개중에 큰거 뽑아라~

## ReLU 함수 그래프
x = np.arange(-5,5,0.1)
y = relu(x)
plt.plot(x,y)
plt.ylim(-0.1, 5.1)
plt.show()


## 다차원 배열의 계산
# 넘파이를 활용해 다차원배열 계산하기
import numpy as np
A = np.array([1,2,3,4])
print(A)
np.ndim(A)   # 차원의 수, dimension
A.shape      # 튜플을 반환함
A.shape[0]   # 튜플의 첫번째 인자 출력

# 2차원 배열 작성
B = np.array([[1,2], [3,4], [5,6]])
print(B)
np.ndim(B)
B.shape

# 행렬의 내적
# (3,2) * (2*3) = (3,3)     <==  행렬곱의 안쪽이 같아야 계산가능, 답은 바깥쪽 수로 나옴
A = np.array([[1,2], [3,4]])
A.shape
B = np.array([[5,6], [7,8]])
B.shape
np.dot(A, B)

A = np.array([[1,2,3], [4,5,6]])
A.shape
B = np.array([[1,2], [3,4], [5,6]])
B.shape
np.dot(A, B)

# (3,2) * (2,4) = (3,4)  # 2차원 내적
# (3,2) * (2,) = (3,)    # 1차원 내적
# (4,) * (4,2) = (2,)    # 1차원 내적

import numpy as np
a = np.array([[1,2], [3,4], [5,6]])
a.shape
b = np.array([7,8])
b.shape
np.dot(a,b)
# (3,2) * (2,) = (3,)

# 신경망의 내적
x = np.array([1,2])
x.shape
w = np.array([[1,3,5], [2,4,6]])
print(w)
w.shape
y = np.dot(x, w)
print(y)

# 3층 신경망 구현하기
# 순방향 처리 구현
# 2 -> 3 -> 2 -> 2  (입력층 2개, 1은닉층 3개, 2은닉층 2개, 출력층 2개)

# 각 층의 신호 전달 구현하기
# 편향은 뉴런의 개수만큼 정해져 있다. 즉 결과값 y의 행렬을 (2,3)으로 가정하면 편향의 행렬은 (3,)이다.
# a1 = w11*x1 + w12*x2 + b1
# a2 = w21*x1 + w22*x2 + b1
# a3 = w31*x1 + w32*x2 + b1
# 여기서 행렬의 내적을 이용하면 다음처럼 간소화할 수 있다.
X = np.array([1.0,0.5])
W1 = np.array([[1.0,0.3,0.5], [0.2,0.4,0.6]])
B1 = np.array([0.1,0.2,0.3])
A1 = np.dot(X, W1) + B1
A1

def sigmoid(x):
    return 1 / (1+np.exp(-x))

# 1층
Z1 = sigmoid(A1)
Z1
# 이 sigmoid() 함수는 앞에서 정의한 함수다. 이 함수는 넘파이 배열을 받아 같은 수의 원소로 구성된 넘파이 배열을 반환한다.

# 2층, 방금 출력한 Z1을 입력값으로 받아서 계산을 수행한다.
W2 = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
B2 = np.array([0.1,0.2])
A2 = np.dot(Z1, W2) + B2
Z2 = sigmoid(A2)
Z2

# 출력층, 위에서 출력한 Z2값을 매개변수로 받아서 항등함수 수행, 항등함수!!!!!!!!!!!!
def identity_function(x):
    return x
W3 = np.array([[0.1,0.3], [0.2,0.4]])
B3 = np.array([0.1,0.2])
A3 = np.dot(Z2, W3) + B3
Y = identity_function(A3)

# 구현정리
def init_network():
    network = {}

