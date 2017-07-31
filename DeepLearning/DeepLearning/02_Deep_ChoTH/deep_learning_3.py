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


######## 3층 신경망 ##################################################################################
######## 구현정리 ####################################################################################
import numpy as np

def init_network():
    network = {}
    network['W1'] = np.array([[0.1,0.3,0.5], [0.2,0.4,0.6]])
    network['b1'] = np.array([0.1,0.2,0.3])
    network['W2'] = np.array([[0.1,0.4], [0.2,0.5], [0.3,0.6]])
    network['b2'] = np.array([0.1,0.2])
    network['W3'] = np.array([[0.1,0.3], [0.2,0.4]])
    network['b3'] = np.array([0.1,0.2])
    print('network', network)
    return network

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def identity_function(x):
    return x

def forward(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)            # 시그모이드
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)            # 시그모이드
    a3 = np.dot(z2, W3) + b3
    y = identity_function(a3)   # 항등함수
    return y

network = init_network()
x = np.array([1.0, 0.5])
y = forward(network, x)
print(y)   # [ 0.31682708  0.69627909]


# 출력층 설계하기
# 일반적으로 회귀에는 항등함수를, 분류에는 소프트맥스함수를 사용한다.

# 항등함수와 소프트맥스함수 구현하기
# 항등함수는 입력을 그대로 출력
# 분류에서 사용하는 소프트맥스함수의 식
y = np.exp(a) / np.sum(np.exp(a))

# 소프트맥스 함수의 구현해보기
a = np.array([0.3,2.9,4.0])
exp_a = np.exp(a)
print(exp_a)
sum_exp_a = np.sum(exp_a)
y = exp_a / sum_exp_a
print(y)
# 확률로 나타낼 수 있다. 아래는 함수로 구현한 것
def softmax(a):
    exp_a = np.exp(a)
    sum_exp_a = np.sum(exp_a)
    y= exp_a / sum_exp_a
    return y

# 소프트맥스 함수 구현시 주의사항
# 입력값이 작다면 상관없지만 커지면 문제가 발생한다. 너무 큰 값끼리 연산을 하면 값이 불안정해진다.
# 아래는 개선한 함수식
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)   # 제일 큰놈 빼주기, 오버플로 빼주기
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

# 소프트맥스 함수의 특징
# softmax() 함수를 사용하면 신경망의 출력은 다음과 같이 계산할 수 있다.
a = np.array([0.3,2.9,4.0])
y = softmax(a)
print(y)   # 값을 극대화시킨다.
np.sum(y)  # 1이 나온다. --> 확률적인 결론을 낼 수 있도록 한다.

# 출력층의 뉴런 수 정하기
# 일반적으로 분류하는 범주의 수에 맞춰 출력층을 정해준다.
# mnist같은 경우 0~9까지 정하는 것이기 때문에 10개가 된다.

# 손글씨 숫자 인식
# 일단 이미 학습된 매개변수를 사용하여 학습과정은 생략하고, 추론과정만 구현한다.
# mnist 가져오기
import sys, os
sys.path.append(os.pardir)   # 부모 디렉토리의 파일을 가져올 수 있도록 설정
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# flatten은 1차원 배열로 축소하는 옵션
# normalize는 픽셀의 0~255 사이의 값을 0~1 사이의 값으로 정규화하는 옵션
# one_hot_label은 [0,1,0,0,0,0,0,0,0,0]의 형태로 저장

print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)

# 이미지 불러오기
# PIL 모듈 사용 (python image library)
import sys, os
sys.path.append(os.pardir)   # 부모 디렉토리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image

def img_show(img):   # 이미지 그려주는 함수생성
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)
img = img.reshape(28,28)   # 원래 이미지의 모양으로 변형
img_show(img)   # 함수실행


# 신경망의 추론처리
# mnist 데이터셋은 입력층 뉴런을 784(28*28)개, 출력층 뉴런을 10(0~9)개로 구성한다.
# 은닉층은 두개, 첫번째 은닉층은 50개의 뉴런, 두번째 은닉층은 100개의 뉴런을 배치
import sys, os
sys.path.append(os.pardir)   # 부모 디렉토리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

def get_data():   # 테스트파일 feature, label 반환하는 함수
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=False)
    return x_test, t_test

def init_network():   # 학습완료된 가중치 가져오기
    with open("C:\\data\\sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
    return network

def sigmoid(x):
    return 1 / (1+np.exp(-x))

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c)   # 제일 큰놈 빼주기, 오버플로 빼주기
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y

def predict(network, x):   # 순전파
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']

    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)            # 시그모이드
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)            # 시그모이드
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)   # 항등함수
    return y

x, t = get_data()   # (x_test, t_test)
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])   # 학습된 가중치 매개변수와 테스트할 데이터
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# Accuracy:0.9207

# 배치처리
x, _ = get_data()   # x만 받고 t는 필요없다!
network = init_network()   # 가중치 매개변수
W1, W2, W3 = network['W1'], network['W2'], network['W3']
x.shape
x[0].shape
W1.shape
W2.shape
W3.shape
# 이 결과에서 다차원 배열의 대응하는 차원의 원소 수가 일치함을 알 수 있다.
# (784,) * (784,50) * (50,100) * (100,10) -> (10,)
# 원소 784개로 구성된 1차원 배열이 입력되어 마지막에는 원소가 10개인 1차원 배열이 출력되는 형태
# 그렇다면 데이터 한건이 아닌 여러개의 데이터를 한번에 입력하는 경우를 생각해보자
# 가령 이미지 100개를 묶어서 한번에 predict() 함수에 넘기기
# (100,784) * (784,50) * (50,100) * (100,10) -> (100,10), 맨처음 100과 제일 마지막 10
# 이처럼 하나로 묶은 입력데이터를 batch(배치)라고 한다.
# 배치처리는 이미지 1장당 처리시간을 대폭 줄여준다. (원리와 이유는 p.103)
# 이제 앞의 구현을 이용해서 배치처리를 적용해보자!
x, t = get_data()   # (x_test, t_test)
network = init_network()
accuracy_cnt = 0
batch_size = 100

for i in range(0, len(x), batch_size):
    x_batch = x[i:i+batch_size]          # 처음에 0부터 99  (100,784)
    y_batch = predict(network, x_batch)  # 배열 입력 가능, 학습된 가중치, x값 이용해서 순전파
    p = np.argmax(y_batch, axis=1)   # 100개의 각 행에서 가장 큰 수의 인덱스를 반환 (실제로 값과 인덱스가 같다.)
    accuracy_cnt += np.sum(p == t[i:i+batch_size])
    # 이렇게 하면 리스트의 각 원소를 비교하며 bool을 반환한다.
    # [True  True  True  True  True  True  True  True False  True  True  True......]
print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
# Accuracy:0.9207
