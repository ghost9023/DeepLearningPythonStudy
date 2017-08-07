## CHAPTER 5 오차역전파법
# 앞에서는 신경망의 가중치 매개변수의 기울기는 수치미분을 사용해 구했다.
# 수치미분은 단순하고 구현하기 쉽지만 계산시간이 오래 걸린다는 단점이 있다.
# 이번장은 가중치 매개변수의 기울기를 효율적으로 계산하는 '오차역전파법'을 배워보자

# 오차역전파법을 제대로 이해하는 방법은 두 가지가 있다.
# 1. 수식을 통해 - 정확하고 간결함
# 2. 계산 그래프를 통해 - 시각적으로!! 본질에 다가가기 쉬움

# 계산그래프
# 노드와 엣지로 표현된다.
# 계산그래프로 풀다.
# 1. 계산그래프를 구성한다.
# 2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다.
# 계산을 왼쪽에서 오른쪽으로 진행하는 것을 순전파라고 한다. 반대로는 역전파

# 계산그래프의 특징은 '국소적 계산'을 전파함으로써 최종 결과를 얻는다는 점에 있다.
# 계산 그래프는 국소적 계산에 집중한다.

# 왜 계산 그래프로 푸는가?
# 계산 그래프를 사용하는 가장 큰 이유는 역전파를 통해 '미분'을 효율적으로 계산할 수 있다는 점때문이다.
# 가령 사과의 가격이 오르면(매개변수) 최종금액(손실함수)에 어떻게 영향을 끼치는지를 알고 싶다고 한다면 이는 사과가격에 대한 지불금액의 미분을 구하는 문제에 해당한다.
# 이 결과로부터 사과가격에 대한 지불금액의 미분값은 2.2라고 할 수 있다. 사과가 1원 오르면 최종금액은 2.2원 오른다는 뜻이다.
# 정확히는 사과값이 아주 조금 오르면 최종금액은 그 아주 작은 값의 2.2배만큼 오른다는 뜻(미분의 개념, 순간기울기)
# 소비세에 대한 지불금액의 미분이나 사과개수에 대한 지불금액의 미분도 같은 순서로 구할수 있다.
# 그리고 그때는 중간까지 구한 미분결과를 공유할 수 있어서 다수의 미분을 효율적으로 계산할 수 있다.

# 연쇄법칙
# 역전파는 '국소적인 미분'을 순방향과는 반대인 오른쪽에서 왼쪽으로 전달한다. 또한 이런 국소적 미분을 전달하는 원리는 연쇄법칙에 따른 것이다.
# 이번 절에서는 연쇄법칙을 설명하고 그것이 계산그래프상의 역전파와 같다는 사실을 밝히겠다.
# 계산그래프의 역전파
# 역전파의 계산절차는 신호 E와 노드의 국소적미분을 곱한 후 다음 노드로 전달하는 것이다.
# 국소적 미분은 순전파 때의 y = f(x) 계산의 미분을 구한다는 뜻이다.
# 가령 y = f(x) = x^2이라면 ∂y/∂x=2x가 된다. 그리고 이 국소적인 미분을 상류에서 전달된 값에 곱해 앞쪽노드로 전달하는 것이다.
# 연쇄법칙이란?
# 합성함수@@@란 여러함수로 구성된 함수이다. 예를 들어서 (z=t^2 / t = x+y)와 같은 함수를 말한다.
# 합성함수의 미분은 합성함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.
# ∂z/∂x(x에 대한 z의 미분)은 ∂z/∂t(t에 대한 z의 미분)과 ∂t/∂x(x에 대한 t의 미분)의 곱으로 나타낼 수 있다.
# ∂z/∂x = (∂z/∂t) * (∂t/∂x)
# ∂z/∂t는 2t이고, ∂t/∂x는 1이다.
# ∂z/∂x = (∂z/∂t)*(∂t/∂x) = 2t*1 = 2*(x+y)
# 역전파의 계산절차에서는 노드로 들어온 입력신호에 그 노드의 국소적 미분(편미분)을 곱한 후 다음 노드로 전달한다.
# 역전파 때는 상류에서 전해진 미분(이 예에서는 ∂z/∂x)에 1을 곱하여 하류로 흘립니다.
# 이 예에서는 상류에서 전해진 미분값을 ∂L/∂z 이라 했는데, 같이 최종적으로 L이라는 값을 출력하는 큰 계산 그래프 가정하기 때문이다.

# 덧셈노드의 역전파
# ∂z/∂x = 1  /  ∂z/∂y = 1
# 변수 z와 나머지 변수를 대상으로 2차원 그래프를 그리면 기울기가 1이고 y절편이 나머지 하나의 변수의 수에 결정되는 1차함수가 그려진다.
# 따라서, x값에 상관없이 모든 x값에 대한 미분값은 1이 된다.
# 덧셈노드 역전파는 입력신호를 받아서 1(미분값)을 곱한다음 다음 노드로 출력할뿐이므로 그냥 그대로 출력하면 된다.

# 곱셈노드의 역전파
# z = x*y를 생각해보자!!!
# 곱셈노드의 역전팓는 상류의 값에 순전파 때의 입력신호들을 '서로 바꾼값'을 곱해서 하류로 보낸다.
# 덧셈노드와 마찬가지로 변수z와 나머지 하나의 변수를 대상으로 2차원 그래프를 그려보면 y의 값에 따라 기울기가 정해지는 1차 함수가 만들어진다.
# 즉 곱셈노드에서는 x의 값에 관계없이 언제나 기울기(미분)가 y의 값으로 정해지게 된다.
# 곱셈노드의 역전파에서는 입력신호를 서로 바꿔서 하류로 흘린다.
# 결과를 보면 사과 가격의 미분은 2.2, 사과개수의 미분은 110 이 된다.

# 곱셈계층
# 이제부터 모든 계층을 forward()-순전파 와 baxkward()-역전파라고 한다.
import numpy as np
class MulLayer:   # 곱셈노드@!@!#!@#!@#!@#!@#!@#!@#
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y    # 순전파는 입력 두개 받아서 곱해서 전해주고
        return out
    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x  # x와 y를 바꾼다.
        return dx, dy # 역전파는 서로의 값을 바꿔서 들어온 입력값을 곱해서 전해준다. (입력 1개, 출력 2개)
#  MulLayer를 사용해서 순전파를 다음과 같이 구현할 수 있다.
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)   # 220.00000000000003

# 역전파 구현
dprice = 1  # 처음에 들어오는 손실함수의 값이라고 볼 수있다.
dapple_price, dtax = mul_tax_layer.backward(dprice)          # self.x에 apple_num이 할당  /  self.y에 tax가 할당
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # self.x에 apple이 할당  /  self.y에 apple_num가 할당
print(dapple, dapple_num, dtax)   # 2.2 110.00000000000001 200


## 덧셈 계층
class AddLayer:   # 덧셈노드!#!@#!@#!@#!@#!@#!@#!@#!@#
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
# 인스턴스 변수를 선언하지 않기 때문에 __init__(): 에서 pass를 해준다.

# p.163의 그래프를 파이썬으로 구현한 함수
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)   # 715.0000000000001

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(apple, apple_num, orange, orange_num, dtax)   # 100 2 150 3 650


## 활성화 함수 계층 구현하기
# 우선 활성화함수인 relu와 sigmoid계층을 구현한다.

# relu 계층
# 활성화함수로 사용되는 relu의 수식
# y = x (x > 0)
# y = 0 (x <= 0)
# ∂y/∂x = 1 (x > 0), x가 0보다 클때 x의 값에 관계없이 기울기는 항상 1이다.
# ∂y/∂x = 0 (x <= 0), x가 0보다 작으면 x의 값에 관계없이 기울기는 항상 0(가로 직선)이다.
# 순전파 때 입력인 x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘리고,
# 순전파 때 입력인 x가 0보다 작으면 역전파는 하류로 신호를 보내지 않는다.
# relu계층 파이썬 구현
# 신경망 계층의 relu계층은 넘파이배열을 인수로 받는다고 가정한다!!!! 넘파이배열!@@!@#!@!@#
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)   # booleah의 배열형태로 나온다. 여기서 주의할 점은 0보다 작은 원소들을 True로 빼준다는 점이다.
        out = x.copy()         # 원본 데이터의 변경을 막기 위해 다른 객체를 만든다.
        out[self.mask] = 0     # 0보다 작아서 True로 나온 원소들을 0으로 바꾼다.
        return out
    def backward(self, dout):  # 역전파는 결국 미분을 한다는 뜻
        dout[self.mask] = 0    # dout배열에서 self.mask에 의해 True가 된 애들은 0으로 만들어주고 아닌(False) 원소는 그대로 둔다.
        dx = dout
        return dx
        # 역전파일 때 입력값이 어떻게 들어오는지 모르겠지만 1이 아닌 경우 그냥 그대로 출력하는데 양수면 무조건 1로 출력해야하는거 아닌가요?
        # 이해가 안감

x = np.array([[1.0,-0.5], [-2.0, 3.0]])
print(x)
mask = (x <= 0)
print(mask)

# sigmoid 계층
# y = 1 / (1+exp(-x))
# 1단계 : y = x^-1을 미분하면 ∂y/∂x = -x^-2(순전파의 입력을 -2승하고 마이너스를 붙인 값)  =>  -y^2(순전파의 출력을 제곱한 후 마이너스를 붙인 값)
# 2단계 : +노드는 강류의 값을 여과럾이 하류로 내보내는 값
# 3단계 : 자연상수(e)는 미분하면 자연상수(e)가 나온다.
# 4단계 : *노드는 순전파때의 값을 서로 바꿔서 -1을 곱한다.
# 최종적으로 E = E*((1/x^2) * exp(-x))  =>  E * (y^2*exp(-x)) p.169  =>  E*(y*(1-y)) - 이렇게 유도 가능
# 여기서 핵심은 결국에 sigmoid함수의 역전파 식은 E*(y*(1-y))@@!@!!@!@!@!!!@#!@#!@#!
import numpy as np
class Sigmoid:
    def __init__(self):
        self.out = None
    def forward(self, x):
        out = 1 / (1+np.exp(-x))
        self.out = out
    def backward(self, dout):
        dx = dout = (1.0 - self.out) * self.out
        return dx

## Affine/Softmax 계층 구현하기
# affine계층
x = np.random.rand(2)   # 0~1사이 아무 숫자, 정상분포 아님
x   # array([ 0.77238513,  0.66408782])

x = np.random.rand(2)
w = np.random.rand(2, 3)
b = np.random.rand(3)
x.shape
w.shape
b.shape
# 그러면 뉴런의 가중치 합은 y = np.array(x, w)+b 로 계산한다.
# 행렬의 내적계산은 대응하는 차원의 원소수를 일치시키는 것이 핵심이다.
# (2, ) * (2, 3) = (3, )
# ∂l/∂X = (∂l/∂Y) * W.T
# ∂l/∂W = X.T * (∂l/∂Y)
# 이걸 유도하는 식은 생략하는데 다시 한번 잘 생각해보기
# (2, 3)의 전치행렬은 (3, 2)
# 위의 식이 affine계층의 역전파이다.(np.dot()의 역전파를 구하는 계층)
# X와 (∂l/∂X)의 형상은 같다. W와 (∂l/∂W)의 형상도 같다.
# 학원가서 다시한번 생각해보기 p.173

# 배치용 affine계층
# N개의 데이터의 입력값을 인수로 받는 affine계층
X_dot_W = np.array([[0,0,0], [10,10,10]])
B = np.array([1,2,3])
X_dot_W
X_dot_W + B

dY = np.array([[1,2,3], [4,5,6]])
dY
dB = np.sum(dY, axis=0)
dB

class Affine:
    def __init__(self):
        self.W = W
        self.b = b
        self.x = None
        self.dW = None
        self.db = None
    def forward(self, x):
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)   # 전치
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)   # 바깥 차원의 대응하는 원소끼리 연산
        return dx
# 그러니까 affine은 가중치와 입력값, 편향을 행렬에 맞게 변환해서 계산해주는 함수이다. (순전파, 역전파)

# Softmax_with_Loss 계층
# 소프트맥스함수는 입력값을 정규화하여 출력한다.
# mnist의 출력값은 분류는 10개이기 때문에 softmax계층의 입력은 10개가 된다.
# 여기서는 소프트맥스와 교차엔트로피(손실함수)를 같이 포함하여 계층을 구현한다.
# Softmax_with_Loss()계층의 주목할 만한 점은 softmax계층의 역전파가 (y-t)라는 말끔한 결과를 내놓고 있다는 것이다.
# (y1, y2, y3)sms softmax계층의 출력이고 (t1, t2, t3)는 정답레이블을 뜻한다.
# 위의 이와 같은 성질은 신경망학습의 중요한 성질이다.
# 사실 이런 말끔한 미분값은 우연이 아니라 교차 엔트로피 오차라는 함수가 그렇게 되도록 설계되었기 때문이다.
# 항등함수의 손실함수로 '평균제곱오차'를 사용하는 것 역시 역전파가 (y-t)로 말끔하게 떨어진다.
# 예를 들어 정답레이블 (0, 1, 0)이고 소프트맥스의 출력계층이 (0.3, 0.2, 0.5)를 출력했다고 한다면 이때의 softmax의 역전파는 (0.3, -0.8, 0.5)라는 값을 출력한다.
# 파이썬으로 softmax-with-loss 계층구현

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)   # 소프트맥스 지나서~ (확률의 형태)
        self.loss = cross_entropy_error(self.y, self.t)   # 손실함수 구하고~ (손실함수 숫자 하나)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]   # 몇개씩 일괄계산하나~
        dx = (self.y - self.t) / batch_size
        return dx

# 오차역전파법 구현하기
# 신경망학습의 전체그림
# 전체 : 신경망에는 적응가능한 가중치와 편향이 있고, 이 가중치의 편향을 훈련데이터에 적응하도록 조정하는 과정을 '학습'이라고 한다. 신경망 학습은 다음과 같이 4단계로 수행한다.
# 1단계 : 훈련데이터 중 일부를 무작위로 가져온다. 이렇게 선별한 데이터를 미니배치라 하며, 그 미니배치의 손실함수값을 줄이는 것이 목표이다.
# 2단계 : 미니배치의 손싱함수값을 줄이기 위해 각 가중치의 매개변수의 기울기를 구한다. 기울기는 손실함수의 값을 가장 작게 하는 방향을 제시한다.
# 3단계 : 가중치 매개변수를 기울기 방향으로 아주 조금 갱신한다.
# 4단계 : 1~3단계를 반복한다.
# 선명한 오차역전파법이 등장하는 단계는 두번째인 기울기 산출이다. 앞장에서는 이 기울기를 구하기 위해서 수치미분을 사용했다. 그러나 이 수치미분은 구현하기는 쉽지만 계산이 매우우매우매우 오래걸린다.
# 오차역전파법은 빠르다.


###################################################
###################################################
###################################################
###################################################
###################################################
###################################################
# coding: utf-8
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

class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0
        return out
    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout
        return dx

class Affine:
    def __init__(self, W, b):
        self.W = W
        self.b = b
        self.x = None
        self.original_x_shape = None
        # 가중치와 편향 매개변수의 미분
        self.dW = None
        self.db = None
    def forward(self, x):
        # 텐서 대응
        self.original_x_shape = x.shape
        x = x.reshape(x.shape[0], -1)
        self.x = x
        out = np.dot(self.x, self.W) + self.b
        return out
    def backward(self, dout):
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis=0)
        dx = dx.reshape(*self.original_x_shape)  # 입력 데이터 모양 변경(텐서 대응)
        return dx

class SoftmaxWithLoss:
    def __init__(self):
        self.loss = None # 손실함수
        self.y = None    # softmax의 출력
        self.t = None    # 정답 레이블(원-핫 인코딩 형태)
    def forward(self, x, t):
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss
    def backward(self, dout=1):
        batch_size = self.t.shape[0]
        if self.t.size == self.y.size: # 정답 레이블이 원-핫 인코딩 형태일 때
            dx = (self.y - self.t) / batch_size
        else:
            dx = self.y.copy()
            dx[np.arange(batch_size), self.t] -= 1
            dx = dx / batch_size
        return dx




import sys, os
sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정
import numpy as np
# from common.layers import *
# from common.gradient import numerical_gradient
from collections import OrderedDict


class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std=0.01):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])

        self.lastLayer = SoftmaxWithLoss()   # 클래스 객체화

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)   # 여기서 포워드로 쫙 실행

        return x

    # x : 입력 데이터, t : 정답 레이블
    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1: t = np.argmax(t, axis=1)

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
        # forward
        self.loss(x, t)

        # backward
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)

        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        # 여기서 미분값 저장함
        return grads
###############################@@@@@@@@@@@@@@
###############################@@@@@@@@@@@@@@
###############################@@@@@@@@@@@@@@
###############################@@@@@@@@@@@@@@

# 오차역전파법을 사용한 학습 구현하기
# coding: utf-8
# 실행절
import sys, os

sys.path.append(os.pardir)

import numpy as np
from dataset.mnist import load_mnist
# from two_layer_net import TwoLayerNet

# 데이터 읽기
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)

iters_num = 10000
train_size = x_train.shape[0]
batch_size = 100
learning_rate = 0.1

train_loss_list = []
train_acc_list = []
test_acc_list = []

iter_per_epoch = max(train_size / batch_size, 1)

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    # 기울기 계산
    # grad = network.numerical_gradient(x_batch, t_batch) # 수치 미분 방식
    grad = network.gradient(x_batch, t_batch)  # 오차역전파법 방식(훨씬 빠르다)

    # 갱신
    for key in ('W1', 'b1', 'W2', 'b2'):
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)

    if i % iter_per_epoch == 0:
        train_acc = network.accuracy(x_train, t_train)
        test_acc = network.accuracy(x_test, t_test)
        train_acc_list.append(train_acc)
        test_acc_list.append(test_acc)
        print(train_acc, test_acc)


            # 정리!!!
# relu계층, softmaxwithloss계층, affine계층, softmax계층
# 모든 계층에서 forward와 backward계층을 구현했다.