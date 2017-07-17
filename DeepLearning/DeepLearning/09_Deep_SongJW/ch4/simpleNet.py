import sys, os
sys.path.append(os.pardir)
import numpy as np
from book.common.functions import softmax, cross_entropy_error
from book.common.gradient import numerical_gradient

class simpleNet:
    def __init__(self):
        self.W = np.random.randn(2,3)

    def predict(self, x):
        return np.dot(x, self.W)

    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss

if __name__ == '__main__' :
    net = simpleNet()
    print(net.W)

    x = np.array([.6, .9])
    p = net. predict(x) # 입력과 가중치의 내적
    print(p)

    print(np.argmax(p)) # 최대값의 인덱스

    t = np.array([0, 0, 1]) # 정답 레이블
    print(net.loss(x, t))   # 손실함수
        # 최대값의 인덱스가 0, 1 일때 = 정답을 맞추지 못했을때 loss = 1 초과
        # 최대값의 인덱스가 2 일때 = 정답을 맞추었을때 loss = 1 미만

    # def f(W) :
    #     return net.loss(x, t)

    f = lambda W : net.loss(x, t)   # lambda 식을 사용한 간단한 함수의 선언

    dW = numerical_gradient(f, net.W)
    print(dW)
        # 그래디언트
        # [[0.0733941   0.13238714 - 0.20578125]
        #  [0.11009116  0.19858072 - 0.30867187]]
