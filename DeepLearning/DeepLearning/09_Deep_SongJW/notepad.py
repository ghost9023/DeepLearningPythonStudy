import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from book.common.gradient import numerical_gradient
from collections import OrderedDict
from book.common.layers import *
import sys
import os

sys.path.append(os.pardir)  # 부모 디렉터리의 파일을 가져올 수 있도록 설정


class TwoLayerNet:
    def __init__(self):
        # 가중치 초기화
        self.params = {}
        self.params['W1'] = np.array([[1, 2, 3], [4, 5, 6]])  # (2,3)
        self.params['b1'] = np.array([1, 2, 3], ndmin=2)  # (2, )
        self.params['W2'] = np.array([[1, 2, 3], [4, 5, 6], [7, 8, 9]])  # (3,3)
        self.params['b2'] = np.array([1, 2, 3], ndmin=2)  # (2, )

        # 계층 생성
        self.layers = OrderedDict()
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()

    def predict(self, x):
        for layer in self.layers.values():
            x = layer.forward(x)
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
            print(layer.__class__.__name__, 'dx :', dout)
        # 결과 저장
        grads = {}
        grads['W1'], grads['b1'] = self.layers['Affine1'].dW, self.layers['Affine1'].db
        grads['W2'], grads['b2'] = self.layers['Affine2'].dW, self.layers['Affine2'].db
        return grads

nn = TwoLayerNet()
x = np.array([[1,2], [3,4], [5,6]])
t = np.array([[3,4,5], [2,1,4], [2,5,6]])
print(nn.predict(x))
print(nn.gradient(x, t))