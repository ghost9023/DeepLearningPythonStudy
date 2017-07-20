import sys, os
sys.path.append(os.pardir)
import numpy as np
from ch5.layers import *
from book.common.gradient import numerical_gradient
from collections import OrderedDict # 순서가 있는 딕셔너리를 구현함.

class TwoLayerNet:
    def __init__(self, input_size, hidden_size, output_size, weight_init_std = .01):
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(input_size, hidden_size)
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b2'] = np.zeros(output_size)

        self.layers = OrderedDict() # 딕셔너리에 순서가 존재하도록 설정함.
        self.layers['Affine1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['Relu1'] = ReLU()
        self.layers['Affine2'] = Affine(self.params['W2'], self.params['b2'])
        self.lastLayer = SoftmaxWithLoss()
            # Affine1 - Relu1 - Affine2 - SoftmaxWithLoss 순으로 레이어를 구성

    def predict(self, x):
        for layer in self.layers.values():  # 각 레이어의 입력을 x 에 계속 갱신하면서 다음 레이어에 입력
            x = layer.forward(x)

        return x    # 최종적으로는 신경망의 출력이 담기게 된다.

    def loss(self, x, t):
        y = self.predict(x)
        return self.lastLayer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis = 1)
        if t.ndim != 1 : t = np.argmax(t, axis = 1)
            # t 가 one-hot-encoding 된 경우 2차원이므로 최대값을 찾아 인덱스를 반환해 레이블로 된 벡터를 만든다.

        accuracy = np.sum(y == t) / float(x.shape[0])

        return accuracy

    def numerical_gradient(self, x, t):
        loss_W = lambda W : self.loss(x, t)

        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        self.loss(x,t)  # 한번 순전파를 행해서 각 레이어의 값들을 갱신한다.

        # 역전파 시작
        dout = 1
        dout = self.lastLayer.backward(dout)

        layers = list(self.layers.values()) # 레이어들을 역순으로 배열
        layers.reverse()
        for layer in layers:    # dout 을 갱신해가며 이전 레이어로 전파
            dout = layer.backward(dout)

        # 각 레이어가 보관하고있는 미분값을 담는다.
        grads = {}
        grads['W1'] = self.layers['Affine1'].dW
        grads['b1'] = self.layers['Affine1'].db
        grads['W2'] = self.layers['Affine2'].dW
        grads['b2'] = self.layers['Affine2'].db

        return grads

