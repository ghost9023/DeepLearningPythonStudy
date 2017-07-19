import numpy as np
from dataset.mnist import load_mnist
from ch5.backpropagation_two_layer_net import TwoLayerNet

'''
수치해석적으로 기울기를 구하는 방법은 구현이 매우 간단.
수치해석적으로 기울기를 구하는 방식에 비해 기울기를 역전파로 구하는 방식은 구현이 어려우나
속도가 빠름.
수치해석적으로 기울기를 구하는 방법은 구현이 간단하므로 실수를 할 가능성이 적다.
구현이 복잡해 실수할 가능성이 큰 역전파방식이 정확하게 구현되었는지를 판단하기위해서 
수치해석적으로 구한 기울기와 역전파방식으로 구한 기울기를 비교해
역전파 방식의 신뢰성을 확인하는 작업을 기울기 확인 gradient check 이라고 한다.
'''
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

network = TwoLayerNet(input_size = 784, hidden_size = 50, output_size = 10)

x_batch = x_train[:3]
t_batch = t_train[:3]

grad_numerical = network.numerical_gradient(x_batch, t_batch)
grad_backprop = network.gradient(x_batch, t_batch)

for key in grad_numerical.keys():
    diff = np.average(np.abs(grad_backprop[key] - grad_numerical[key]))
    print(key + ':' + str(diff))

'''
W1:2.59328420969e-13
b1:9.24720992054e-13
W2:8.9643377194e-13
b2:1.2012613404e-10
두 방식으로 구한 기울기의 차이가 매우 작으므로
역전파 방식도 제대로 구현되었다고 할 수 있다.
'''