import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
import numpy as np
from ch5.backpropagation_two_layer_net import TwoLayerNet, SoftmaxWithLoss

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True, flatten=True)

nn = TwoLayerNet(784, 50, 10)
print(x_train.shape[0])

iter_num = 1000
batch_size = 100
lr = 0.1

loss_list=[]
acc = []

for i in range(iter_num):
    batch_mask = np.random.choice(x_train.shape[0], batch_size)
    train = x_train[batch_mask]
    label = t_train[batch_mask]

    grad = nn.gradient(train, label)

    for j in ['W1', 'b1', 'W2', 'b2']:
        nn.params[j] -= lr * grad[j]

    loss = nn.loss(train, label)
    loss_list.append(loss)

print(nn.accuracy(x_test, t_test))
plt.plot(loss_list, '.-', ms=3 ,lw=0.5)
plt.show()