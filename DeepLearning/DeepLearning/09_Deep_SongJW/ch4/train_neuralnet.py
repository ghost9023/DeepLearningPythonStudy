import matplotlib.pyplot as plt
import numpy as np
from dataset.mnist import load_mnist
from ch4.two_layer_net import TwoLayerNet

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

train_loss_list = []    # loss 를 기록할 리스트

# hyper parameters
iters_num = 1000   # 미니배치 학습 반복 횟수
train_size = x_train.shape[0]   # 훈련데이터의 수
batch_size = 100    # 미니배치 크기. 한번 학습시 몇건의 데이터를 학습시킬것인지.
learning_rate = 0.1 # 학습률

network = TwoLayerNet(input_size=784, hidden_size=50, output_size=10)
    # 28 x 28 이미지 : 784 길이 벡터, 은닉층 노드 50개, 분류 클래스 0~9 총 10개

for i in range(iters_num):
    batch_mask = np.random.choice(train_size, batch_size)
        # 무작위로 0 ~ (train_size-1) 범위에서 batch_size 개의 수를 뽑음
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    grad = network.numerical_gradient(x_batch, t_batch)

    for key in ['W1', 'b1', 'W2', 'b2']:
        network.params[key] -= learning_rate * grad[key]

    loss = network.loss(x_batch, t_batch)
    train_loss_list.append(loss)
    print(loss)

plt.plot(train_loss_list)
plt.show()