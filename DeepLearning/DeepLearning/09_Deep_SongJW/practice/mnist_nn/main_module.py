import matplotlib.pyplot as plt

from dataset.mnist import load_mnist
from practice.mnist_nn.network_module import *

(x_train, t_train), (x_test, t_test) = load_mnist(one_hot_label=True, normalize=True)
# nn_structure = (x_train.shape[1], 'ReLU', 50, 'ReLU', 35, 'SoftmaxWithLoss', 10)
nn_structure = (x_train.shape[1], 'ReLU', 50, 'SoftmaxWithLoss', 10)

nn = NeuralNetwork(nn_structure=nn_structure, lr=.1)

iteration = 10000
batch_size = 100
loss_lst = []
test_acc_lst = []
train_acc_lst = []
iter_per_epoch = x_train.shape[0] / batch_size

for i in range(iteration):
    mask = np.random.choice(x_train.shape[0], batch_size)
    x_batch = x_train[mask]
    t_batch = t_train[mask]

    nn.gradient_descent(x_batch, t_batch)
    loss = nn.temp_loss
    loss_lst.append(loss)

    if i % iter_per_epoch == 0:
        print(loss)
        train_acc = nn.accuracy(x_train, t_train)
        test_acc = nn.accuracy(x_test, t_test)
        train_acc_lst.append(train_acc)
        test_acc_lst.append(test_acc)
        print('iter',i,'- train data set acc :',train_acc,', test data set acc :', test_acc)

plt.plot(train_acc_lst, label='train')
plt.plot(test_acc_lst, label='test')
plt.legend()
plt.show()
