import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=False, one_hot_label=True)

print(x_train[0].reshape(1,1,28,28).shape)