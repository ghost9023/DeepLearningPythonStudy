import numpy as np
from practice.cnn.layer_module import Pooling

x = np.array([[[1, 2, 3, 0], [0, 1, 2, 4], [1, 0, 4, 2], [3, 2, 0, 1]],
              [[3, 0, 6, 5], [4, 2, 4, 3], [3, 0, 1, 0], [2, 3, 3, 1]],
              [[4, 2, 1, 2], [0, 1, 0, 4], [3, 0, 6, 2], [4, 2, 4, 5]]], ndmin=4)
print(x)
p = Pooling()
y = p.forward(x)
print(y)
z = p.backward(y)
print(z)
# b = p.backward(y)
# print(b)