import numpy as np
from practice.cnn.function_module import img2col

x = np.arange(48*2).reshape(2,3,4,4)
print(x)

x1 = x.reshape(1, 1, -1, 4)
print(x1)

x2 = img2col(x1, 2, 2, 2)
print(x2)

max_ind = np.argmax(x2, axis=1)
x3 = x2[[i for i in range(x2.shape[0])], max_ind]
print(x3.reshape(2, 3, 2, 2))

# x2 = img2col(x, 2, 2, 2).reshape(-1, 4)
# print(x2)
#
# max_arg = np.argmax(x2, axis=1)
# x3 = x2[[i for i in range(x2.shape[0])], max_arg]
# print(x3)
#
# x4 = x3.reshape(2, -1, 2, 3).transpose(0,3,1,2)
# print(x4)