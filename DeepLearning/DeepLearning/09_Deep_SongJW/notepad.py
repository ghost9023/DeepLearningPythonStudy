import numpy as np

a = np.arange(60).reshape(5, 12).T
print(a)
b = a.reshape(3,2,2,-1).transpose(0,3,1,2)
print(b)
