import numpy as np

x = np.array([[1,2], [2,4]])
w = np.array([[1,3,5], [2,4,6]])
b = np.array([1,2,3])

y = np.dot(x, w) + b
print(y)

dx = np.dot(y, w.T)
dw = np.dot(x.T, y)
print(dx)
print(dw)