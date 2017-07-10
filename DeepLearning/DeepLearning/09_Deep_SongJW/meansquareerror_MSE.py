import numpy as np
def meanSquaredError(y, t):
    return 0.5 * np.sum((y-t)**2)

t=np.array([0,0,0,0,0,1,0,0,0,0])
y=np.array([0.1, 0.03, 0.05, 0.2, 0.03, 0.9, 0.1, 0.2, 0.12, 0.0])

print('정답인 경우', np.sum(y))
print('MSE : ', meanSquaredError(y, t))

y=np.array([0.1, 0.03, 0.05, 0.2, 0.03, 0.0, 0.1, 0.2, 0.12, 0.9])
print('오류인 경우', np.sum(y))
print('MSE : ', meanSquaredError(y, t))