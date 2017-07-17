'''
4.4.1 경사법(경사 하강법) gradient method

손실함수의 값이 최솟값이 되는 가중치와 편향을 찾는것이 목표. 기울기를 이용해 손실함수의 최솟값을 찾는 방법이 경사법.
다만 기울기가 가리키는 방향이 항상 손실함수의 최솟값이 존재하는 방향은 아니다.

최솟값에 도달할수도, 극솟값(특정 범위내에서의 최솟값)에 도달할수도, 안장점에 도달할수도, 
최악으로는 고원 : 플래토 plateau 에 도달하여 학습이 정체기에 빠질수도 있다.

기울기의 방향이 항상 의도한대로 손실함수의 최솟값을 찾는것은 아니지만 일단은 기울기의 방향으로 가는것이 손실함수의 값을 
줄이는 방법은 맞다.

경사법 : 경사법은 현 위치에서 기울어진 방향으로 일정 거리만큼 이동한다.
기울기가 음수이면 기울기 방향으로 나아가면 손실함수가 작아지는 방향으로 나아가는 것이고
기울기가 양수이면 기울기 반대방향으로 나아가면 손실함수가 작아지는 방향으로 나아간다.
나아간 위치에서 다시 기울기를 구하여 손실함수가 작아지는 방향으로 나아가는 것이 경사법이다.

x0 = x0 - eta * (df/dx0)
x1 = x1 - eta * (df/dx1)
...
xn = xn - eta * (df/dxn)

그래디언트가 음수이면 - eta 가 곱해져 양수가 되고, 그래디언트가 양수이면 - eta 가 곱해져 음수가 되어
x 값은 손실함수가 작아지는 방향으로 나아가게된다.

에타 eta 를 학습률 learning rate 이라고 한다. 한번의 학습으로 가중치, 편향의 값을 얼마만큼 갱신할지를 결정하게된다. 
'''

import numpy as np
from ch4.numerical_differentiation import numerical_gradient

def gradient_descent(f, init_x, lr = 0.01, step_num = 100):
    x = init_x.copy()
    for i in range(step_num):
        grad = numerical_gradient(f, x)
        x -= grad * lr
    return x

def func_2(x) :
    return np.sum(x**2)

init_x = np.array([-3., 4.])
print(gradient_descent(func_2, init_x = init_x, lr = 0.1, step_num = 100))  # [ -6.11110793e-10   8.14814391e-10] == [ 0. 0.]
# 너무 큰 학습률
print(gradient_descent(func_2, init_x = init_x, lr = 10.0, step_num = 100)) # [ -2.58983747e+13  -1.29524862e+12]
# 너무 작은 학습률
print(gradient_descent(func_2, init_x = init_x, lr = 1e-10, step_num = 100))    # [-2.99999994  3.99999992]

## 학습률이 너무 크면 발산해버리고, 학습률이 너무 작으면 갱신이 제대로 되지 않는다.