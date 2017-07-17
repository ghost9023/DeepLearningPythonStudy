import numpy as np

# print(np.float32(1e-50))    # 반올림 오차. 0 이 출력된다. 매우 작은 값은 소숫점 8자리 이하정도에서 값이 생략된다.
'''
수치미분
아주 작은 차분(임의의 두 점의 함수값들의 차)으로 미분하는 것을 수치미분이라고 한다.
'''
def numerical_diff(f, x):
    '''
    함수 f 의 x 에서의 기울기를 구한다.
    :param f: 기울기를 구하고자 하는 함수 
    :param x: 기울기를 구하고자 하는 위치
    :return: 기울기값
    '''
    h = 1e-4    # 0.0001 이 적당히 좋은 값으로 알려져있다.
    return (f(x+h) - f(x-h)) / (2 * h)
        # 위 식은 중앙 차분 호은 중심 차분이라고 한다. (차분은 두 점에서의 함수값의 차를 말한다.)
        # (f(x+h) - f(x)) / h 는 전방차분(x+h 로 x 에서 h 만큼 전진한 점과 x 점을 잇는 기울기를 구하므로 전방차분)
        # 전방차분은 엄밀히 말하면 x 에서의 기울기가 아님. h 가 0이 될수 없기에 생기는 오차인데
        # 이 오차를 줄이기위해서 중앙차분을 사용하는 것.

# ex) y = 0.01x^2 + 0.1x

def func_1(x):
    return 0.01 * x**2 + 0.1 * x

# import matplotlib.pyplot as plt
# x = np.arange(0, 20.0, .1)
# y = func_1(x)
# plt.xlabel('x')
# plt.ylabel('f(x)')
# plt.plot(x,y)
# plt.show()

# print(numerical_diff(func_1, 5))    # 5 에서의 기울기 약 2
# print(numerical_diff(func_1, 10))   # 10에서의 기울기 약 3

'''
편미분
독립변수가 2개 이상인 함수를 하나의 독립변수를 제외한 나머지 독립변수들을 상수로 취급하여 미분하는 것.
모든 독립변수의 편미분을 벡터로 정리한것을 기울기(그래디언트 gradient) 라고 한다.
'''

def numerical_gradient(f, x):
    h = 1e-4
    grad = np.zeros_like(x) # x 와 형상이 같은 0 배열을 생성

    for idx in range(x.size) :
        temp = x[idx]

        # x의 idx 번째 값만 변화시켜 수치미분함
        x[idx] = temp + h
        fh1 = f(x)
        x[idx] = temp - h
        fh2 = f(x)

        grad[idx] = (fh1 - fh2) / (2 * h)
        x[idx] = temp

    return grad

# ex)  y = x0**2 + x1**2

def func_2(x):
    return np.sum(x**2)

# x = np.array([3, 4], dtype=np.float)    # 반드시 [3., 4.] 또는 [3.0, 4.0] 또는 [3, 4], dtype = np.float 으로 작성
# print(numerical_gradient(func_2, x))    # [ 6. 8.]
# print(numerical_gradient(func_2, np.array([.0, 2.])))   # [ 0. 4.]
# print(numerical_gradient(func_2, np.array([3., .0])))   # [ 6. 0.]

# 그래디언트는 함수값이 가장 크게 증가하는 방향을 가리킨다.
# func_2 : y = x0**2 + x1**2 은 (0,0) 에서 최소값을 가지고 (0,0) 을 중심으로 하는 원에서 같은 함수값을 보이는 형태.
# 점(3,4) 에서의 기울기는 (6,8) 이 된다. 중심에서 멀어지는 방향으로 가장 값이 빠르게 증가하는 방향이다.
# -1 을 곱하면 함수값이 가장 빠르게 감소하는 방향이 된다.

