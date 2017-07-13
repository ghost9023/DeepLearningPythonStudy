import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    '''
    계단 함수
    1 (0 < x)
    0 (0 >= x)
    '''
    return np.array(x>0, dtype=np.int)

def sigmoid(x):
    '''
    시그모이드
     0~1 사이의 연속적인 값 반환
    '''
    return 1/(1+np.exp(-x))

def relu(x):
    '''
    렐루
     x (0 < x)
     0 (0 >= x)
    '''
    return np.maximum(0,x)

def identity_function(x):
    '''
    regression 출력층에 사용
     입력을 그대로 반환
    '''
    return x

def softmax(x):
    '''
    classification 출력층에 사용
     출력층의 입력 전체를 입력으로 요구함
     출력의 합은 1이 됨 -> 출력을 확률로 해석할 수 있음
    '''
    max_x = np.max(x)
    exp_x = np.exp(x - max_x)
    sum_exp_x = np.sum(exp_x)
    y = exp_x / sum_exp_x
    return y

if __name__ == '__main__':
    '''
    plot
    '''
    # x=np.arange(-5, 5, 0.1)
    # y=relu(x)
    # plt.plot(x,y)
    # plt.ylim(-0.1, 1.1)
    # plt.show()

    '''
    softmax 내용
    '''
    # a = np.array([.3, 2.9, 4.0])
    # exp_a=np.exp(a)
    # print(exp_a)
    # sum_exp_a = np.sum(exp_a)
    # print(sum_exp_a)
    # y = exp_a / sum_exp_a
    # print(y)
    # print(softmax(a))

    a = np.array([1010, 1000, 990])
    # print(np.exp(a)/np.sum(np.exp(a)))  # [ nan  nan  nan] -> RuntimeWarning: overflow encountered in exp
    c = np.max(a)   # 1010
    print(a - c)
    print(np.exp(a-c)/np.sum(np.exp(a-c)))  # overflow 해결
