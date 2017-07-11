import numpy as np
import matplotlib.pyplot as pyplot

def step_func(x) :
    '''
    안쓴다. 
    '''
    return np.array(x>0, dtype=np.int)

def sigmoid_func(x) :
    '''
    안쓴다. 
    '''
    return 1/(1+np.exp(-x))

def ReLU_func(x):
    '''
    쓴다. 
    '''
    return np.maximum(0, x)

def parametric_ReLU_func(x):
    '''
    쓴다 
    '''
    a=0.2
    return np.where(x>0, x, a*x)

def identity_func(x):
    '''
    출력층의 활성함수
     항등함수. 출력=입력
     회귀문제에서 사용
    '''
    return x

def softmax_func_overflow(x):
    '''
    출력층 활성함수
     신경망의 출력값으로 확률벡터를 얻는다. (해석 용이)
     분류문제에서 사용
     신경망의 출력값을 0~1 로 제한. 모든 출력값의 합은 1.
    '''
    expX=np.exp(x)
    sumX=np.sum(expX)
    return expX/sumX

def softmax_func(x):
    '''
    출력층 활성함수
     신경망의 출력값으로 확률벡터를 얻는다. (해석 용이)
     분류문제에서 사용
     신경망의 출력값을 0~1 로 제한. 모든 출력값의 합은 1.
     이전 버전에서 오버플로우가 발생하므로 x 에서 x 의 최대값을 빼서 오버플로우를 막는다.
    '''
    maxX=np.max(x)
    expX=np.exp(x-maxX)
    sumX=np.sum(expX)
    return expX/sumX

if __name__ == '__main__' :
    print('activation function')
    x=np.arange(-5, 5, 0.1)
    y_step=step_func(x)
    y_sig=sigmoid_func(x)
    y_ReLU=ReLU_func(x)
    y_para_ReLU=parametric_ReLU_func(x)

    pyplot.plot(x,y_step, '--')
    pyplot.plot(x,y_sig, '-.')
    pyplot.plot(x,y_ReLU,':')
    pyplot.plot(x,y_para_ReLU, ',')
    pyplot.show()

    print('\nsoftmax function')
    x=np.array([2.3, -0.9, 3.6])
    y=softmax_func_overflow(x)
    print(y, np.sum(y))

    # x1 = np.array([900, 1000, -1000]) # 입력이 너무 큰 경우 overflow
    # y1 = softmax_func_overflow(x1)
    # print(y1, np.sum(y1))

    print('\n개선된 softmax function') # overflow 가 해결됨
    x2=np.array([900, 1000, 1000])
    y2=softmax_func(x2)
    print(y2, np.sum(y2))

