import numerical_differentiaition as nd
import numpy as np

def gradientDecent(x, epsilon, n) :
    '''
    경사감소법
    :param x: init weights
    :param epsilon: learning rate
    :param n: number of method running
    :return: optimal weights
    '''
    xVal=np.array(x)
    for i in range(n) :
        grad=nd.numericalGradient(sample_function, xVal)
        xVal-=epsilon*grad
    return xVal

def sample_function(x) :
    return x[0]**2 + x[1]**2

w=np.array([-3.0, 4.0])

print('n = 100, init weights = [-3, 4]')
print('learning rate : 0.1\n', gradientDecent(w, 0.1, 100))
print('learning rate : 2\n', gradientDecent(w, 2, 100))
print('learning rate : 0.001\n', gradientDecent(w, 0.001, 100))