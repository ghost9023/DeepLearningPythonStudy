import numpy as np
import matplotlib.pyplot as pyplot

def step_func(x) :
    return np.array(x>0, dtype=np.int)

def sigmoid_func(x) :
    return 1/(1+np.exp(-x))

def ReLU_func(x):
    return np.maximum(0, x)

def parametric_ReLU_func(x):
    a=0.3
    return np.where(x>0, x, a*x)

x=np.arange(-5, 5, 0.1)
y_step=step_func(x)
y_sig=sigmoid_func(x)
y_ReLU=ReLU_func(x)
y_para_ReLU=parametric_ReLU_func(x)

pyplot.plot(x,y_step)
pyplot.show()