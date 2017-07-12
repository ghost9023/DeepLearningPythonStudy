import numpy as np
import matplotlib.pyplot as plt

def step_function(x):
    return np.array(x>0, dtype=np.int)

def sigmoid(x):
    return 1/(1+np.exp(-x))

def relu(x):
    return np.maximum(0,x)

if __name__ == '__main__':
    x=np.arange(-5, 5, 0.1)
    y=relu(x)
    plt.plot(x,y)
    plt.ylim(-0.1, 1.1)
    plt.show()