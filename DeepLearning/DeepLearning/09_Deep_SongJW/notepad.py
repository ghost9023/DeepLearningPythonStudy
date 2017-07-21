import numpy as np

def numerical_diff(f, x):
    h = 1e-4
    return (f(x+h) - f(x-h)) / (2*h)

def func(x) :
    y = .01 * x ** 2 + .1 * x
    return y

print(numerical_diff(func, 10))