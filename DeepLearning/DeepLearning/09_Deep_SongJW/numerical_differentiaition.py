import numpy as np
import matplotlib.pyplot as pypl

'''
일반적인 미분. 기울기 = (f(x+h)-f(x-h))/2*h
'''
def numericalDiff(f, x) :
    h=1e-4
    return (f(x+h)-f(x-h))/(2*h)

def sample_func1(x) :
    return 0.01*x**2+0.1*x

'''
편미분 : 두개 이상의 독립변수가 존재하는 함수에서 하나의 독립변수에 대한 편미분은
나머지 독립변수들을 상수로 취급하여 미분하는 것. 두개 이상의 독립변수를 가진 함수의 기울기는
모든 독립변수에 대한 편미분을 벡터로 정리한 것을 말한다.
'''

def numericalGradient(f, x) :
    h=1e-4
    grad=np.zeros(x.size)
    for i in range(x.size):
        x_copy=np.array(x)

        x_copy[i]=x[i]+h
        fh1=f(x_copy)

        x_copy[i]=x[i]-h
        fh2=f(x_copy)

        grad[i]=(fh1-fh2)/(2*h)

    return grad

def sample_func2(x) :
    return x[0]**2 + x[1]**2

# x=np.arange(0, 20, 0.1)
# y=sample_func1(x)
# pypl.xlabel('x')
# pypl.ylabel('f(x)')
# pypl.plot(x,y)
# pypl.show()
#
# print(numericalDiff(sample_func1,5))
# print(numericalDiff(sample_func1,10))

print(numericalGradient(sample_func2,np.array([3.0, 4.0])))
print(numericalGradient(sample_func2,np.array([0.0, 2.0])))
print(numericalGradient(sample_func2,np.array([3.0, 0.0])))
