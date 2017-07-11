import numpy as np
import activationFunctions as af
import numerical_differentiation as nd
import cost_functions as cf

class SimpleNet:
    def __init__(self):
        self.w = np.random.randn(2,3)   # 정규분포로 2행 3열 매트릭스 생성
        print(self.w)

    def predict(self, x):
        input=np.dot(x, self.w)
        output=af.softmax_func(input)
        print(output)
        return output

    def loss(self, x, t):
        y=self.predict(x)
        loss=cf.crossEntropyError(y,t)
        return loss

sn = SimpleNet()
x=np.array([0.6, 0.9])
t=np.array([0,0,1])

def f(w) :
    return sn.loss(x, t)

def numericalGradient(f, w) :
    delta=1e-4
    grad=np.zeros(w.shape)
    for i in range(w.shape[0]) :
        for j in range(w.shape[1]) :
            temp=w[i,j]
            w[i,j]=temp+delta
            fh1=f(w)

            w[i,j]=temp-delta
            fh2=f(w)
            grad[i,j]=(fh1-fh2)/(2*delta)
            w[i,j]=temp
    return grad

gradient = numericalGradient(f, sn.w)
print(gradient)