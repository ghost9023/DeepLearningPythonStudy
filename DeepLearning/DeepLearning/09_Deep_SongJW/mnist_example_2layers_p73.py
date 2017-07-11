import numpy as np

class MyTwoLayerNet :

    def __init__(self, inputSize, hiddenSize, outputSize, weightIniStd = 0.01):
        self.params={}
        self.params['W1']=weightIniStd * np.random.randn(inputSize, hiddenSize)
        self.params['W2']=weightIniStd * np.random.randn(hiddenSize, outputSize)
        self.params['b1']=np.zeros(hiddenSize)
        self.params['b2']=np.zeros(outputSize)

    def predict(self, x):
        W1=self.params['W1']
        W2=self.params['W2']
        b1=self.params['b1']
        b2=self.params['b2']
        z=sigmoid(np.dot(x, W1)+b1)
        y=softmax(np.dot(z, W2)+b2)
        return y

    def loss(self, x, t):
        y=self.predict(x)
        return crossEntropyError(y, t)

    def accuracy(self, x, t):
        y=self.predict(x)
        y=np.argmax(y, axis=1)
        t=np.argmax(t, axis=1)
        accuracy=np.sum(y==t)/float(x.shape[0])
        return accuracy

    def numericalGradient(self, x, t):
        lossW=lambda W : self.loss(x,t)
        grads={}
        grads['W1'] = numerical_gradient(lossW, self.params['W1'])
        grads['W2'] = numerical_gradient(lossW, self.params['W2'])
        grads['b1'] = numerical_gradient(lossW, self.params['b1'])
        grads['b2'] = numerical_gradient(lossW, self.params['b2'])
        return grads

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def softmax(x) :
    max_val = np.max(x, axis=1)
    numerator=np.exp(x-max_val.reshape(max_val.shape[0],1))
    denominator=np.sum(numerator)
    return numerator/denominator

def crossEntropyError(y, t) :
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))

def numerical_gradient(f, x) :
    is_vector = False
    h=1e-4
    if len(x.shape) == 1 :
        x=x.reshape(1,x.shape[0])
        is_vector = True
    grad=np.zeros(x.shape)
    for i in range(x.shape[0]) :
        for j in range(x.shape[1]) :
            tempX = x[i,j]
            x[i,j] = tempX+h
            fh1=f(0)

            x[i,j] = tempX-h
            fh2=f(0)

            partial_gradient=(fh1-fh2)/(2*h)
            grad[i,j]=partial_gradient
            x[i,j]=tempX
    if is_vector :
        x=x.reshape(x.shape[1],)
        grad=grad.reshape(x.shape[0],)
    return grad