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

def sigmoid(x) :
    return 1/(1+np.exp(-x))

def softmax(x) :
    max_val = max(x)
    numerator=np.exp(x-max_val)
    denominator=np.sum(numerator)
    return numerator/denominator

def crossEntropyError(y, t) :
    delta = 1e-7
    return -np.sum(t*np.log(y+delta))