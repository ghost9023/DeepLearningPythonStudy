import numpy as np
import matplotlib.pyplot as plt
import mnist_example_2layers_p73 as me73

network=me73.MyTwoLayerNet(10, 5, 2)
input_x=np.array([
    [1,2,3,4,5,6,7,8,9,10],
    [3,2,5,3,1,6,4,2,5,2],
    [5,2,1,3,5,3,2,3,5,10],
    [5,2,6,7,3,2,4,1,1,2],
    [7,5,4,5,2,2,1,5,3,1]
])
label_x=np.array([
    [0,1],
    [0,1],
    [1,0],
    [0,1],
    [1,0]
])

itersNum=1000
learningRate=0.01

trainLossList=[]

temp=network.params['W1'][2,2]

plt.ion()

for i in range(itersNum):
    grad=network.numericalGradient(input_x, label_x)
    for key in ('W1', 'W2', 'b1', 'b2'):
        network.params[key]-=learningRate*grad[key]
    loss=network.loss(input_x, label_x)
    trainLossList.append(loss)
    if i%10==0:
        plt.scatter(i, loss, color='r')
        plt.pause(0.01)
    print('iteration', i, ':', loss)

print(temp)
print(network.params['W1'][2,2])

while True :
    plt.pause(1)

