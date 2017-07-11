import numpy as np
import matplotlib.pyplot as plt
import mnist_example_2layers_p73 as me73
import dataset_mnist as dm

(trainImg, trainLbl), (testImg, testLbl) = dm.load_mnist(one_hot_label=True)

print(trainImg.shape, trainLbl.shape)
print(testImg.shape, testLbl.shape)

network = me73.MyTwoLayerNet(784, 50, 10)

# hyper parameters : 사람이 직접 결정하는 변수
itersNum = 1000 # 반복횟수
trainSize = trainImg.shape[0] # 학습데이터 건수 60000건
batchSize = 10 # mini batch 사이즈
learningRate = 0.1 # learning rate

# 누적기록
trainLossList=[]

print('-- start learning --')
for i in range(itersNum):
    # mini batch 획득
    miniBatchMask=np.random.choice(trainSize, batchSize)
    trainImgBatch=trainImg[miniBatchMask]
    trainLblBatch = trainLbl[miniBatchMask]
    # gradient 계산
    grad = network.numericalGradient(trainImgBatch, trainLblBatch)
    # weights, bias 갱신
    for key in ('W1', 'W2', 'b1', 'b2') :
        network.params[key] -= learningRate*grad[key]
    loss=network.loss(trainImgBatch, trainLblBatch)
    trainLossList.append(loss)
    print('iteration', i, ':', loss)

print('-- end learning --')