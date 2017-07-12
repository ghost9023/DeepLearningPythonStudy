import numpy as np
import matplotlib.pyplot as plt
import mnist_example_2layers_p73 as me73
import dataset_mnist as dm
# import warnings
#
# warnings.filterwarnings('ignore', '.*GUI is implemented.*')

(trainImg, trainLbl), (testImg, testLbl) = dm.load_mnist(one_hot_label=True)

print(trainImg.shape, trainLbl.shape)
print(testImg.shape, testLbl.shape)

network = me73.MyTwoLayerNet(784, 50, 10)

# hyper parameters : 사람이 직접 결정하는 변수
itersNum = 1 # 반복횟수
trainSize = trainImg.shape[0] # 학습데이터 건수 60000건
batchSize = 60000 # mini batch 사이즈
learningRate = 0.1 # learning rate

# 테스트데이터
testImg_part = testImg[0:100,]
testlbl_part = testLbl[0:100,]

# 누적기록
trainLossList=[]

plt.ion()
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
    plt.scatter(i, loss, color='r')
    plt.pause(0.01)
    print('iteration', i, ':', loss)

print('-- end learning --')

print(network.accuracy(testImg_part, testlbl_part))
