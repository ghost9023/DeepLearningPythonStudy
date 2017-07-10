import sys, os
import numpy as np
import pickle
import random
from PIL import Image

# https://github.com/oreilly-japan/deep-learning-from-scratch/common, dataset
sys.path.append(os.pardir)
from dataset_mnist import load_mnist
from common_functions import sigmoid, softmax


class NeuralNetMnist:
    def __init__(self):
        # 1. MNIST 데이터 로드하기.
        (self.trainX, self.trainT), (self.testX, self.testT) = load_mnist(normalize=True, flatten=True,
                                                                          one_hot_label=False)

        # X = images, T = labels

        # 2. 이미 학습된 신경망 객체를 로드하기.
        # https://github.com/oreilly-japan/deep-learning-from-scratch/blob/master/ch03/sample_weight.pkl
        with open("sample_weight.pkl", "rb") as f:
            self.network = pickle.load(f)

    # 추론하기. 확률로 리턴.
    def predict(self, x):
        # x: 784 바이트 이미지 데이터.
        # self.network: 이미 학습된 신경망 객체.
        network = self.network

        W1, W2, W3 = network['W1'], network['W2'], network['W3']
        b1, b2, b3 = network['b1'], network['b2'], network['b3']

        a1 = np.dot(x, W1) + b1
        z1 = sigmoid(a1)
        a2 = np.dot(z1, W2) + b2
        z2 = sigmoid(a2)
        a3 = np.dot(z2, W3) + b3
        y = softmax(a3)

        # y: [0.1, 0.3, 0.2, ...] 해당 인덱스의 수가 답일 확률.
        return y

        # 추론하기. 예상되는 숫자로 리턴.

    def predictNumber(self, x):
        return np.argmax(self.predict(x))

    # 정확도 구하기.
    def getAccuracy(self):
        x = self.testX
        t = self.testT
        correctCount = 0  # 정답을 맞춘 횟수.
        for i in range(len(x)):  # 10000 번
            y = self.predict(x[i])
            p = np.argmax(y)  # 가장 높은 확률을 가진 인덱스. 즉, 예측되는 숫자.
            if p == t[i]:  # 예측값이 실제 답과 같다면?
                correctCount += 1

        print("정확도: " + str(float(correctCount) / len(x)))

    # 묶음방식으로 정확도 구하기.
    def getAccuracyBatch(self, batchSize):
        x = self.testX
        t = self.testT
        correctCount = 0

        for i in range(0, len(x), batchSize):
            x_batch = x[i:i + batchSize]
            y_batch = self.predict(x_batch)  # y_batch: 2차원 배열, 100 x 10
            p = np.argmax(y_batch, axis=1)  # axis=1 은 두번째 차원을 의미.
            correctCount += np.sum(p == t[i:i + batchSize])

        print("정확도: " + str(float(correctCount) / len(x)))

    # 데이터 형태 출력하기.
    def printDataShape(self):
        print("MNIST")
        print("\ttrainX.shape: " + str(self.trainX.shape))  # (60000, 784) : 학습용 손글씨 이미지 데이터. 784=28x28x8bitGray
        print("\ttrainT.shape: " + str(self.trainT.shape))  # (60000,)     : 위 이미지가 의미하는 실제 수. (0~9)
        print("\ttestT.shape: " + str(self.testX.shape))  # (10000, 784) : 테스트용 손글씨 이미지 데이터.
        print("\ttestT.shape: " + str(self.testT.shape))  # (10000,)     : 위 이미지가 의미하는 실제 수. (0~9)
        print("Network")
        print("\tW1.shape: " + str(self.network['W1'].shape))  # (784, 50)
        print("\tW2.shape: " + str(self.network['W2'].shape))  # (50, 100)
        print("\tW3.shape: " + str(self.network['W3'].shape))  # (100, 10)
        print("\tb1.shape: " + str(self.network['b1'].shape))  # (50,)
        print("\tb2.shape: " + str(self.network['b2'].shape))  # (100,)
        print("\tb3.shape: " + str(self.network['b3'].shape))  # (10,)

    # 하나의 숫자 이미지를 테스트하기.
    def test(self):
        n = random.randrange(0, len(self.testX))
        x = self.testX[n]
        img = x.reshape(28, 28)
        pil_img = Image.fromarray(np.uint8(img * 255))
        print("이 숫자는 " + str(self.predictNumber(m.testX[n])) + " 같습니다.")
        pil_img.show()


m = NeuralNetMnist()
m.printDataShape()
m.getAccuracy()
m.test()