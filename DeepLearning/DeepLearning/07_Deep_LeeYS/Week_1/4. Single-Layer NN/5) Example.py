# 5) 예제 데이터셋 사용해보기


############KEYWORD###############




#################################



#신경망을 가장 쉽게 접할 수 있는 예제중 하나가 MNIST데이터.
#일반적으로 훈련용 이미지로 모델을 학습시키고 테스트용 이미지로 성능 평가.
#성능평가 기준은 분류 정확도.

import sys,os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

(x_train,t_train), (x_test,t_test) = load_mnist(flatten=True,normalize=False)
#flatten : 입력이미지를 1차원 배열로 만들지 정하는 옵션
#false로 설정하면 입력 이미지를 1x28x28의 3차원 배열로, True면 784개의 원소로 이루어진 1차원 배열로 저장
#1차원 배열로 저장한 데이터는 .reshape(픽셀값)으로 원래 이미지로 볼 수 있다. 예를 들어
#img.reshape(28,28)이라고 하면 원래 이미지인 28x28 픽셀의 이미지를 볼 수 있다.
#normalize는 이미지의 픽셀값을 0부터 1사이의 값으로 정규화 할 지 정하는 옵션.
#one_hot encoding이란 정답을 뜻하는 원소만 1이고 나머진 0으로 두는 인코딩 방법으로 이 인코딩 방법을 사용할 것인지 정하는 부분


#피클데이터로 저장된 가중치를 불러오는 코드
import pickle
with open(r"D:\Python\DeepLearningPythonStudy\DeepLearning\DeepLearning\07_Deep_LeeYS\Week_1\sample_weight.pkl",'rb') as f:
    network = pickle.load(f)

print(network)
#피클 파일을 작업환경 루트에 넣고 이 코드를 파이썬에서 실행해보면 가중치를 확인할 수 있다.
#이미지는 28*28 픽셀. 총 784개의 입력값을 가진다. 가중치도 784개가 존재

def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) #오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
def sigmoid(x):
    return 1 / (1+np.exp(-x))
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False,one_hot_label=False)
    return x_test, t_test
def init_network(): #가중치편향을 초기화, 인스턴스화
    with open(r"D:\Python\DeepLearningPythonStudy\DeepLearning\DeepLearning\07_Deep_LeeYS\Week_1\sample_weight.pkl",'rb') as f:
        network = pickle.load(f)
        return network
def predict(network,x):#은닉층 활성함수로 시그모이드 함수, 출력층 활성함수로 소프트맥스 함수를 쓴 순전파 신경망
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1,b2,b3 = network['b1'],network['b2'],network['b3']
    a1 = np.dot(x,W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1,W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2,W3) + b3
    y = softmax(a3)
    return y

x,t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network,x[i])
    p = np.argmax(y)
    if p == t[i]:
        accuracy_cnt += 1
print('ACCURACY : '+str(float(accuracy_cnt)/len(x)))