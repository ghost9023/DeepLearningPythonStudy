"""
mnist데이터 셋은 파일이 크기에, 첫 실행에서 다운 받은 후,
pickle로 로드하여 객체를 보원하는 식으로 속도를 줄일 수 있다.

"""

import sys, os
import numpy as np
from mnist import load_mnist
import pickle
sys.path.append(os.pardir) #부모 디렉터리의 파일을 가져올 수 있도록 설정한다.
#load_mnist 메소드의 3가지 매개변수
#1. flatten --> 입려기미지의 생성 배열 설정 false = 13차원배열, true = 1차원 배열
#1차원 배열저장한 데이터는 .reshape으로 원래 이미지를 볼 수 있다.
#2.normalize --> 0~ 1사이의 값으로 정규화 여부옵션
#3.one_hot encoding --> 정답을 뜻하는 원소만 1이고, 나머진 0으로 두는 인코딩 방법

# with open('sample_weight.pkl', 'rb') as f:
#     network= pickle.load(f)
#     print(network)
#
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
# 소프트맥스함수
def softmax(a):
    c = np.max(a)
    exp_a = np.exp(a-c) # 오버플로 방지
    sum_exp_a = np.sum(exp_a)
    y = exp_a / sum_exp_a
    return y
# 시그모이드함수
def sigmoid(x):
    return 1 / (1 + np.exp(-x))
def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,
    one_hot_label=False)
    return x_test, t_test
# 가중치와 편향을 초기화, 인스턴스화
def init_network():
    with open("sample_weight.pkl", 'rb') as f:
        network = pickle.load(f)
        return network
# 은닉층 활성함수로 시그모이드함수, 출력층 활성함수로 소프트맥스함수를 쓴 순전파 신경망
def predict(network, x):
    W1, W2, W3 = network['W1'], network['W2'], network['W3']
    b1, b2, b3 = network['b1'], network['b2'], network['b3']
    a1 = np.dot(x, W1) + b1
    z1 = sigmoid(a1)
    a2 = np.dot(z1, W2) + b2
    z2 = sigmoid(a2)
    a3 = np.dot(z2, W3) + b3
    y = softmax(a3)
    return y
x, t = get_data()
network = init_network()
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p= np.argmax(y) # 확률이 가장 높은 원소의 인덱스를 얻는다.
    if p == t[i]:
        accuracy_cnt += 1
        print("Accuracy:" + str(float(accuracy_cnt) / len(x)))
