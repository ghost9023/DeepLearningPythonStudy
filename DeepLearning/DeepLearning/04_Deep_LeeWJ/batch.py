"""
mnist데이터 셋은 파일이 크기에, 첫 실행에서 다운 받은 후,
pickle로 로드하여 객체를 보원하는 식으로 속도를 줄일 수 있다.

"""

import sys, os
import numpy as np
from mnist import load_mnist
import pickle
from PIL import Image

sys.path.append(os.pardir) #부모 디렉터리의 파일을 가져올 수 있도록 설정한다.
#load_mnist 메소드의 3가지 매개변수
#1. flatten --> 입려기미지의 생성 배열 설정 false = 13차원배열, true = 1차원 배열
#1차원 배열저장한 데이터는 .reshape으로 원래 이미지를 볼 수 있다.
#2.normalize --> 0~ 1사이의 값으로 정규화 여부옵션
#3.one_hot encoding --> 정답을 뜻하는 원소만 1이고, 나머진 0으로 두는 인코딩 방법



## 하나로 묶은 입력 데이터를 배치라고 한다.
# 이미지가 지폐처럼 다발로 묶여있다고 생각하면 됨


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
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, flatten=True,one_hot_label=False)
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
batch_size = 100   # 배치크기 100

#################################################################
####################메인 명령문 ###################################
for i in range(0,len(x), batch_size):    #step = batch_size만큼 건너뛴다.
    x_batch = x[i:i+batch_size]   # i=0일때 x_batch = x[0:100], x[100:200]... 100장씩 묶어 꺼내게 된다.
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)  #해당 배열의 최댓값에 해당하는 원소의 인덱싱 가져오기
                                      #이는 100 X 10의 배열 중 1번째 차원을 구성하는 각 원소에서 최댓값의 인덱스를 찾도록 하는 것이다.
    accuracy_cnt += np.sum(p == t[i:i+batch_size])    # Bool배열을 만들어서, True가 몇개인지 센다.

print("정확도 : "+ str(float(accuracy_cnt) / len(x)))
#################################################################

#정확도 : 0.9352







############## np.argmax 예제 #############

#                    *                   *         *         *
x = np.array([[0.1, 0.8, 0.1], [0.3,0.1,0.6],[0.2,0.5,0.3],[0.8,0.1,0.1]])  # 1,2,1,0

y = np.argmax(x , axis=1)
print(y)  # 1,2,1,0 출력

######말 그대로 각 배열에서의 최댓값의 인덱스를 가져온다.