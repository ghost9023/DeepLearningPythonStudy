import numpy as np
from PIL import Image
from common.activation_functions import sigmoid, softmax
from dataset.mnist import load_mnist
import pickle

def get_data():
    (x_train, t_train), (x_test, t_test) = load_mnist(normalize = True, flatten = True, one_hot_label = False)

    return x_train, t_train

def init_network():
    with open('sample_weight.pkl', 'rb') as f:
        network = pickle.load(f)

    return network

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

'''
학습된 가중치로 mnist 데이터 분류
'''
# x, t = get_data()
# print(x.shape, t.shape)
# network = init_network()
#
# accuracy_cnt = 0
# for i in range(len(x)) :
#     y = predict(network, x[i])
#     p = np.argmax(y)
#     if p == t[i] :
#         accuracy_cnt += 1
#
# print('Accuracy : ' + str(float(accuracy_cnt) / len(x)))  # 0.9357...

'''
배치처리
어떤 수만큼의 입력 데이터를 하나로 묶은 것을 배치 batch 라고 함
1개의 데이터씩 처리하는 것에 비교해 배치처리의 체감속도가 확연히 빠름
'''
# 입력데이터 벡터와 각 가중치 행렬의 형상 - 대응하는 차원수가 같음
# x, _ = get_data()
# network = init_network()
# W1, W2, W3 = network['W1'], network['W2'], network['W3']
# print('x shape : ', x.shape)
# print('x[0] shape : ', x.shape[0])
# print('W1 shape : ', W1.shape)
# print('W2 shape : ', W2.shape)
# print('W3 shape : ', W3.shape)

x, t = get_data()
network = init_network()

batch_size = 100    # 배치 크기
accuracy_cnt = 0

for i in range(0, len(x), batch_size) :
    x_batch = x[i:i+batch_size]
    y_batch = predict(network, x_batch)
    p = np.argmax(y_batch, axis = 1)    # axis = 1 : 1번째 차원을 축으로 최대값 획득
    accuracy_cnt += np.sum(p == t[i:i+batch_size])

print('Accuracy : ' + str(float(accuracy_cnt) / len(x)))

'''
mnist 데이터를 이미지로 보여주는 예제
'''
# def img_show(img):
#     '''
#     행렬을 이미지로 변환하여 보여줌
#     :param img: 행렬
#     '''
#     pil_img=Image.fromarray(np.uint8(img))
#     pil_img.show()
#
#
# # mnist 데이터를 다운로드
# (x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False)
#
# # 데이터의 형상 확인
# print(x_train.shape, t_train.shape)
# print(x_test.shape, t_test.shape)
#
# # 훈련데이터의 첫 데이터가 어떤 것인지 확인
# img = x_train[0]
# label = t_train[0]
# print(label)
# print(img.shape)
# img = img.reshape(28, 28)
# print(img.shape)
# img_show(img)