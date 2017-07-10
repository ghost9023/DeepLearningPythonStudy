import numpy as np
import pickle
import struct
import mnist
import activationFunctions as af

# read mnist data

trainLbl, trainImg=list(mnist.read(dataset='training', path='c:\\data\\mnist'))
testLbl, testImg=list(mnist.read(dataset='testing', path='c:\\data\\mnist'))
print(trainImg.shape)
print(trainLbl.shape)
print(testImg.shape)
print(testLbl.shape)

with open('c:\\data\\mnist\\sample_weight.pkl', 'rb') as f:
    network=pickle.load(f)

w1, w2, w3 = network['W1'], network['W2'], network['W3']
b1, b2, b3 = network['b1'], network['b2'], network['b3']

accuracyCnt=0
# for idx in range(len(trainImg)):
#     a1=np.dot(trainImg[idx], w1)+b1
#     z1=af.sigmoid_func(a1)
#     a2=np.dot(z1, w2) + b2
#     z2=af.sigmoid_func(a2)
#     a3=np.dot(z2,w3) + b3
#     y=af.softmax_func(a3)
#     p=np.argmax(y)
#     if p==trainLbl[idx] :
#         accuracyCnt+=1

print(w1.shape, w2.shape, w3.shape)
print(b1.shape, b2.shape, b3.shape)

batch_size=100
for idx in range(0, len(trainImg), batch_size):
    a1=np.dot(trainImg[idx:idx+batch_size], w1)+b1
    z1=af.sigmoid_func(a1)
    a2=np.dot(z1, w2) + b2
    z2=af.sigmoid_func(a2)
    a3=np.dot(z2,w3) + b3
    y=af.softmax_func(a3)   # 100 행 10 열 매트릭스 생성
    p=np.argmax(y,axis=1)   # axis=1 : 행단위로 최고값의 인덱스 반환
    accuracyCnt+=np.sum(p==trainLbl[idx:idx+batch_size])

print(accuracyCnt)
print('Accuracy: '+str(accuracyCnt/len(trainImg)))

'''
활성함수
sigmoid 이용시 overflow 1회 발생, 55511 적중, Accuracy: 0.9251833333333334
ReLU, parametric ReLU 이용시 정확도가 훨씬 떨어짐

배치처리
훨씬 빠름.
'''