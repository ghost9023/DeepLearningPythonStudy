import numpy as np
import activationFunctions as af

w1=np.array([[0.1, 0.3, 0.5], [0.2, 0.4, 0.6]])
w2=np.array([[0.1, 0.4], [.2, .5], [.3, .6]])
w3=np.array([[.1, .3], [.2, .4]])
b1=np.array([.7, .8, .9])
b2=np.array([.7, .8])
b3=np.array([.7, .8])

ia=np.array([4.5, 6.2])
z1=af.sigmoid_func(np.dot(ia, w1)+b1)
print('1st 은닉층 값 : '+str(z1))
z2=af.sigmoid_func(np.dot(z1, w2)+b2)
print('2nd 은닉층 값 : '+str(z2))
y=af.identity_func(np.dot(z2,w3)+b3)
print('출력값 : '+str(y))

