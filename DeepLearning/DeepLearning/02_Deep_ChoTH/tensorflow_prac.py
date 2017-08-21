##### 텐서 딥러닝 1장
import tensorflow as tf
hello = tf.constant('Hello, Tensorflow')
sess = tf.Session()
print(sess.run(hello))
# 'b'는 bytes literals라는 뜻이다.

node1 = tf.constant(3.0, tf.float32)  # 숫자, 데이터타입
node2 = tf.constant(4.0)  # 숫자, 데이터타입
node3 = tf.add(node1, node2)  # 숫자, 데이터타입
# node3 = node1 + node2   # 이렇게도 사용가능

print(node1)
print(node2)
print(node3)
sess = tf.Session()
print('sess.run(node1, node2):', sess.run([node1, node2]))
print('sess.run(node3):', sess.run(node3))

# 그래프는 미리 만들어놓고 실행시키는 단계에서 값을 주고 싶을 때
# placeholder
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b
print(sess.run(adder_node, feed_dict={a:3, b:4.5}))
print(sess.run(adder_node, feed_dict={a:[1,3], b:[2,4]}))

# tendor는 array를 말한다.
# 어레이의 랭크
# 0:scalar // 1:vector // 2:matrix // n:n-tensor.....

# tensor의 shape
# .shape()해서 나오는 모양

# type
# int32 // float32

# 정리
# 그래프를 설계, 빌드!
# 그래프 실행(sess.run, 변수설정)
# 결과 반환


#### 텐서 딥러닝 4장 - 파일에서 데이터 읽어오기
import numpy as np
import tensorflow as tf
xy = np.loadtxt('C:\python\DeepLearningPythonStudy\DeepLearning\DeepLearning\\02_Deep_ChoTH\data\data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
# 참고
# b = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# b[:, 1]    # 전체 행의 1번 열 다 출력
# b[-1]      # 마지막행
# b[-1, :]   # 마지막 행 전체 출력
# b[0:2, :]  # 1,2번 행의 전체 열

# 몇차원 어레이냐? -> 랭크, rank
# 어떤 모양의 어레이냐? -> 셰입, shape
# 축, axis
sess = tf.InteractiveSession()

t = tf.constant([1,2,3,4])
tf.shape(t).eval()

t = tf.constant([[1,2],
                 [3,4]])
tf.shape(t).eval()

t = tf.constant([[[[1, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12]],
                  [[13, 14, 15, 16], [17, 18, 19, 20], [21, 22, 23, 24]]]])
tf.shape(t).eval()

m1 = tf.constant([[1.,2.]])
m2 = tf.constant(3.)
tf.shape(m1+m2).eval()

tf.reduce_mean([1.,2.], axis=0).eval()   # integer이면 안된다. float!!!

x = [[1.,2.],
     [3.,4.]]
tf.reduce_mean(x).eval()
tf.reduce_mean(x, axis=1).eval()
tf.reduce_mean(x, axis=0).eval()   # 가장 바깥쪽의 축이 0이 된다.
tf.reduce_mean(x, axis=-1).eval()   # 가장 안쪽의 축이 -1이 된다.

tf.reduce_sum(x).eval()
tf.reduce_sum(x, 1).eval()
tf.reduce_sum(x, 0).eval()
tf.reduce_sum(x, -1).eval()   # 가장 안쪽

x = [[0,1,2],
     [2,1,0]]
tf.argmax(x).eval()      # 가장 큰 수의 인덱스를 반환하는 함수, 축을 적지 않으면 0으로 간주
tf.argmax(x, 1).eval()
tf.argmax(x, 0).eval()
tf.argmax(x, -1).eval()

t = np.array([[[0, 1, 2],
               [3, 4, 5]],

              [[6, 7, 8],
               [9, 10, 11]]])
t.shape

tf.reshape(t, shape=[-1,3]).eval()     # 안쪽은 3, 나머지는 알아서 해(-1), 2차원으로
tf.reshape(t, shape=[-1,1,3]).eval()   # 안쪽은 3, 그다음은 1, 나머지는 알아서(-1), 2차원으로

tf.squeeze([[0], [1], [2]]).eval()  # 차원축소
tf.expand_dims([0,1,2], 1).eval()   # 차원추가

# one hot
tf.one_hot([[0], [1], [2], [0]], depth=3).eval()  # 랭크가 자동으로 추가

t = tf.one_hot([[0], [1], [2], [0]], depth=3)   # 랭크가 자동적으로 추가되는 것을 막기 위해 reshape
tf.reshape(t, shape=[-1, 3]).eval()

tf.cast([1.8, 2.2, 3.3, 4.9], tf.int32).eval()
tf.cast([True, False, 1 == 1, 0 == 1], tf.int32).eval()

x = [1, 4]
y = [2, 5]
z = [3, 6]
# Pack along first dim.
tf.stack([x, y, z]).eval()
tf.stack([x, y, z], axis=0).eval()
tf.stack([x, y, z], axis=1).eval()

x = [[0, 1, 2],
     [2, 1, 0]]
tf.ones_like(x).eval()
tf.zeros_like(x).eval()

for x, y in zip([1,2,3], [4,5,6]):
    print(x, y)

for x, y, z in zip([1,2,3], [4,5,6], [7,8,9]):
    print(x, y, z)

# K = tf.sigmoid(tf.matmul(X, W1) + b1)
# hypothesis = tf.sigmoid(tf.matmul(K, W2) + b2)

# ML lab 09-1:Neural Net for XOR
# XOR 신경망 코드
import numpy as np
import tensorflow as tf
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
X = tf.placeholder(tf.float32)
Y  = tf.placeholder(tf.float32)
W = tf.Variable(tf.random_normal([2,1]), name='weight')
b = tf.Variable(tf.random_normal([1]), name='bias')
# 데이터가 적어서 softmax 함수 생략
hypothesis = tf.sigmoid(tf.matmul(X,W) + b)
cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))   # 손실함수 구하기
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)   # 경사감소법으로 손실함수 줄여나가기

# Accuracy computation
# True is hypothesis>0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Lounch graph
sess = tf.Session()
    # Initioalize Tensorflow variables
sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%100 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
# Accuracy report
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
print("\nHypothesis:", h, "\nCorrect:", c, "\nAccuracy:", a)
# 오류는 없지만 손실함수가 감소하지 않는다. 지나치게 단순해서, 1층!~
# accuracy : [0.50208956]


# 위의 망과 비슷한 2층 신경망
import numpy as np
import tensorflow as tf
x_data = np.array([[0,0], [0,1], [1,0], [1,1]], dtype=np.float32)
y_data = np.array([[0], [1], [1], [0]], dtype=np.float32)
X = tf.placeholder(tf.float32)
Y  = tf.placeholder(tf.float32)

W1 = tf.Variable(tf.random_normal([2,2]), name='weight1')   # 앞의 2는 데이터수, 뒤의 2는 노드수(출력값의 개수)
b1 = tf.Variable(tf.random_normal([2]), name='bias1')       # 바이어스는 출력값의 개수와 맞춰줘야 한다.
layer1 = tf.sigmoid(tf.matmul(X,W1) + b1)
# layer1 = tf.nn.relu(tf.matmul(X,W1) + b1)


W2 = tf.Variable(tf.random_normal([2,1]), name='weight2')
b2 = tf.Variable(tf.random_normal([1]), name='bias2')
hypothesis = tf.sigmoid(tf.matmul(layer1,W2) + b2)

cost = -tf.reduce_mean(Y*tf.log(hypothesis) + (1-Y) * tf.log(1-hypothesis))   # 손실함수 구하기
train = tf.train.GradientDescentOptimizer(learning_rate=0.1).minimize(cost)   # 경사감소법으로 손실함수 줄여나가기

# Accuracy computation
# True is hypothesis>0.5 else False
predicted = tf.cast(hypothesis>0.5, dtype=tf.float32)
accuracy = tf.reduce_mean(tf.cast(tf.equal(predicted, Y), dtype=tf.float32))

# Lounch graph
sess = tf.Session()
    # Initioalize Tensorflow variables
sess.run(tf.global_variables_initializer())

for step in range(1001):
    sess.run(train, feed_dict={X:x_data, Y:y_data})
    if step%100 == 0:
        print(step, sess.run(cost, feed_dict={X:x_data, Y:y_data}))
# Accuracy report
h, c, a = sess.run([hypothesis, predicted, accuracy], feed_dict={X:x_data, Y:y_data})
print("\nHypothesis:", h, "\nCorrect:", c, "\nAccuracy:", a)
# Accuracy: 0.75

# 층이 많다고 무조건 정확도가 올라가는 것이 아니다.
# 왜냐하면 오차역전파를 하면서 시그모이드에 의해 항상 1보다 작은 숫자가 계속 곱해지면서 최종적인 값이 점점 작아지게 된다.
# 뒤로 갈 수록, 즉 입력값에 가까울 수록 영향력이 작아지면서 기울기가 사라지게 된다. vanishing gradient
# 그래서 렐루를 사용한다. 마지막만 시그모이드를 사용한다. 0~1 사이의 값을 가져야하기 때문에

# 초기값을 줄 때 유의사항
# 1. 0을 주면 안된다.
# 2. RBM은 어려우니 싸비에르, He
# W = np.random.randn(fan_in, fan_out/np.sqrt(fan_in))    # 싸비에르
# W = np.random.randn(fan_in, fan_out/np.sqrt(fan_in/2))  # He

# CNN 제외하고 xavier, relu, dropout, adam 사용
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

nb_classes = 10
keep_prob = tf.placeholder(tf.float32)
X = tf.placeholder(tf.float32, [None, 784])
Y = tf.placeholder(tf.float32, [None, nb_classes])

###################################################
W1 = tf.get_variable("W1", shape=[784, 256], initializer=tf.contrib.layers.xavier_initializer())
b1 = tf.Variable(tf.random_normal([256]))
layer1 = tf.nn.relu(tf.matmul(X, W1) + b1)
layer1 = tf.nn.dropout(layer1, keep_prob=keep_prob)

W2 = tf.get_variable("W2", shape=[256, 128], initializer=tf.contrib.layers.xavier_initializer())
b2 = tf.Variable(tf.random_normal([128]))
layer2 = tf.nn.relu(tf.matmul(layer1, W2) + b2)
layer2 = tf.nn.dropout(layer2, keep_prob=keep_prob)

W3 = tf.get_variable("W3", shape=[128, nb_classes], initializer=tf.contrib.layers.xavier_initializer())
b3 = tf.Variable(tf.random_normal([nb_classes]))
hypothesis = tf.matmul(layer2, W3) + b3
###################################################

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

# Test model
is_correct = tf.equal(tf.arg_max(hypothesis, 1), tf.arg_max(Y, 1))
accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

training_epochs = 15
batch_size = 100
sess = tf.Session()
sess.run(tf.global_variables_initializer())
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)  # 1에폭 도는데 필요한 횟수
    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys, keep_prob:0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch
    print('Epoch:', '%04d' % (epoch +1), 'cost=', '{:.9f}'.format(avg_cost))
print("Accuracy:", accuracy.eval(session=sess, feed_dict={X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))


#### CNN 실습
import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
sess = tf.InteractiveSession()
image = np.array([[[[1],[2],[3]],
                   [[4],[5],[6]],
                   [[7],[8],[9]]]], dtype=np.float32)
print(image.shape)
plt.imshow(image.reshape(3,3), cmap='Greys')
plt.show()

##########################
### 2층 CNN 진짜 실습(mnist)
##########################
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100


nb_classes = 10
X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, nb_classes])

#L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))   # 필터의 크기, 색깔, 필터의 개수
# W1 = tf.get_variable("W1", shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer())???
# Conv통과 후   -> (?, 28, 28, 32)
# Pool통과 후   -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
print(L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # pooling 스트라이드 2
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)  # 1층에서 출력값!!!!!의 형태
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))   # 필터의 크기, 필터의 두께(L1의 출력값이랑 맞춘다.32), 필터의 개수(이미지 64개가 만들어짐)
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2_flat = tf.reshape(L2, [-1, 7*7*64])  # 다시 1차원으로 죽 펴준다.
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

L2 = tf.reshape(L2, [-1,7*7*64])  # 위에꺼 출력해보고 적는다.
W3 = tf.get_variable("W3", shape=[7*7*64, 10], initializer=tf.contrib.layers.xavier_initializer())
b = tf.Variable(tf.random_normal([10]))
hypothesis = tf.matmul(L2, W3) + b

# define cost/Loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=hypothesis, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=0.001).minimize(cost)

sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X: batch_xs, Y: batch_ys}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X: mnist.test.images, Y: mnist.test.labels}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))


##########################
### 3층 CNN 진짜 실습(mnist)
##########################
##########################
##########################
##########################
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data
import random
mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

learning_rate = 0.001
training_epochs = 15
batch_size = 100
keep_prob = tf.placeholder(tf.float32)
nb_classes = 10

X = tf.placeholder(tf.float32, [None, 784])
X_img = tf.reshape(X, [-1, 28, 28, 1])   # img 28x28x1 (black/white)
Y = tf.placeholder(tf.float32, [None, nb_classes])

#L1 ImgIn shape=(?, 28, 28, 1)
W1 = tf.Variable(tf.random_normal([3,3,1,32], stddev=0.01))   # 필터의 크기, 색깔, 필터의 개수
# W1 = tf.get_variable("W1", shape=[3,3,1,32], initializer=tf.contrib.layers.xavier_initializer())???
# Conv통과 후   -> (?, 28, 28, 32)
# Pool통과 후   -> (?, 14, 14, 32)
L1 = tf.nn.conv2d(X_img, W1, strides=[1,1,1,1], padding='SAME')
# print(L1)
L1 = tf.nn.relu(L1)
L1 = tf.nn.max_pool(L1, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')  # pooling 스트라이드 2
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
'''
Tensor("Conv2D:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("Relu:0", shape=(?, 28, 28, 32), dtype=float32)
Tensor("MaxPool:0", shape=(?, 14, 14, 32), dtype=float32)  # 1층에서 출력값!!!!!의 형태
'''

# L2 ImgIn shape=(?, 14, 14, 32)
W2 = tf.Variable(tf.random_normal([3, 3, 32, 64], stddev=0.01))   # 필터의 크기, 필터의 두께(L1의 출력값이랑 맞춘다.32), 필터의 개수(이미지 64개가 만들어짐)
#    Conv      ->(?, 14, 14, 64)
#    Pool      ->(?, 7, 7, 64)
L2 = tf.nn.conv2d(L1, W2, strides=[1, 1, 1, 1], padding='SAME')
L2 = tf.nn.relu(L2)
L2 = tf.nn.max_pool(L2, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)
'''
Tensor("Conv2D_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("Relu_1:0", shape=(?, 14, 14, 64), dtype=float32)
Tensor("MaxPool_1:0", shape=(?, 7, 7, 64), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 3136), dtype=float32)
'''

W3 = tf.Variable(tf.random_normal([3,3,64,128], stddev=0.01))
L3 = tf.nn.conv2d(L2, W3, strides=[1,1,1,1], padding='SAME')
L3 = tf.nn.relu(L3)
L3 = tf.nn.max_pool(L3, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)
L3 = tf.reshape(L3, [-1, 128*4*4])
'''
Tensor("Conv2D_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("Relu_2:0", shape=(?, 7, 7, 128), dtype=float32)
Tensor("MaxPool_2:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("dropout_2/mul:0", shape=(?, 4, 4, 128), dtype=float32)
Tensor("Reshape_1:0", shape=(?, 2048), dtype=float32)
'''

W4 = tf.get_variable("W4", shape=[128*4*4, 625], initializer=tf.contrib.layers.xavier_initializer())
b4 = tf.Variable(tf.random_normal([625]))
L4 = tf.nn.relu(tf.matmul(L3, W4) + b4)
L4 = tf.nn.dropout(L4, keep_prob=keep_prob)
'''
Tensor("Relu_3:0", shape=(?, 625), dtype=float32)
Tensor("dropout_3/mul:0", shape=(?, 625), dtype=float32)
'''

# L5 final fc 625 inputs -> 10 outputs
W5 = tf.get_variable("W5", shape=[625, 10], initializer=tf.contrib.layers.xavier_initializer())
b5 = tf.Variable(tf.random_normal([10]))
logits = tf.matmul(L4, W5) + 5
'''
Tensor("add_1:0", shape=(?, 10), dtype=float32)
'''

# define cost/Loss & optimizer
cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=logits, labels=Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# initialize
sess = tf.Session()
sess.run(tf.global_variables_initializer())

# train my model
print('Learning started. It takes sometime.')
for epoch in range(training_epochs):
    avg_cost = 0
    total_batch = int(mnist.train.num_examples / batch_size)

    for i in range(total_batch):
        batch_xs, batch_ys = mnist.train.next_batch(batch_size)
        feed_dict = {X:batch_xs, Y:batch_ys, keep_prob:0.7}
        c, _ = sess.run([cost, optimizer], feed_dict=feed_dict)
        avg_cost += c / total_batch

    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))

print('Learning Finished!')

# Test model and check accuracy
correct_prediction = tf.equal(tf.argmax(hypothesis, 1), tf.argmax(Y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_prediction, tf.float32))
print('Accuracy:', sess.run(accuracy, feed_dict={
      X:mnist.test.images, Y:mnist.test.labels, keep_prob:1}))

# Get one and predict
r = random.randint(0, mnist.test.num_examples - 1)
print("Label: ", sess.run(tf.argmax(mnist.test.labels[r:r + 1], 1)))
print("Prediction: ", sess.run(
    tf.argmax(hypothesis, 1), feed_dict={X: mnist.test.images[r:r + 1]}))