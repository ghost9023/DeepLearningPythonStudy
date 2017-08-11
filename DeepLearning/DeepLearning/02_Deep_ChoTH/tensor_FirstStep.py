###################################################
############### 4장. 단일계층신경망 #################
###################################################
# mnist데이터셋은 텐서플로 코드베이스에 통합되어있다. 따로 다운로드할 필요 없음
# mnist데이터셋 다운 및 로드
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets("MNINST_data/", one_hot=True)

# 구조확인, 현재 이 데이터는 배열형태의 객체이므로 텐서플로의 convert_to_tensor 함수를 이용해 텐서로 변환한 다음
# get_shape 함수를 사용해서 구조를 확인한다.
import tensorflow as tf
tf.convert_to_tensor(mnist.train.images).get_shape()
# TensorShape([Dimension(55000), Dimension(784)])  # 픽셀 784개
# 텐서의 모든 원소는 픽셀의 밝기를 나타내는 0에서 1사이의 값이다.
tf.convert_to_tensor(mnist.train.labels).get_shape()
# TensorShape([Dimension(55000), Dimension(10)])
# W와 b 매개변수를 구해서 가중치 합을 계산하고 나면 z에 저장된 결과를 0과 1사이의 실수로리턴하는 시그모이드 함수를 사용해야 한다.
# 시그모이드 함수에서 z가 충분히 큰 양수면 y는 1에 가까워지고 충분히 작은 음수면 y는 거의 0이 된다.


# 인공뉴런
# 선형회귀를 생각해보자. 일반화해서 말한다면 가중치W와 오프셋b를 학습해서 어떻게 점들을 분류하는지를 배워야한다.
# 출력계층에서 두가지 이상의 클래스로 데이터를 분류하고 싶을 때에는 시그모이드 함수의 일반화된 형태인 소프트맥스 함수를 활성화함수로 사용할 수 있다.
# 소프트맥스 함수는 출력값을 확률로 변환한다.
# 클래스 소속근거
# 주어진 클래스에는 없는 진한픽셀이 이미지에 있다면 가중치는 음의 값이 되고, 클래스의 진한 픽셀이 이미지와 자주 겹친다면 가중치는 양의 값이 된다.



###################################################
############### 5장. 다중계층신경망 #################
###################################################
from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data/', one_hot=True)
import tensorflow as tf

x = tf.placeholder('float', shape=[None, 784])  # 입력값을 담을 변수
y_ = tf.placeholder('float', shape=[None, 10])  # 라벨을 담을 변수 (mnist 라벨 10개)

x_image = tf.reshape(x, [-1,28,28,1])   # 입력 이미지를   cnn에 입력하기 위해 reshape
print('x_image=')
print(x_image)

def weight_variable(shape):   # 가중치의 값을 초기화 해주는 함수
    initial = tf.truncated_normal(shape, stddev=0.1)
    return tf.Variable(initial)

def bias_variable(shape):     # 바이어스를 초기화해주는 함수
    initial = tf.constant(0.1, shape=shape)
    return tf.Variable(initial)

def conv2d(x, W):   # 스트라이드는 1로 하고 패딩은 output 사이즈가 입력과달라지지 않게하겠다고 'SAME'을 사용한다.
    return tf.nn.conv2d(x, W, strides=[1,1,1,1], padding='SAME')  # 아주 간단하다.

def max_pool_2x2(x):
    return tf.nn.max_pool(x, ksize=[1,2,2,1], strides=[1,2,2,1], padding='SAME')
# ksize(filter 사이즈) 윈도루 사이즈를 2로 하고 스트라이드도 2로 하겠다.
# 스트라이드도 2로 하기 때문에 기존 사이즈가 절반으로 줄어들 것이다.

W_conv1 = weight_variable([5,5,1,32])
# 가로5, 세로5, inpput채널1, output채널32로 하는 가중치 매개변수 생성
# feature map을 32개 생성하겠다.
b_conv1 = bias_variable([32])   # 편향도 32개 만들어서 더해줌

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)  # 입력값과 가중치의 합성곱에 편향을 더한걸 relu함수에 집어넣음
h_pool1 = max_pool_2x2(h_conv1)   # 위의 결과를 풀링계층에 입력

W_conv2 = weight_variable([5,5,32,64])   # 두번째 conv계층에서 쓰일 가중치인데 가로5, 세로5, 입력값32, 출력값이64인 가중치 매개변수
b_conv2 = bias_variable([64])   # 편향 64개 생성

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)  # 첫번째 conv 계층과 pooling까지 통과한 결과가
h_pool2 = max_pool_2x2(h_conv2)
print(h_pool1.get_shape())

W_fc1 = weight_variable([7*7*64, 1024])   # affine 계층에서 쓰일 가중치 매개변수를 생성, 가로7, 세로7, feature map64, 노드의개수1024
b_fc1 = bias_variable([1024])   # 편향1024


h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc1) + b_fc1)

keep_prob = tf.placeholder('float')   # dropout을 수행할지 말지 결정하기 위한 변수 생성
h_fc1_drop = tf.nn.dropout(h_fc1, keep_prob)

W_fc2 = weight_variable([1024, 10])
b_fc2 = bias_variable([10])

y_conv = tf.nn.softmax(tf.matmul(h_fc1_drop, W_fc2) + b_fc2)
cross_entropy = -tf.reduce_sum(y_*tf.log(y_conv))

train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy)    # 오차함수에 오차를 최소화하는 경사감소법으로 Adam사용
correct_prediction = tf.equal(tf.argmax(y_conv,1), tf.argmax(y_,1))  # 실제라벨과 예상라벨을 비교한 결과를 correct_predition에 담기
accuracy = tf.reduce_mean(tf.cast(correct_prediction, 'float'))      # 불리언을 인티저로 바꿔 평균

sess = tf.Session()
sess.run(tf.global_variables_initializer())

for i in range(1000):
    batch = mnist.train.next_batch(100)
    if i%100 == 0:
        train_accuracy = sess.run(accuracy, feed_dict={x:batch[0], y_:batch[1], keep_prob:1.0})
        print("step %d, training accuracy %g"%(i, train_accuracy))
    sess.run(train_step, feed_dict={x:batch[0], y_:batch[1], keep_prob:0.5})

test_x, test_y = mnist.test.next_batch(1000)
print("test accuracy %g"% sess.run(accuracy, feed_dict={x:mnist.test.images, y_:mnist.test.labels, keep_prob:1.0}))
# 맨 마지막 test할 때는 dropout하지 않는다.