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
xy = np.loadtxt('DeepLearning/DeepLearning/02_Deep_ChoTH/data/data-01-test-score.csv', delimiter=',', dtype=np.float32)
x_data = xy[:, 0:-1]
y_data = xy[:, [-1]]
print(x_data.shape, x_data, len(x_data))
print(y_data.shape, y_data)
# 참고
# b = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])
# b[:, 1]   # 전체 행의 1번 열 다 출력
# b[-1]     # 마지막행
# b[-1, :]  # 마지막 행 전체 출력
# b[0:2, :] # 1,2번 행의 전체 열


