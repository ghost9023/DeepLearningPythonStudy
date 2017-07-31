# 넘파이
# 넘파이의 산술연산
import numpy as np

x = np.array([1.0, 2.0, 3.0])
y = np.array([2.0, 4.0, 6.0])
x + y
x - y
x * y
x / y

x = np.array([1.0, 2.0, 3.0])
x / 2.0

A = np.array([[1,2], [3,4]])
print(A)
A.shape
A.dtype
B = np.array([[3,0], [0,6]])
A + B
A * B   # 배열연산

print(A)
A * 10

# 브로드캐스트
A = np.array([[1,2], [3,4]])
B = np.array([10, 20])
A * B

X = np.array([[51,55], [14,19], [0,4]])
print(X)
X[0]
X[0][1]

for row in X:
    print(row)

X > 15
X[X>15]   # 트루인 애들만 출력

# matplotlib
import numpy as np
import matplotlib.pyplot as plt
x = np.arange(0,6,0.1)
y = np.sin(x)
plt.plot(x, y)
plt.show()

x = np.arange(0, 6, 0.1)   # 0 ~ 6, 0.1 간격으로 생성
y1 = np.sin(x)
y2 = np.cos(x)

plt.plot(x, y1, label='sin')
plt.plot(x ,y2, linestyle='--', label='cos')
plt.xlabel('x')   # x축 이름
plt.ylabel('y')   # y축 이름
plt.title('sin & cos')   # 제목
plt.show()   # 안해도 그래프 그려짐

#
#