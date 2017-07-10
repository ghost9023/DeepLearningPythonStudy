import numpy as np

print('create array\n')
a=np.array([1,5])
b=np.array([[1,2],[2,3]])
c=np.array([[1],[2],[3]])
d=np.arange(1,5,1)  # 1~4 까지 1 간격으로 나열
e=np.arange(1,7,1).reshape(2,3) # 1~6 까지 1간격으로 2행 3열 배치
print(a)
print(b)
print(c)
print(d)
print(e,'\n')

print('operation\n')
x1=np.array([1,2,3])
y1=np.array([5,10,15])
x2=np.array([[1,2],[3,4]])
y2=np.array([[5,10],[15,20]])
z1=np.array([-1, -2])
z2=np.array([[5],[10],[15]])

print('일반 연산은 대응하는 원소끼리')
print(x1+y1)
print(x1-y1)
print(x1*y1)
print(x1/y1)
print(x2+y2)
print(x2*y2,'\n')

print('브로드캐스팅\n매트릭스의 열의 수와 벡터의 원소수가 같은 경우의 연산은\n'
      '벡터를 매트릭스의 행의 수만큼 복제하여 대응하는 원소끼리 연산')
print(x2+z1)
print(x2*z1,'\n')

print('기타')
print(x1**2)
print(x1>=2)
print(x2.flatten())
print(x2.reshape(4,1))
print(x2.reshape(1,4))

print('쌓기')
a=np.array([1,2,3])
b=np.array([3,4,5])
print(a.shape,b.shape)
print(np.vstack([a,b]))
print(np.hstack([a,b]))