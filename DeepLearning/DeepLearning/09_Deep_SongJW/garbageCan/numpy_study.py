import numpy as np

# print('create array\n')
# a=np.array([1,5])
# b=np.array([[1,2],[2,3]])
# c=np.array([[1],[2],[3]])
# d=np.arange(1,5,1)  # 1~4 까지 1 간격으로 나열
# e=np.arange(1,7,1).reshape(2,3) # 1~6 까지 1간격으로 2행 3열 배치
# print(a)
# print(b)
# print(c)
# print(d)
# print(e,'\n')
#
# print('operation\n')
# x1=np.array([1,2,3])
# y1=np.array([5,10,15])
# x2=np.array([[1,2],[3,4]])
# y2=np.array([[5,10],[15,20]])
# z1=np.array([-1, -2])
# z2=np.array([[5],[10],[15]])
#
# print('일반 연산은 대응하는 원소끼리')
# print(x1+y1)
# print(x1-y1)
# print(x1*y1)
# print(x1/y1)
# print(x2+y2)
# print(x2*y2,'\n')
#
# print('브로드캐스팅\n매트릭스의 열의 수와 벡터의 원소수가 같은 경우의 연산은\n'
#       '벡터를 매트릭스의 행의 수만큼 복제하여 대응하는 원소끼리 연산')
# print(x2+z1)
# print(x2*z1,'\n')
#
# print('기타')
# print(x1**2)
# print(x1>=2)
# print(x2.flatten())
# print(x2.reshape(4,1))
# print(x2.reshape(1,4))
#
# print('쌓기')
# a=np.array([1,2,3])
# b=np.array([3,4,5])
# print(a.shape,b.shape)
# print(np.vstack([a,b]))
# print(np.hstack([a,b]))
#
# print('일반함수')
# a=np.array([1,2,3,6,5,4])
# print(np.argmax(a), a[np.argmax(a)]) # 차례로 최대값의 인덱스, 인덱스로 출력한 최대값
# a=np.array([[1,2,3],[4,6,5],[9,8,7]])
# print(np.argmax(a,axis=0), np.argmax(a,axis=1)) # axis : 0은 열단위로, 1은 행단위로 최대값의 인덱스 반환
# print()
#
# print('전치')
# a=np.array([[1,2,3],[4,5,6]])
# print(a,'\n',np.transpose(a))
# b=np.array([1,2,3,4,5])
# print(np.transpose(b))  # 벡터는 전치가 되지 않는다.
#
# print('\n내적-dot')
# a=np.array([[1,2],[3,4]])
# b=np.array([[5,6],[7,8]])
# c=np.array([1,2,3])
# d=np.array([[1],[2],[3]]) # 벡터끼리의 곱은 행벡터, 열벡터 간의 곱과 같다.
# print(np.dot(a,b))
# print(np.dot(c,d))
#
# print('\n신경망의 두 레이어 사이의 모습-2입력과 3노드의 연결')
# input=np.array([1,2])   # 입력 1, 2
# weight=np.array([[1,3,5],[2,4,6]])  # 노드의 연결이 차례로 (1,2), (3,4), (5,6) 가중치를 가짐
# net_input=np.dot(input,weight)
# print(net_input)

print('\n벡터 원소에 접근')
a=np.array([1,2,3,4,5])
print(a.size) # 벡터 사이즈
print(a[3])

print('\n벡터, 매트릭스의 복사')
b=a
c=a[:]
print(id(a), id(b), id(c))
d=np.array([[1,2],[3,4]])
e=d
f=d[:]
print(id(d),id(e),id(f))