import numpy as np

'''
벡터, 행렬의 생성, 차원수, 형상
'''
# A=np.array([1,2,3,4])
# print(A)
# print(np.ndim(A))   # ndim() : 차원 반환
# print(A.shape)  # shape : 튜플 형태로 형상 반환. 벡터의 경우 반환된 튜플이 한개의 원소만 갖음. (4,)
# print(A.shape[0])

# B=np.array([[1,2],[3,4],[5,6]])
# print(B)
# print(np.ndim(B)) # 2
# print(B.shape)    # (3,2)

'''
행렬의 내적
'''
# A=np.array([[1,2],[3,4]])
# print(A.shape)
# B=np.array([[5,6],[7,8]])
# print(B.shape)
# print(np.dot(A,B))  # dot(A,B) : 내적. 일반적으로 dot(A,B) != dot(B,A)

# A=np.array([[1,2,3],[4,5,6]])
# print(A.shape)
# B=np.array([[1,2],[3,4],[5,6]])
# print(B.shape)
# print(np.dot(A,B))
# print(np.dot(B,A))  # 2x3 X 3x2 = 2x2, 3x2 X 2x3 = 3x3

# 에러. 앞 행렬의 열 수와 뒷 행렬의 행 수가 일치하지 않음
# A=np.array([[1,2,3],[4,5,6]])
# C=np.array([[1,2], [3,4]])
# print(C.shape)
# print(A.shape)
# print(np.dot(A,C))    # ValueError: shapes (2,3) and (2,2) not aligned: 3 (dim 1) != 2 (dim 0)
                        # index : 행 = 0, 열 = 1

# # 행렬과 벡터의 곱.
# A=np.array([[1,2],[3,4],[5,6]])
# print(A.shape)
# B=np.array([7,8])
# print(B.shape)
# C=np.dot(A,B)
# print(C, C.shape)

####################################################################################
#
# print('\n원소접근')
# a=np.array([[51, 55],[14, 19],[0,4]])
# print(a)
# print(a[0])
# print(a[0][1])
# b=np.array([1,2,3,4,5,6])
# print(b[np.array([0,1,3])]) # 인덱스벡터로 벡터 원소에 접근
# x=np.array([10,20,25,30,5,10])
# print(x[x>15])  # 원소에 조건걸기
# print(x>15) # bool 벡터 생성


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
# print(x2.flatten())   # 메트릭스를 벡터로 평탄화
# print(x2.reshape(4,1))    # 메트릭스 형태 변환
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

# print('\n벡터 원소에 접근')
# a=np.array([1,2,3,4,5])
# print(a.size) # 벡터 사이즈
# print(a[3])
#
# print('\n벡터, 매트릭스의 복사')
# b=a
# c=a[:]
# print(id(a), id(b), id(c))
# d=np.array([[1,2],[3,4]])
# e=d
# f=d[:]
# print(id(d),id(e),id(f))