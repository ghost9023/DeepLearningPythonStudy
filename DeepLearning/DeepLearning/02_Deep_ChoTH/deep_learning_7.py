# CHAPTER 7 합성곱 신경망(CNN)
# 전체구조
# 합성곱계층과 풀링계층이 추가된다.
# 지금까지 본 신경망은 인접하는 계층의 모든 뉴런과 결합되어 있다. 이를 완전연결이라고 하며, 완전히 연결된 계층을 Affine계층이라는 이름으로 구현했다.
##########################################
# 2차원 배열 합성곱 #@#!@#!$!$!@#@!$%!@#!@#
##########################################
import numpy as np
data = np.array(range(0,81)).reshape(9,9)
filter = np.array(range(0,16)).reshape(4,4)

def find_pad(data, filter, s, oh):
    h = len(data)
    fh = len(filter)
    return (((oh-1)*s)+fh-h) / 2

def padding(data, x):
    if x%1 == 0:
        x = int(x)
        return np.pad(data, pad_width=x, mode='constant', constant_values=0)
    else:
        x1 = int(x+0.5)
        x2 = int(x-0.5)
        return np.pad(data, pad_width=((x1,x2), (x1,x2)), mode='constant', constant_values=0)

def output(data, filter):
    num = len(data) - len(filter) + 1
    result = []
    for rn in range(num):
        for cn in range(num):
            result.append(np.sum(data[rn:rn+len(filter), cn:cn+len(filter)] * filter))
    return np.array(result).reshape(num, num)

f_p = find_pad(data, filter, 1, 9)   # Straid(s) / 출력값(oh)
data = padding(data, f_p)
print('q3\n', output(data, filter))
print('q4\n', output(data, filter) * 3)


############################################
# 3차원 배열 합성곱!@#!@#!@#!@#!@#@!#!
##############################################
import numpy as np
def find_pad(data, filter, s, oh):
    h = len(data[0])
    fh = len(filter[0])
    return (((oh-1)*s)+fh-h) / 2

def padding(data, x):
    if x%1 == 0:
        x = int(x)
        lst = []
        for i in range(len(data)):
            lst.append(np.pad(data[i], pad_width=x, mode='constant', constant_values=0))
            lst = np.array(lst)
            return lst
    else:
        x1 = int(x+0.5)
        x2 = int(x-0.5)
        lst = []
        for i in range(len(data)):
            lst.append(np.pad(data[i], pad_width=((x1,x2), (x1,x2)), mode='constant', constant_values=0))
            lst = np.array(lst)
            return lst

def output(data, filter):
    num = len(data[0]) - len(filter[0]) + 1   # 가장 상위 차원의 수는 입력과 필터가 같다. 행과 열은 정사각형 형태를 이루지만 두께까지 같은 정육면체는 아니다.
    result = []
    for i in range(len(data)):
        for rn in range(num):
            for cn in range(num):
                result.append(np.sum(data[i, rn:rn+len(filter[0]), cn:cn+len(filter[0])] * filter[i]))
    return np.array(result).reshape(len(data), num, num)

data = np.array([[[1,2,0,0], [0,1,-2,0], [0,0,1,2], [2,0,0,1]], [[1,0,0,0], [0,0,-2,-1], [3,0,1,0], [2,0,0,1]]])
filter = np.array([[[-1,0,3], [2,0,-1], [0,2,1]], [[0,0,0], [2,0,-1], [0,-2,1]]])

f_p = find_pad(data, filter, 1, 3)   # Straid(s) / 출력값(oh)
data = padding(data, f_p)
print('q\n', output(data, filter))
