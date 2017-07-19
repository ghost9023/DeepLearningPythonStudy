import numpy as np

def AND_GATE(x):
    vec_x = np.array(x)
    w = [0.5, 0.5]
    vec_w = np.array(w)
    b = -0.7

    dot_product = np.sum(vec_x * vec_w)
    net_value = dot_product + b

    return False if net_value <= 0 else True


print(AND_GATE([0, 1]))

def NAND_GATE(x):
    vec_x = np.array(x)
    w = [-0.5, -0.5]
    vec_w = np.array(w)
    b = 0.7

    dot_product = np.sum(vec_x * vec_w)
    net_value = dot_product + b

    return False if net_value <= 0 else True


print(NAND_GATE([0, 1], ))

def OR_GATE(x):
    vec_x = np.array(x)
    w = [0.5, 0.5]
    vec_w = np.array(w)
    b = -0.2

    dot_product = np.sum(vec_x * vec_w)
    net_value = dot_product + b

    return False if net_value <= 0 else True


print(OR_GATE([0, 1], ))


#XOR 배타적 논리합 게이트는 단층 퍼셉트론으론 구현 불가능하다.
'''
단층퍼셉트론은 선형방정식을 사용하여 0 또는 1을 구별하는데, 선형으로만 그려지니, 분류가 단순함.
XOR는 자기자신과 같지 않은것은 배타해버리니, 2가지 경우가 False되면서
곡선형으로 분류할 수 있다.

하지만 다층 퍼셉트론의 학습에서 나타나는 느린 학습 속도와 지역 극소는 실제 응용문제에 적용함에
있어서 가장 큰 문제로 지적되어왔습니다. 따라서 다층 퍼셉트론은 딥러닝의 개념과 뉴런 회로구조를 이해하는 용도로 사용하고 
이제는 신경망을 공부해야 한다.

'''
import numpy as np

def calc(x,w, b=-0.7):
    array_x = np.array(x)
    array_w = np.array(w)
    y = np.sum(array_x*array_w)
    return y+b


calc([0,1],[0.5,0.5])





def AND(x,b=-0.7):
    array_x = np.array(x)
    array_w = np.array([0.5, 0.5])
    threshold = np.sum(array_x*array_w)+b
    return 0 if threshold <= 0 else 1




def NAND(x, b=0.7):
    array_x = np.array(x)
    array_w = np.array([-0.5,-0.5])
    threshold = np.sum(array_x*array_w)+b
    return 0 if threshold <= 0 else 1



def OR(x, b=-0.2):
    array_x = np.array(x)
    array_w = np.array([0.5, 0.5])   # AND와는 가중치만 다르다.
    threshold = np.sum(array_x*array_w)+b
    return 0 if threshold <= 0 else 1


x = [[0,0],[1,0],[0,1],[1,1]]


def XOR(x):
    temp = []
    for i in x:
        s1 = NAND(i)
        s2 = OR(i)
        temp.append(AND([s1,s2]))
    print(temp)


## 비선형 분류기 ###
XOR(x)
