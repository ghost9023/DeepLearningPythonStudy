import numpy as np

def AND(x1, x2) :
    '''
    AND gate
    :param x1: int 0 or 1
    :param x2: "
    :return: int 0 or 1
    '''
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.75
    net=np.sum(w*x)+b
    if net > 0 : return 1
    else : return 0

def OR(x1, x2) :
    '''
    OR gate
    :param x1: int 0 or 1
    :param x2: "
    :return: int 0 or 1
    '''
    x=np.array([x1,x2])
    w=np.array([0.5, 0.5])
    b=-0.25
    net=np.sum(w*x)+b
    if net > 0 : return 1
    else : return 0

def NAND(x1, x2) :
    '''
    NAND gate
    :param x1: int 0|1
    :param x2: int 0|1
    :return: int 0|1
    '''
    x=np.array([x1,x2])
    w=np.array([-0.5, -0.5])
    b=0.75
    net=np.sum(w*x)+b
    if net > 0 : return 1
    else : return 0

def XOR(x1, x2) :
    '''
    XOR gate.
    :param x1: int 0|1 
    :param x2: "
    :return: int 0|1
    '''
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    return AND(s1, s2)

gate_lst=[AND, OR, NAND, XOR]
input_lst=((0,0), (0,1), (1,0), (1,1))

for i in gate_lst :
    for j in input_lst :
        print(i(j[0], j[1]))

########################################################################################################################
'''
단일 퍼셉트론으로는 1차 선형 방정식을 활성함수로 할 수 있는 AND, OR, NAND 정도만 구현 가능. XOR 게이트를 만들기 위해
다층 퍼셉트론 이용
'''
########################################################################################################################

