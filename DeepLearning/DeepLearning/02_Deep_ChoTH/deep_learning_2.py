# 퍼셉트론이란 다수의 신호를 받아 하나의 신호를 출력한다.
# 입력신호가 노드에 보내지고 각각에 가중치가 곱해진다.
# 거기에 고유한 편향(bias)가 더해져서 정해진 값을 넘어서면 1, 넘지 못하면 0이 출력된다.

# y = {0 (w1*x1 + w2*x2) <= theta}
# y = {1 (w1*x1 + w2*x2) > theta}
# 여기서 b는 편향을 의미하며 w는 가중치


# 단순한 논리회로
# AND, OR, NAND 게이트
# AND, OR, NAND 게이트는 입력이 둘이고 출력은 하나, 입력이 모두 1일때 1을 출력하고 그 외에는 모두 0을 출력
# AND, OR, NAND 게이트를 진리표대로 작동하도록 하는 가중치를 구하자!
# 퍼셉트론 구현하기
def AND(x1, x2):
    w1, w2, theta = 0.5, 0.5, 0.7
    tmp = x1*w1 + x2*w2
    if tmp <= theta:
        return 0
    elif tmp > theta:
        return 1
AND(0, 0)
AND(0, 1)
AND(1, 0)
AND(1, 1)

# 가중치와 편향도입
# y = {0 (b + w1*x1 + w2*x2) <= 0}
# y = {1 (b + w1*x1 + w2*x2) > 0}

# 위의 결과를 넘파이로 구현
import numpy as np
x = np.array([0, 1])
w = np.array([.5, .5])
b = -.7
w * x
np.sum(w*x)      # 각 노드로 들어가는 가중치 다 더해줌
np.sum(w*x) + b  # 편향 더함, -0.19999999999999996

# 가중치와 편향 구현하기
def AND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 여기를 조정
    b = -0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def NAND(x1, x2):
    x = np.array([x1, x2])
    w = np.array([-0.5, -0.5])  # 여기를 조정
    b = 0.7
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

def OR(x1, x2):
    x = np.array([x1, x2])
    w = np.array([0.5, 0.5])  # 여기를 조정
    b = -0.2
    tmp = np.sum(w*x) + b
    if tmp <= 0:
        return 0
    else:
        return 1

# 퍼셉트론의 한계
# XOR을 구현하지 못한다.
# 퍼셉트론은 직선 하나로만 나눈 영역만 표현할 수 있다는 한계가 있다.
# x1 + x2 <= 0.5처럼

# 다층 퍼셉트론을 사용하여 XOR구현하기
# 기존 게이트 조합하기
# (NAND + OR) + AND ==> XOR

def XOR(x1, x2):
    s1 = NAND(x1, x2)
    s2 = OR(x1, x2)
    y = AND(s1, s2)
    return y
XOR(0,0)
XOR(1,0)
XOR(0,1)
XOR(1,1)