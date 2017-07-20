import numpy as np
from book.common.functions import softmax, cross_entropy_error

'''
역전파 backpropagation
미분을 구하는 방법. 수치미분보다 빠르게 미분을 구할 수 있다.
연쇄법칙을 이용하는 방법으로 연산을 우에서 좌로 거슬러 올라가며
연산 전 입력에 대한 연산 후 출력의 미분을 구해서 각 연산을 지날때마다 곱해나가는 방식
'''

# 곱셈 계층 - 곱셈 노드
class MulLayer:
    def __init__(self):
        self.x = None
        self.y = None

    def forward(self, x, y):
        '''
        순전파. 두 수의 곱을 출력하는데 역전파를 위해 x, y 값을 보관한다.
        :param x: 곱셈의 피연산자1 : float
        :param y: 곱셈의 피연산자2 : float
        :return: x*y : float
        '''
        self.x = x
        self.y = y
        out = x * y

        return out

    def backward(self, dout):
        '''
        역전파. x, y 에 대한 미분은 역전파된 미분값에 y, x 를 곱한 값이다.
        :param dout: 상류에서 역전파된 미분값 : float
        :return: dout/dx, dout/dy : float, float
        '''
        dx = dout * self.y
        dy = dout * self.x

        return dx, dy

class AddLayer:
    def __init__(self):
        pass

    def forward(self, x, y):
        return x + y

    def backward(self, dout):
        return dout, dout

class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        '''
        ReLU 함수 순전파.
        numpy 벡터 x 에서 0과 같거나 0보다 작은 요소를 0으로 바꿔서 출력. 
        :param x: numpy vector
        :return: numpy vector
        '''
        self.mask = (x <= 0)
        out = x.copy()
        out[self.mask] = 0

        return out

    def backward(self, dout):
        dout[self.mask] = 0
        dx = dout

        return dx

class Sigmoid:
    '''
    sigmoid 함수의 입력을 x, 출력을 y 라 할때 연산 단계를 거슬러올라가며 미분을 역전파하면
    입력 x 에 대한 출력 y 의 미분은 y*(1-y) 로 정리할 수 있다. 즉, 출력값만으로 미분의 계산이 가능해진다.
    '''
    def __init__(self):
        self.out = None

    def forward(self, x):
        out = 1 / (1 + np.exp(-x))
        self.out = out
        return out

    def backward(self, dout):
        dx = dout * self.out * (1 - self.out)
        return dx

class Affine:
    '''
    신경망의 순전파에서 행해지는 내적을 기하학에서는 어파인 변환 affine transformation 이라고 한다.
    '''
    def __init__(self, W, b):
        '''
        입력에 가중치를 곱하고 편향을 더하는 affine 변환 전체를 정의한 클래스이므로
         가중치 W, 편향 b 를 포함하고, 이 값들을 갱신하기 위해 dW, db 값을 갖는다.
         미분값을 역전파하기위해서 입력 x 를 보관한다.
         :param W: numpy matrix : 가중치
         :param b: numpy array : 편향
        '''
        self.W = W
        self.b = b
        self.x = None
        self.db = None
        self.dW = None

    def forward(self, x):
        '''
        순전파
        :param x: numpy matrix : 입력
        :return: numpy matrix : 입력*가중치+편향
        '''
        self.x = x
        out = np.dot(x, self.W) + self.b
        return out

    def backward(self, dout):
        '''
        역전파. x 에 대한 미분만 반환하고 W, b 에 대한 미분은 보관
        :param dout: ? :역전파받은 미분값
        :return: numpy matrix : x 에 대한 미분
        '''
        dx = np.dot(dout, self.W.T)
        self.dW = np.dot(self.x.T, dout)
        self.db = np.sum(dout, axis = 0)
        return dx

class SoftmaxWithLoss:
    '''
    softmax 함수와 cross entropy error 를 결합한 레이어
    '''
    def __init__(self):
        self.loss = None
        self.y = None
        self.t = None

    def forward(self, x, t):
        '''
        순전파
        :param x: numpy matrix : 입력 
        :param t: numpy matrix : 레이블
        :return: float : loss
        '''
        self.t = t
        self.y = softmax(x)
        self.loss = cross_entropy_error(self.y, self.t)
        return self.loss

    def backward(self, dout = 1):
        '''
        역전파. 역전파시에는 미분값을 배치 크기로 나눠서 데이터 한개당 오차를 전달한다.
        :param dout: float : default 1
        :return: numpy matrix : 입력에 대한 미분 
        '''
        batch_size = self.t.shape[0]
        dx = (self.y -self.t) / batch_size
        return dx


if __name__ == '__main__':
    # # a = apple_example()
    # # print(a.forward_prop())
    # # a.backward_prop()
    #
    # # 예제 : 개당 100원 사과 2개를 사는데 세금은 10 % 가 붙는다면 최종 가격은?
    #
    # print('예제 : 개당 100원 사과 2개를 사는데 세금은 10 % 가 붙는다면 최종 가격은?')
    #     # 순전파
    # apple = 100
    # apple_num = 2
    # tax = 1.1
    #
    # mul_apple_layer = MulLayer()
    # mul_tax_layer = MulLayer()
    #
    # apple_price = mul_apple_layer.forward(apple, apple_num)
    # price = mul_tax_layer.forward(apple_price, tax)
    #
    # print('\t100원 사과 2개의 가격 (세금 10%) :',price)    # 각 레이어를 통과하며 값을 전달하여 최종 가격을 출력한다.
    #
    #     # 역전파
    # dprice = 1
    # dapple_price, dtax = mul_tax_layer.backward(dprice)
    # dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    #
    # print('\t사과 가격에 대한 미분 :',dapple,'\n\t사과 개수에 대한 미분 :',dapple_num,'\n\t부가세에 대한 미분 :',dtax)
    #
    # # 예제 : 100원 사과 2개, 150원 귤 3개를 사는데 세금이 10 % 붙는다면 최종 가격은?
    #
    # print('\n예제 : 100원 사과 2개, 150원 귤 3개를 사는데 세금이 10 % 붙는다면 최종 가격은?')
    # apple = 100
    # orange = 150
    # apple_num = 2
    # orange_num = 3
    # tax = 1.1
    #
    # mul_apple_layer = MulLayer()
    # mul_orange_layer = MulLayer()
    # add_apple_orange_layer = AddLayer()
    # mul_tax_layer = MulLayer()
    #
    # # 순전파
    # apple_price = mul_apple_layer.forward(apple, apple_num)
    # orange_price = mul_orange_layer.forward(orange, orange_num)
    # all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    # price = mul_tax_layer.forward(all_price, tax)
    # print('\t100원 사과 2개, 150원 오렌지 2개를 사는데 세금 10 % 가 붙는다면 최종 가격은? :', price)
    #
    # # 역전파
    # dprice = 1
    # dall_price, dtax = mul_tax_layer.backward(dprice)
    # dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    # dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    # dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    # print('\t사과 가격에 대한 미분 :', dapple, '\n\t사과 개수에 대한 미분 :', dapple_num)
    # print('\t귤 가격에 대한 미분 :', dorange, '\n\t귤 개수에 대한 미분 :', dorange_num)
    # print('\t부가세에 대한 미분 :', dtax)

    '''
    순전파의 역순으로 역전파를 한다
    '''