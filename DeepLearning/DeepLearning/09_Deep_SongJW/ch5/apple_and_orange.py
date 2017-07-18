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

if __name__ == '__main__':
    # a = apple_example()
    # print(a.forward_prop())
    # a.backward_prop()

    # 예제 : 개당 100원 사과 2개를 사는데 세금은 10 % 가 붙는다면 최종 가격은?

    print('예제 : 개당 100원 사과 2개를 사는데 세금은 10 % 가 붙는다면 최종 가격은?')
        # 순전파
    apple = 100
    apple_num = 2
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_tax_layer = MulLayer()

    apple_price = mul_apple_layer.forward(apple, apple_num)
    price = mul_tax_layer.forward(apple_price, tax)

    print('\t100원 사과 2개의 가격 (세금 10%) :',price)    # 각 레이어를 통과하며 값을 전달하여 최종 가격을 출력한다.

        # 역전파
    dprice = 1
    dapple_price, dtax = mul_tax_layer.backward(dprice)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)

    print('\t사과 가격에 대한 미분 :',dapple,'\n\t사과 개수에 대한 미분 :',dapple_num,'\n\t부가세에 대한 미분 :',dtax)

    # 예제 : 100원 사과 2개, 150원 귤 3개를 사는데 세금이 10 % 붙는다면 최종 가격은?

    print('\n예제 : 100원 사과 2개, 150원 귤 3개를 사는데 세금이 10 % 붙는다면 최종 가격은?')
    apple = 100
    orange = 150
    apple_num = 2
    orange_num = 3
    tax = 1.1

    mul_apple_layer = MulLayer()
    mul_orange_layer = MulLayer()
    add_apple_orange_layer = AddLayer()
    mul_tax_layer = MulLayer()

    # 순전파
    apple_price = mul_apple_layer.forward(apple, apple_num)
    orange_price = mul_orange_layer.forward(orange, orange_num)
    all_price = add_apple_orange_layer.forward(apple_price, orange_price)
    price = mul_tax_layer.forward(all_price, tax)
    print('\t100원 사과 2개, 150원 오렌지 2개를 사는데 세금 10 % 가 붙는다면 최종 가격은? :', price)

    # 역전파
    dprice = 1
    dall_price, dtax = mul_tax_layer.backward(dprice)
    dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
    dapple, dapple_num = mul_apple_layer.backward(dapple_price)
    dorange, dorange_num = mul_orange_layer.backward(dorange_price)
    print('\t사과 가격에 대한 미분 :', dapple, '\n\t사과 개수에 대한 미분 :', dapple_num)
    print('\t귤 가격에 대한 미분 :', dorange, '\n\t귤 개수에 대한 미분 :', dorange_num)
    print('\t부가세에 대한 미분 :', dtax)

'''
순전파의 역순으로 역전파를 한다
'''
