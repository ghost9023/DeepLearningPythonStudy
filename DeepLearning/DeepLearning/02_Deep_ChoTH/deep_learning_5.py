## CHAPTER 5 오차역전파법
# 앞에서는 신경망의 가중치 매개변수의 기울기는 수치미분을 사용해 구했다.
# 수치미분은 단순하고 구현하기 쉽지만 계산시간이 오래 걸린다는 단점이 있다.
# 이번장은 가중치 매개변수의 기울기를 효율적으로 계산하는 '오차역전파법'을 배워보자

# 오차역전파법을 제대로 이해하는 방법은 두 가지가 있다.
# 1. 수식을 통해 - 정확하고 간결함
# 2. 계산 그래프를 통해 - 시각적으로!! 본질에 다가가기 쉬움

# 계산그래프
# 노드와 엣지로 표현된다.
# 계산그래프로 풀다.
# 1. 계산그래프를 구성한다.
# 2. 그래프에서 계산을 왼쪽에서 오른쪽으로 진행한다.
# 계산을 왼쪽에서 오른쪽으로 진행하는 것을 순전파라고 한다. 반대로는 역전파

# 계산그래프의 특징은 '국소적 계산'을 전파함으로써 최종 결과를 얻는다는 점에 있다.
# 계산 그래프는 국소적 계산에 집중한다.

# 왜 계산 그래프로 푸는가?
# 계산 그래프를 사용하는 가장 큰 이유는 역전파를 통해 '미분'을 효율적으로 계산할 수 있다는 점때문이다.
# 가령 사과의 가격이 오르면(매개변수) 최종금액(손실함수)에 어떻게 영향을 끼치는지를 알고 싶다고 한다면 이는 사과가격에 대한 지불금액의 미분을 구하는 문제에 해당한다.
# 이 결과로부터 사과가격에 대한 지불금액의 미분값은 2.2라고 할 수 있다. 사과가 1원 오르면 최종금액은 2.2원 오른다는 뜻이다.
# 정확히는 사과값이 아주 조금 오르면 최종금액은 그 아주 작은 값의 2.2배만큼 오른다는 뜻(미분의 개념, 순간기울기)
# 소비세에 대한 지불금액의 미분이나 사과개수에 대한 지불금액의 미분도 같은 순서로 구할수 있다.
# 그리고 그때는 중간까지 구한 미분결과를 공유할 수 있어서 다수의 미분을 효율적으로 계산할 수 있다.

# 연쇄법칙
# 역전파는 '국소적인 미분'을 순방향과는 반대인 오른쪽에서 왼쪽으로 전달한다. 또한 이런 국소적 미분을 전달하는 원리는 연쇄법칙에 따른 것이다.
# 이번 절에서는 연쇄법칙을 설명하고 그것이 계산그래프상의 역전파와 같다는 사실을 밝히겠다.
# 계산그래프의 역전파
# 역전파의 계산절차는 신호 E와 노드의 국소적미분을 곱한 후 다음 노드로 전달하는 것이다.
# 국소적 미분은 순전파 때의 y = f(x) 계산의 미분을 구한다는 뜻이다.
# 가령 y = f(x) = x^2이라면 ∂y/∂x=2x가 된다. 그리고 이 국소적인 미분을 상류에서 전달된 값에 곱해 앞쪽노드로 전달하는 것이다.
# 연쇄법칙이란?
# 합성함수@@@란 여러함수로 구성된 함수이다. 예를 들어서 (z=t^2 / t = x+y)와 같은 함수를 말한다.
# 합성함수의 미분은 합성함수를 구성하는 각 함수의 미분의 곱으로 나타낼 수 있다.
# ∂z/∂x(x에 대한 z의 미분)은 ∂z/∂t(t에 대한 z의 미분)과 ∂t/∂x(x에 대한 t의 미분)의 곱으로 나타낼 수 있다.
# ∂z/∂x = (∂z/∂t) * (∂t/∂x)
# ∂z/∂t는 2t이고, ∂t/∂x는 1이다.
# ∂z/∂x = (∂z/∂t)*(∂t/∂x) = 2t*1 = 2*(x+y)
# 역전파의 계산절차에서는 노드로 들어온 입력신호에 그 노드의 국소적 미분(편미분)을 곱한 후 다음 노드로 전달한다.
# 역전파 때는 상류에서 전해진 미분(이 예에서는 ∂z/∂x)에 1을 곱하여 하류로 흘립니다.
# 이 예에서는 상류에서 전해진 미분값을 ∂L/∂z 이라 했는데, 같이 최종적으로 L이라는 값을 출력하는 큰 계산 그래프 가정하기 때문이다.

# 덧셈노드의 역전파
# ∂z/∂x = 1  /  ∂z/∂y = 1
# 변수 z와 나머지 변수를 대상으로 2차원 그래프를 그리면 기울기가 1이고 y절편이 나머지 하나의 변수의 수에 결정되는 1차함수가 그려진다.
# 따라서, x값에 상관없이 모든 x값에 대한 미분값은 1이 된다.
# 덧셈노드 역전파는 입력신호를 받아서 1(미분값)을 곱한다음 다음 노드로 출력할뿐이므로 그냥 그대로 출력하면 된다.

# 곱셈노드의 역전파
# z = x*y를 생각해보자!!!
# 곱셈노드의 역전팓는 상류의 값에 순전파 때의 입력신호들을 '서로 바꾼값'을 곱해서 하류로 보낸다.
# 덧셈노드와 마찬가지로 변수z와 나머지 하나의 변수를 대상으로 2차원 그래프를 그려보면 y의 값에 따라 기울기가 정해지는 1차 함수가 만들어진다.
# 즉 곱셈노드에서는 x의 값에 관계없이 언제나 기울기(미분)가 y의 값으로 정해지게 된다.
# 곱셈노드의 역전파에서는 입력신호를 서로 바꿔서 하류로 흘린다.
# 결과를 보면 사과 가격의 미분은 2.2, 사과개수의 미분은 110 이 된다.

# 곱셈계층
# 이제부터 모든 계층을 forward()-순전파 와 baxkward()-역전파라고 한다.
class MulLayer:   # 곱셈노드@!@!#!@#!@#!@#!@#!@#!@#
    def __init__(self):
        self.x = None
        self.y = None
    def forward(self, x, y):
        self.x = x
        self.y = y
        out = x * y    # 순전파는 입력 두개 받아서 곱해서 전해주고
        return out
    def backward(self, dout):
        dx = dout * self.y  # x와 y를 바꾼다.
        dy = dout * self.x  # x와 y를 바꾼다.
        return dx, dy # 역전파는 서로의 값을 바꿔서 들어온 입력값을 곱해서 전해준다. (입력 1개, 출력 2개)
#  MulLayer를 사용해서 순전파를 다음과 같이 구현할 수 있다.
apple = 100
apple_num = 2
tax = 1.1

mul_apple_layer = MulLayer()
mul_tax_layer = MulLayer()
apple_price = mul_apple_layer.forward(apple, apple_num)
price = mul_tax_layer.forward(apple_price, tax)
print(price)   # 220.00000000000003

# 역전파 구현
dprice = 1  # 처음에 들어오는 손실함수의 값이라고 볼 수있다.
dapple_price, dtax = mul_tax_layer.backward(dprice)          # self.x에 apple_num이 할당  /  self.y에 tax가 할당
dapple, dapple_num = mul_apple_layer.backward(dapple_price)  # self.x에 apple이 할당  /  self.y에 apple_num가 할당
print(dapple, dapple_num, dtax)   # 2.2 110.00000000000001 200


## 덧셈 계층
class AddLayer:   # 덧셈노드!#!@#!@#!@#!@#!@#!@#!@#!@#
    def __init__(self):
        pass
    def forward(self, x, y):
        out = x + y
        return out
    def backward(self, dout):
        dx = dout * 1
        dy = dout * 1
        return dx, dy
# 인스턴스 변수를 선언하지 않기 때문에 __init__(): 에서 pass를 해준다.

# p.163의 그래프를 파이썬으로 구현한 함수
apple = 100
apple_num = 2
orange = 150
orange_num = 3
tax = 1.1

# 계층들
mul_apple_layer = MulLayer()
mul_orange_layer = MulLayer()
add_apple_orange_layer = AddLayer()
mul_tax_layer = MulLayer()

# 순전파
apple_price = mul_apple_layer.forward(apple, apple_num)
orange_price = mul_orange_layer.forward(orange, orange_num)
all_price = add_apple_orange_layer.forward(apple_price, orange_price)
price = mul_tax_layer.forward(all_price, tax)
print(price)   # 715.0000000000001

# 역전파
dprice = 1
dall_price, dtax = mul_tax_layer.backward(dprice)
dapple_price, dorange_price = add_apple_orange_layer.backward(dall_price)
dapple, dapple_num = mul_apple_layer.backward(dapple_price)
dorange, dorange_num = mul_orange_layer.backward(dorange_price)
print(apple, apple_num, orange, orange_num, dtax)   # 100 2 150 3 650


## 활성화 함수 계층 구현하기
# 우선 활성화함수인 relu와 sigmoid계층을 구현한다.

# relu 계층
# 활성화함수로 사용되는 relu의 수식
# y = x (x > 0)
# y = 0 (x <= 0)
# ∂y/∂x = 1 (x > 0), x가 0보다 클때 x의 값에 관계없이 기울기는 항상 1이다.
# ∂y/∂x = 0 (x <= 0), x가 0보다 작으면 x의 값에 관계없이 기울기는 항상 0(가로 직선)이다.
# 순전파 때 입력인 x가 0보다 크면 역전파는 상류의 값을 그대로 하류로 흘리고,
# 순전파 때 입력인 x가 0보다 작으면 역전파는 하류로 신호를 보내지 않는다.
# relu계층 파이썬 구현
# 신경망 계층의 relu계층은 넘파이배열을 인수로 받는다고 가정한다!!!! 넘파이배열!@@!@#!@!@#
class Relu:
    def __init__(self):
        self.mask = None
    def forward(self, x):
        self.mask = (x <= 0)   # booleah의 배열형태로 나온다. 여기서 주의할 점은 0보다 작은 원소들을 True로 빼준다는 점이다.
        out = x.copy()         # 원본 데이터의 변경을 막기 위해 다른 객체를 만든다.
        out[self.mask] = 0     # 0보다 작아서 True로 나온 원소들을 0으로 바꾼다.
        return out
    def backward(self, dout):  # 역전파는 결국 미분을 한다는 뜻
        dout[self.mask] = 0    # dout배열에서 self.mask에 의해 True가 된 애들은 0으로 만들어주고 아닌(False) 원소는 그대로 둔다.
        dx = dout
        return dx
        # 역전파일 때 입력값이 어떻게 들어오는지 모르겠지만 1이 아닌 경우 그냥 그대로 출력하는데 양수면 무조건 1로 출력해야하는거 아닌가요?
        # 이해가 안감

x = np.array([[1.0,-0.5], [-2.0, 3.0]])
print(x)
mask = (x <= 0)
print(mask)

# sigmoid 계층
# y = 1 / (1+exp(-x))

























