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

# 블록으로 생각하기
# 3차원의 합성곱 연산은 데이터와 필터를 직육면체 블록이라고 생각하면 쉽다.
# 3차원 데이터를 다차원 배열로 나타낼 때는 (채널, 높이, 너비) 순서로 쓴다.
# 채널:C, 높이:H, 너비:W // 필터채널:C, 필터높이:FH, 필터너비:FW
# 합성곱에서 출력되는 데이터는 한장의 특징맵이다.
# 그렇다면 합성곱 연산의 출력으로 다수의 채널을 내보내려면 어떻게 해야할까?
# 그 답은 필터(가중치)를 다수 사용하는 것이다.
# 필터를 FN개 적용하면 출력맵고 FN개가 된다. 그리고 FN개의 맵을 모으면 형상이 (FN, OH, OW)인 블록이 완성된다.
# (이 완성된 블록을 다음 계층으로 넘기겠다는 것이 CNN의 처리흐름이다.)
# 위의 예처럼 합성곱 연산에서는 필터의 수도 고려해야한다. 필터의 가중치 데이터는 4차원데이터이며
# (출력채널수, 입력채널수, 높이, 너비) 순으로 쓴다. p.238 참조!























###########################
# im2col로 데이터 전개하기
###########################
import numpy as np
def im2col(input_data, filter_h, filter_w, stride=1, pad=0):
    """다수의 이미지를 입력받아 2차원 배열로 변환한다(평탄화).

    Parameters
    ----------
    input_data : 4차원 배열 형태의 입력 데이터(이미지 수, 채널 수, 높이, 너비)
    filter_h : 필터의 높이
    filter_w : 필터의 너비
    stride : 스트라이드
    pad : 패딩

    Returns
    -------
    col : 2차원 배열
    """
    N, C, H, W = input_data.shape
    out_h = (H + 2 * pad - filter_h) // stride + 1
    out_w = (W + 2 * pad - filter_w) // stride + 1

    img = np.pad(input_data, [(0, 0), (0, 0), (pad, pad), (pad, pad)], 'constant')
    col = np.zeros((N, C, filter_h, filter_w, out_h, out_w))

    for y in range(filter_h):
        y_max = y + stride * out_h
        for x in range(filter_w):
            x_max = x + stride * out_w
            col[:, :, y, x, :, :] = img[:, :, y:y_max:stride, x:x_max:stride]

    col = col.transpose(0, 4, 5, 1, 2, 3).reshape(N * out_h * out_w, -1)
    return col
##################################
##################################
##################################
import sys, os
sys.path.append(os.pardir)
from common.util import im2col
x1 = np.random.rand(1,3,7,7)
col1 = im2col(x1, 5, 5, stride=1, pad=0)
print(col1.shape)

x2 = np.random.rand(10, 3, 7, 7)
col2 = im2col(x2, 5, 5, stride=1, pad=0)
print(col2.shape)

####################################################
################## 합성곱계층 구현 ###################
####################################################
class Convolution:    #
    def __init__(self, W, b, stride=1, pad=0):
        self.W = W
        self.b = b
        self.stride = stride
        self.pad = pad
    def forward(self, x):
        FN, C, FH, FW = self.W.shape
        N, C, H, W = x.shape
        out_h = int(1 + (H + 2*self.pad - FH) / self.stride)
        out_w = int(1 + (W + 2*self.pad - FH) / self.stride)
        col = im2col(x, FH, FW, self.stride, self.pad)
        col_W = self.W.reshape(FN, -1).T
        out = np.dot(col, col_W) + self.b
        out = out.reshape(N, out_h, out_w, -1).transpose(0, 3, 1, 2)
        return

##################################################
################## 풀링계층 구현 ###################
##################################################
class Pooling:
    def __init__(self, pool_h, pool_w, stride=1, pad=0):
        self.pool_h = pool_h
        self.pool_w = pool_w
        self.stride = stride
        self.pad = pad
    def forward(self, x):
        N, C, H, W = x.shape
        out_h = int(1 +(H-self.pool_h) / self.stride)
        out_w = int(1 +(W-self.pool_w) / self.stride)
        col = im2col(x, self.pool_h, self.pool_w, self.stride, self.pad)  # 전개 (1)
        col = col.reshape(-1, self.pool_h*self.pool_w)
        out = np.max(col, axis=1)  # 최대값 (2)
        out = out.reshape(N, out_h, out_w, C).transpose(0, 3, 1, 2)  # 성형 (3)
        return out


# 풀링계층 구현은 [그림 7-22]와 같이 다음의 세 단계로 진행합니다.
# 1. 입력데이터를 전개한다.
# 2. 행 별 최대값을 구한다.
# 3. 적절한 모양으로 성형한다.
# 앞의 코드에서와 같이 각 단계는 한 두줄 정도로 간단히 구현됩니다.

# CNN 구현하기
# 합성곱 계층과 풀링계층을 조합하여 손글씨 숫자를 인식하는 CNN을 조립할 수 있다.
# 단순 CNN의 네트워크 구성
# conv -> relu -> pooling -> affine -> relu -> affine -> softmax ->
# 위의 순서로 흐르는 CNN 신경망 구현
# 초기화 때 받는 인수
# input_dim - 입력데이터(채널 수, 높이, 너비)의 차원
# conv_param - 합성곱계층의 하이퍼파라미터(딕셔너리). 딕셔너리의 키는 다음과 같다.
# filter_num - 필터 수
# filter_size - 필터크기
# stride - 스트라이드
# pad - 패딩
# hidden_size - 은닉층(완전연결)의 뉴런수
# output_size - 출력층(완전연결)의 뉴런수
# weight_init_std - 초기화 때의 가중치 표준편차
# 여기서 합성곱 계층의 매개변수는 딕셔너리 형태로 주어진다.(conv_param)
# 예를 들어 {'filter_num':30, 'filter_size':5, 'pad':0, 'stride':1}처럼 저장된다.
class SimpleConvNet:   # CNN 초기화
    def __init__(self, input_dim=(1,28,28), conv_param={'filter_num':30, 'filter_size':5
                                                        'pad':0, 'stride':1},
                 hidden_size=100, output_size=10, weight_init_std=0.01):
        filter_num = conv_param['filter_num']     # 초기화 인수로 주어진 합성곱 계층의 하이퍼파라미터를 딕셔너리에서 꺼낸다.
        filter_size = conv_param['filter_size']
        filter_pad = conv_param['pad']
        filter_stride = conv_param['stride']
        input_size = input_dim[1]
        conv_output_size = (input_size - filter_size + 2 * filter_pad) / filter + 1
        pool_output_size = int(filter_num * (conv_output_size/2) * (conv_output_size/2))   # 합성곱 계층의 출력크기를 계산한다.
        self.params = {}
        self.params['W1'] = weight_init_std * np.random.randn(filter_num, input_dim[0], filter_size, filter_size)
        self.params['b1'] = np.zeros(filter_num)
        self.params['W2'] = weight_init_std * np.random.randn(pool_output_size, hidden_size)
        self.params['b2'] = np.zeros(hidden_size)
        self.params['W3'] = weight_init_std * np.random.randn(hidden_size, output_size)
        self.params['b3'] = np.zeros(output_size)
        self.layers = OrderedDict()   # 순서가 있는 딕셔너리 -> layers에 계층들을 차례대로 추가
        self.layers['Conv1'] = Convolution(self.params['W1'],
                                           self.params['b1'],
                                           conv_param['stride'],
                                           conv_param['pad'])
        self.layers['Relu1'] = Relu()
        self.layers['Pool1'] = Pooling(pool_h=2, pool_w=2, stride=2)
        self.layers['Affine1'] = Affine(self.params['W2'],
                                        self.params['b2'])
        self.layers['Relu2'] = Relu()
        self.layers['Affine2'] = Affine(self.params['W3'],
                                        self.params['b3'])
        self.last_layers = SoftmaxWithLoss()   # 마지막계층은 따로 저장

    def predict(self, x):
        for layer in layers.values():
            x = layer.forward(x)
            return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def gradient(self, x, t):   # 매개변수의 기울기는 오차역전파로 구한다. 이 과정은 순전파와 역전파를 반복한다.
        self.loss(x, t)   # 순전파
        dout = 1   # 역전파
        dout = self.last_layer.backward(dout)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            dout = layer.backward(dout)
        grads = {}
        grads['W1'] = self.layers['Conv1'].dW
        grads['b1'] = self.layers['Conv1'].dW
        grads['W2'] = self.layers['Affine1'].dW
        grads['b2'] = self.layers['Affine1'].dW
        grads['W3'] = self.layers['Affine2'].dW
        grads['b3'] = self.layers['Affine2'].dW
        return grads


# CNN 시각화하기
# CNN을 구성하는 합성곱계층은 입력으로 받은 이미지에서 보고 있는 것이 무엇인지 알아보도록 하자!
# 1번째 층의 가중치 시각화하기
# 1번째 층의 합성곱 계층의 가중치는 (30, 1, 5, 5)이다. - 필터30개, 채널1개, 5X5 크기 - 회색조필터!
# 학습을 마친 필터는 규칙성 있는 이미지가 된다.
# 층 깊이에 따른 추출정보 변화
# 계층이 깊어질 수록 추출되는 정보(정확히는 강하게 반응하는 뉴런)는 더 추상화 된다.
# 층이 깊어지면서 더 복잡하고 추상화된 정보가 추출된다. 처음층은 단순한 에지에 반응하고 이어서 텍스쳐에 반응한다.
# 층이 깊어지면서 뉴런이 반응하는 대상이 단순한 모양에서 '고급'정보로 변화해간다.

# 대표적인 CNN
# LeNet@@@@은 손글씨 숫자를 인식하는 네트워크로 1998년에 제안되었다.
# 합성곱계층과 풀링 계층(정확히는 원소를 줄이기만 하는 서브샘플링)을 반복하고, 마지막으로 완전연결 계층을 거치면서 결과를 출력한다.
# LeNet과 '현재의 CNN'을 비교하면 몇가지 차이가 있다.
# 1. 활성화함수의 차이 - 르넷은 시그모이드, 현재는 렐루
# 2. 르넷은 서브샘플링을 하여 중간 데이터의 크기가 달라지지만 현재는 최대풀링이 주류이다.

# AlexNet은 딥러닝 열풍을 일으키는 데 큰 역할을 했다.
# AlexNet은 합성곱계층과 풀링계층을 거듭하며 마지막으로 완전연결 게층을 거쳐 결과를 출력한다.
# AlexNet은 활성화함수로 렐루를 이용한다.
# LRN이라는 국소적 정규화를 실시하는 계층을 이용한다.
# 드롭아웃을 사용한다.

# 정리
# CNN은 지금까지의 완전연결계층 네트워크와 합성곱 계층과 풀링계층을 새로 추가한다.
# 합성곱계층과 풀링계층은 im2col을 이용하면 간단하고 효율적으로 구현할 수 있다.
# CNN을 시각화해보면 계층이 깊어질 수록 고급정보가 추출되는 모습을 확인할 수 있다.
# 대표적인 CNN 에는 르넷과 알렉스넷이 잇다.
# 딥러닝의 발전에는 빅데이터와 GPU가 공헌햇다.
