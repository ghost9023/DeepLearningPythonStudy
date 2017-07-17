import numpy as np

'''
4.2 손실함수
비용함수 cost function or 손실함수 loss functions
 신경망에서 학습이 제대로 이루어졌는지 평가하는 하나의 지표.
 비용함수의 값을 최소로 하는 가중치를 찾는것이 학습의 목표.
    1. 평균제곱오차 mean squared error - MSE
    2. 교차 엔트로피 오차 cross entropy error - CEE 
'''

t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])  # 답 : 2
y1 = np.array([.1, .05, .6, .0, .05, .1, .0, .1, .0, .0])    # 예측 : 2 (확률 0.6)
y2 = np.array([.1, .05, .1, .0, .05, .1, .0, .6, .0, .0])    # 예측 : 7 (확률 0.6)

#####
# 4.2.1 평균 제곱 오차 MSE
# E = (1/2) * (sum( (y - e)**2 ))
#   (y = 신경망의 예측값, t = 실제라벨, y, t 모두 one-hot-encoding)
#
# ex) mnist
# y = [ 0.1,    0.05,   0.6,    0.0,    0.05,   0.1,    0.0,    0.1,    0.0,    0.0 ]
# t = [ 0,      0,      1,      0,      0,      0,      0,      0,      0,      0   ]
# y 는 신경망이 예측한 입력데이터가 0~9 까지 각 숫자일 확률
# t 는 실제 입력데이터의 라벨
# MSE 는 대응하는 y, t 값의 차를 제곱하여 모두 더한 후 2로 나눈 값
#####

def MeanSquaredError(y, t):
    return np.sum((y - t) ** 2)/2

# print(MeanSquaredError(y1, t))   # 정답 예측 - 0.0975
# print(MeanSquaredError(y2, t))   # 오답 예측 - 0.5975

    # 잘못된 예측을 하는 경우 MSE 값이 커지게된다.
    # 정확한 값을 예측할뿐만 아니라 더 높은 확률을 부여할수록 오차는 작아지게된다.


#####
# 4.2.2 교차 엔트로피 오차 CEE
# E = - sum(t * log(y))
# log 는 자연로그 ln, y 와 t 모두 one-hot-encoding
# 정답인 레이블만 제외하면 t 의 각 요소는 모두 0이 된다.
# 따라서 하나의 입력 데이터에 대해서 E = -log(예측확률), 정답일때의 출력이 전체 값을 정하게 된다.
# 로그 함수는 입력이 0에 가까울수록 급격하게 증가하므로 잘못된 예측을 하거나
# 예측이 맞아도 예측확률이 낮다면 급격하게 에러가 증가한다.
# 이상적이게 예측이 완벽하다면 ( 확률 : 1.0 ) 에러는 0이 된다.
#####

# def CrossEntropyError(y, t):
#     delta = 1e-7    # 로그함수에는 0 이 입력될 수 없음. y 에 0 값이 존재할 수 있으니 작은 값을 넣어줌.
#     return -np.sum(t * np.log(y + delta))

def CrossEntropyError(y, t):    # t 가 one-hot-encoding 일때
    delta = 1e-7
    if y.ndim == 1: # 벡터를 (1, 벡터길이) 의 매트릭스로 변한
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0] # 입력된 미니배치가 몇건의 데이터묶음인지 확인
    return -np.sum(t * np.log(y + delta)) / batch_size  # 미니배치의 평균 교차엔트로피 오차를 구한다.

def CrossEntropyError_label(y, t):
    delta = 1e-7
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t])) / batch_size
        # y[np.arange(batch_size), t] :
        # np.arange(batch_size) 는 0 ~ (y 미니배치의 데이터 수-1) 범위의 수가 담긴 벡터 생성
        # t 는 정답 라벨이 담긴 벡터(one-hot-encoding 이 아니므로 숫자가 나열되있음)
        # y[벡터x, 벡터z] 는 y[(x1,z1), (x2,z2), (x3,z3), ...] 이고 이는
        # y=np.array([y(x1,z1), y(x2,z2), y(x3,z3), ...])
        # 벡터x, 벡터z 의 대응하는 원소들을 인덱스로 갖는 y의 요소들이 나열된 벡터를 얻게된다.
        # 정리하자면 y[np.arange(batch_size), t] 는 t 에 담긴 라벨에 해당하는 신경망의 출력(확률값)이 나열된 벡터

# print(CrossEntropyError(y1, t))
#     # 정답 예측 - 0.510825457099
# print(CrossEntropyError(y2, t))
#     # 오답 예측 - 2.30258409299
#     # 이 값은 오답에 높은 확률을 부여했기때문이 아닌 정답에 낮은 확률을 부여했기 때문에 계산되는 오차.

'''
비용함수, 손실함수를 사용하는 이유
 = 신경망의 분류결과와 실제라벨링을 비교하여 계산한 정확도를 신경망의 정확함을 나타내는 지표로 사용하지 않는 이유?

매개변수를 매우 조금씩 조정할때 정확도 (ex - 100 개의 입력 데이터중 32 개를 정확히 분류했다 = 정확도 : 32 %) 는 변하지 않음.
 매개변수를 매우 조금씩 바꾼다고 신경망의 분류결과가 크게 바뀌지 않으며
 바뀐다고 해도 정확도는 연속적인 값이 아닌 이산적인 값. 33 % -> 34 %.
 계단함수처럼 대부분의 구간에서 미분값이 0 이 된다.
 
손실함수은 매개변수를 매우 조금씩 조정할때마다 수치가 계속 바뀌며 이는 연속적인 실수값을 갖는다.

매개변수의 작은 변화에 따른 신경망의 정확도를 나타내는 지표의 변화를 포착하여 가중치를 개선해야하는데
 정확도는 계단함수처럼 매개변수의 작은 변화에도 반응이 거의 없다시피 하여 신경망의 개선이 힘들다.
 
'''