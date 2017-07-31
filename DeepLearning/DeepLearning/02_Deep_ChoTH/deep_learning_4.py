# CHAPTER 4 신경망 학습
# 데이터 주도학습
# 특징을 추출하고 그 특징의 패턴을 기계학습으로 학습하는 방법
# 신경망은 이미지를 있는 그대로 학습한다. 즉, 특징을 사람이 설계하지 않는다.
# 신경망(딥러닝)은 종단간(end-to-end) 기계학습이라고 한다. 여기서 종단간은 처음부터 끝까지라는 의미이다.
# 신경망의 이점은 모든 문제를 같은 맥락에서 풀 수 있다는 점에 있다.
# 예를 들어 숫자를 인식하든, 개를 인식하든, 사람 얼굴을 인식하든 세부사항에 관계없이 주어진 데이터를 온전히 학습하고,
# 주어진 문제의 패턴을 발견하려고 시도한다.

# 훈련데이터와 시험데이터
# 기계학습 문제느느 데이터를 훈련데이터와 시험데이터로 나눠 학습과 실험을 수행하는 것이 일반적이다.
# 우리가 원하는 것은 범용적으로 사용할 수 있는 모델을 만드는 것이기 때문에 과적합(오버피팅)을 확인하기 위해 시험데이터가 필요하다.
# 시험데이터는 훈련데이터에 포함되어 있지 않은 데이터

# 손실함수
# 정답이 아닌 나머지 모두는 t가 0이기 때문에 손실함수에 영향을 주지 않는다.
# 손실함수는 현재의 상태를 측정하는 하나의 지표이다.
# 일반적으로 평균제곱오차와 교차엔트로피오차를 사용한다.
# 평균제곱오차
# E = 1/2 * sum((y-t)**2)
# 여기서 y는 신경망의 출력 값(0.284783), t는 정답레이블(1), k는 차원 수를 나타낸다.
# 예시 y = [0.1,0.05,0.6,0.0,0.05,0.1,0.0,0.0,0.0,0.0]  |  t = [0,0,0,0,0,0,1,0,0,0,0]
# 위의 t같이 정답을 표시하는 것을 원-핫 인코딩이라고 한다.
# 평균제곱오차는 각 원소의 출력값들과 정답레이블들의 차를 제곱한 후 그 총합을 구한다.
# 파이썬으로 구현
import numpy as np
def mean_squared_error(y, t):
    return 0.5 * np.sum((y-t) ** 2)
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])   # 소프트맥스 통과해서 나온 값 y
mean_squared_error(y, t)   # 결과값이 맞을때

t = np.array([0, 0, 0, 0, 0, 1, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
mean_squared_error(y, t)   # 결과값이 틀릴때

# 교차엔트로피오차
# E = -sum(t*np.log(y))
# 여기에서 log는 밑이 e인 자연로그
# 0에 가까울 수록 정답에 가깝다.
# 교차엔트로피함수 구현
def cross_entropy_error(y,t):
    delta = 1e-7
    return -np.sum(t * np.log(y+delta))   # 각 원소끼리 계산해서 합
# 여기에서 y와 t는 넘파이 배열이다. 왜냐하면 np.log()함수에 0이 들어가면(혹은 컴퓨터 특성상 0에 가까운값) 무한대가 되기 때문이다.
# 즉, 무한대가 발생하지 않도록 delta를 삽입
t = np.array([0, 0, 1, 0, 0, 0, 0, 0, 0, 0])
y = np.array([0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0])
cross_entropy_error(y,t)

# 미니배치학습(손실함수)
# 기계학습은 훈련데이터에 대한 손실함수의 값을 구하고 그 값을 최대한 줄여주는 매개변수를 찾아내는 것이다.
# 이렇게 하려면 모든 데이터를 대상으로 손실함수의 값을 구해야 한다.
# 하나씩 처리하면 오래 걸리기 때문에 이럴 경우 훈련데이터로부터 일부만 골라 학습을 진행한다. (표본추출)
# 여기서 이 일부를 mini_batch라고 한다.
# 아래는 무작위로 지정한 수의 데이터를 골라내는 식
import sys, os
sys.path.append(os.pardir)   # 부모 디렉토리의 파일을 가져올 수 있도록 설정
import numpy as np
from dataset.mnist import load_mnist
from PIL import Image
import pickle

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True, normalize=False, one_hot_label=True)
print(x_train.shape)  # (60000, 784)
print(t_train.shape)  # (60000,)
print(x_test.shape)   # (10000, 784)
print(t_test.shape)   # (10000,)
train_size = x_train.shape[0]   # 60000개
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # 60000개 중에 10개 무작위로 선택, 나중에 인덱스로 사용
# array([8013, 14666, 58452, 4582, 7895, 35245....])
x_batch = x_train[batch_mask]   # 이건 왜 따로 뽑는거야?
t_batch = t_train[batch_mask]   # 인덱스로 사용해서 10개 선택하고 그것과 비교

def cross_entropy_error(y,t):
    if y.dim == 1:
        t = t.reshape(1, t.size)   # 2차원으로 바꿔주기
        y = y.reshape(1, y.size)   # 2차원으로 바꿔주기
    batch_size = y.shape[0]        # 들어온 행의 수대로 배치사이즈 할당
    return -np.sum(t * np.log(y)) / batch_size   # 다 더해서 배치사이즈로 나눠주기, 즉 평균 구하는 것