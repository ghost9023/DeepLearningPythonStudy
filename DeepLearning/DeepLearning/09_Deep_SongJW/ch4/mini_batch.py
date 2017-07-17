'''
미니배치 학습

N 건의 데이터의 E = (N건의 데이터 각각의 오차의 총합) / N
데이터의 수에 상관없이 통일된 지표로서 평균손실함수를 구하게 된다.
신경망을 학습시킬때 오차를 구하기 위해서 각 데이터에 대한 신경망의 출력과 실제 라벨을 비교해 오차를 구하고
이 오차를 모두 더해서 신경망의 정확성의 지표로 삼는다. 그런데 데이터의 수가 늘어날수록 이 계산의 시간이 늘어나기때문에
데이터의 일부를 전체 데이터의 근사치로 삼아 데이터 일부만을 학습시키는 방법을 미니배치 학습이라고 한다.

데이터 일부 = 미니배치
미니배치를 통한 학습 = 미니배치 학습
'''

import sys, os
sys.path.append(os.pardir)
import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)

print(x_train.shape)    # (60000, 784)
print(t_train.shape)    # (60000, 10)

# hyperparameter -  프로그래머가 경험이 기반해 직접 정하는 변수들
train_size = x_train.shape[0]   # 60000
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)   # train_size 의 범위에서 (0~59999) 에서 무작위로 batch_size 만큼 픽
x_batch = x_train[batch_mask]   # batch_mask 에 속하는 인덱스를 가지는 데이터만 선택함
t_batch = t_train[batch_mask]

print(batch_mask)

