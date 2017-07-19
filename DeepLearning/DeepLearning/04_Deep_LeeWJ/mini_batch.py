import sys, os
sys.path.append(os.pardir)
import numpy as np
from mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
#훈련데이터 ,
print(x_train.shape)  #(60000, 784) --> 훈련데이터는 60,000개이며 입력데이터는 784개이다.
print(t_train.shape)  #(60000, 10)  ---> 훈련데이터 60,000개이며 출력데이터는 10개이다.


'''
이 훈련에서 무작위로 10장만 빼내려면 , 즉 mini-batch = 10 하려면 어떻게 해야할까?
--> np.random.choice() 사용
'''

train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)  #60,000개 중, 무작위로 10개를 골라라.
                                                       #np.array (ndarray타입)
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]



def cross_entropy_error(y,t):
    if y.ndim == 1:  #1차원 배열이라면
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)

    batch_size = y.shape[0]   #아마 60,000개
    # result = -np.sum( t * np.log(y)) / batch_size
    result = -np.sum( np.log(y[np.arange(batch_size), t])) / batch_size
    '''
    이 구현에서는 원-핫 인코딩일때 t가 0인 원소는 CEE 도 0이므로, 그 계산은 무시해도 좋다는 것이 핵심.
    따라서, 정답에 해당하는 신경망의 출력만으로 교차 엔트로피 오차를 계산 할 수 있다.
    따라서 t * np.log(y) 를 위처럼 수정한 것.
    
    
    1) np.arange(batch_size)
        0 ~ batch_size -1 까지 배열을 생성! 
        ex) batch_size = 5---> array([0,1,2,3,4]) 생성
        
        
    '''
    print(result)
    return result


