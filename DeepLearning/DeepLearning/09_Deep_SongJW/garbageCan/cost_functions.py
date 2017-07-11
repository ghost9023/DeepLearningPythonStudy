import numpy as np

'''
cost function : 학습에서 학습 데이터에 대한 오차를 측정하는 척도.
실제값과 신경망에 의한 출력값간의 오차를 측정한다.
'''

def meanSquaredError(y, t):
    '''
    평균제곱오차
    :param y: 신경망의 출력
    :param t: 실제값
    :return: 오차
    '''
    return 0.5 * np.sum((y-t)**2)

def crossEntropyError(y, t):
    '''
    교차엔트로피오차, 신경망의 출력이 0, 1 사이의 값이어야하므로
     sigmoid 함수나 softmax 함수를 채택한 신경망에서 주로 사용한다.
    :param y: 신경망의 출력 (0~1 사이의 값)
    :param t: 실제값
    :return: 오차
    '''
    delta=1e-7  # y 가 0인 경우 -inf 를 막는다.
    return -np.sum(t*np.log(y+delta))

if __name__ == '__cost_functions__' :

    t=np.array([0,0,0,0,0,1,0,0,0,0])
    y_correct=np.array([0.1, 0.03, 0.05, 0.2, 0.03, 0.9, 0.1, 0.2, 0.12, 0.0])
    y_incorrect=np.array([0.1, 0.03, 0.05, 0.2, 0.03, 0.0, 0.1, 0.2, 0.12, 0.9])



    print('\n----- meanSquaredError -----')
    print('정답인 경우', np.sum(y_correct))
    print('MSE : ', meanSquaredError(y_correct, t))
    print('오류인 경우', np.sum(y_incorrect))
    print('MSE : ', meanSquaredError(y_incorrect, t))



    print('\n----- crossEntropyError -----')
    print('정답인 경우', np.sum(y_correct))
    print('CEE : ', crossEntropyError(y_correct, t))
    print('오류인 경우', np.sum(y_incorrect))
    print('CEE : ', crossEntropyError(y_incorrect, t))