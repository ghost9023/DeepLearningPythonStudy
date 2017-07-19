import numpy as np

def mse(y,t):
    result = 0.5 * np.sum((y-t)**2)    #예측값 - 참값
    print(result)
    return result

#출력층 활성함수에 의해 출력된 결과  (소프트맥스)   -----> 예측값
y3 = [0.1, 0.05, 0.6, 0.0, 0.05, 0.1, 0.0, 0.1, 0.0, 0.0]  # 1일 확률이 0.1, 2일 확률이 0.05 등..
#원핫 인코딩   ----->  참값
t3 = [0, 0, 1, 0, 0, 0, 0, 0, 0, 0]  #np.argmax(y3)을 통해서 가장 확률이 높은 값을 표현하는 원핫인코딩

# 분류가 3일 확률이 0.6으로 가장 높은 경우 실제 3라는 결과의 손실 함수 값
mse(np.array(y3), np.array(t3)) #0.0975



# 분류가 7일 확률이 가장 높으나 실제 3이라는 결과의 손실 함수 값

y7 = [0.1, 0.05, 0.1, 0.0, 0.05, 0.1, 0.0, 0.6, 0.0, 0.0]
mse(np.array(y7), np.array(t3)) #0.5975

#코스트가 줄지 않았다..?
#그러니까, 모든 케이스를 봤을 때, 손실함수 값이 가장 작았던 값을 첫 추정결과로 판단하게 되는 근거가 된다.
#그게 바로 MSE! 손실함수! cost !!


def cross_entropy_error(y,t):  # y와 t는 넘파이 배열이다.
    delta = 1e-7
    result = -np.sum(t* np.log(y+delta)) #np.log계산할때 아주 작은 값인 delta를 더했다.
                                          #0을 함수에 입력하면 -inf가 되버리기 때문에, 작은 값을 더해 무한대가 발생하지 않도록 하는 것.
    print(result)
    return result


cross_entropy_error(np.array(y3), np.array(t3))  #0.510825457099
cross_entropy_error(np.array(y7), np.array(t3))  #2.30258409299

