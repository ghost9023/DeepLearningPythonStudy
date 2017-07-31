import matplotlib.pyplot as plt
from practice.kingyonghyun.cifar_network_module import *

train_file_list = ['c:/data/train/train_data_' + str(i) + '.csv' for i in range(1, 51)]
test_file_list = ['c:/data/test/test_data_' + str(i) + '.csv' for i in range(1, 11)]

def data_setting(data):
    # x : 데이터, y : 라벨
    x = (np.array(data[:, 0:-1]) / 255).tolist()
    y_tmp = np.zeros([len(data), 10])
    for i in range(0, len(data)):
        label = int(data[i][-1])
        y_tmp[i, label - 1] = 1
    y = y_tmp.tolist()

    return x, y

def read_data(filename):
    ####################################################################################################################
    ## ▣ Data Loading
    ##  - 각각의 파일에 대해 load 후 전처리를 수행
    ####################################################################################################################
    data = np.loadtxt(filename, delimiter=',')
    np.random.shuffle(data)
    return data_setting(data)

# 단층 : 입력 -> softmax with loss 10 -> 출력
# nn_structure = (x_train.shape[1], 'SoftmaxWithLoss', 10)
# 2층 : 입력 -> ReLU 50 -> softmax with loss 10 -> 출력
nn_structure = (1024, 'ReLU', 50, 'SoftmaxWithLoss', 10)
# 3층 : 입력 -> ReLU 50 -> ReLU 50 -> softmax with loss 10 -> 출력
# nn_structure = (x_train.shape[1], 'ReLU', 50, 'ReLU', 50, 'SoftmaxWithLoss', 10)

network = NeuralNetwork(nn_structure=nn_structure, lr=.0008)

optimize_method = 'Adam' # SGD, Momentum, AdaGrad
iteration = 5
batch_size = 100

train_loss_list = []  # 매 배치마다 cost 값을 저장하는 리스트 변수
train_acc_list = []  # 매 epoch 마다 train accuracy 를 저장하는 리스트 변수
test_acc_list = []  # 매 epoch 마다 test accuracy 를 저장하는 리스트 변수

# 학습 시작
print('Learning Started!')

for epoch in range(iteration):
    tot_train_acc = []
    for index in range(0, len(train_file_list)):
        total_x, total_y = read_data(train_file_list[index])
        for start_idx in range(0, 1000, batch_size):
            train_x_batch, train_y_batch = np.array(total_x[start_idx:start_idx + batch_size]), np.array(total_y[start_idx:start_idx + batch_size])  # 배치 단위로 data load

            network.gradient_descent(train_x_batch, train_y_batch, optimize_method)

            loss = network.temp_loss  # 변경된 Weight, Bias 을 가지고 loss 구함
            train_loss_list.append(loss)  # 매 batch 단위 수행시마다 loss 값을 저장

            train_acc = network.accuracy(train_x_batch, train_y_batch)  # 배치 단위 train 데이터에 대해 정확도를 구함
            tot_train_acc.append(train_acc)  # 각 배치 단위마다 구한 정확도를 저장
    print('epoch - {} :'.format(epoch), np.mean(tot_train_acc))
    train_acc_list.append(np.mean(tot_train_acc))  # 매 epoch 마다 구한 train 데이터의 정확도를 저장

# 테스트 시작
print('Testing Started!')

tot_test_acc = []
for index in range(0, len(test_file_list)):
    total_x, total_y = read_data(test_file_list[index])
    for start_idx in range(0, 1000, batch_size):
        test_x_batch, test_y_batch = np.array(total_x[start_idx:start_idx + batch_size]), np.array(total_y[start_idx:start_idx + batch_size])

        test_acc = network.accuracy(test_x_batch, test_y_batch)  # 배치 단위 test 데이터에 대해 정확도를 구함
        tot_test_acc.append(test_acc)  # 각 배치 단위마다 구한 정확도를 저장
test_acc_list.append(np.mean(tot_test_acc))  # 전체 test 데이터의 정확도를 저장

print('train accuracy :', train_acc_list)
print('test accuracy :', test_acc_list)
