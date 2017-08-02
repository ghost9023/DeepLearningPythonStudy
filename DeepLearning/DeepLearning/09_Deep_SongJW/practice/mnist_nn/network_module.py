from practice.mnist_nn.layer_module import *
from book.common.layers import BatchNormalization as book_BatchNormalization
from book.common.layers import Dropout

'''
신경망을 구성하는 네트워크 클래스.
'''

class NeuralNetwork:

    # def __init__(self, nn_structure, std_scale_method=.01, lr=.1, gd_method='SGD'):
    #     self.layers = []
    #     self.lr = lr
    #     # 784 - 20 - 20 - 10
    #     # input:100x784 - Affine(W:784x20, b:20) - 'ReLU' -  Affine(W:20x20, b:20) - 'ReLU' - Affine(W:20x10, b:10) - SoftmaxWithLoss
    #     # nn_structure == (784, 'ReLU', 20, 'ReLU', 20, 'SoftmaxWithLoss, 10)
    #     self.std_scale = std_scale_method
    #     for i in range(0, len(nn_structure) // 2):
    #         input_num = nn_structure[2 * i]
    #         output_num = nn_structure[2 * i + 2]
    #         activation = eval(nn_structure[2 * i + 1])
    #         if str(std_scale_method).isalpha():
    #             if std_scale_method == 'Xavier':
    #                 self.std_scale = np.sqrt(1 / input_num)
    #             elif std_scale_method == 'He':
    #                 self.std_scale = np.sqrt(2 / input_num)
    #         self.layers.append(Affine(self.std_scale * np.random.randn(input_num, output_num),
    #                                   np.zeros(output_num), self.lr))
    #         self.layers.append(activation())
    #
    #     self.lastLayer = SoftmaxWithLoss()
    #     self.temp_loss = None

    def __init__(self, nn_structure, std_scale_method=.01, lr=.1, gd_method='SGD', normalization=True):
        self.layers = []
        self.lr = lr
        self.weight_decay_lambda = .01
        # 784 - 20 - 20 - 10
        # input:100x784 - Affine(W:784x20, b:20) - 'ReLU' -  Affine(W:20x20, b:20) - 'ReLU' - Affine(W:20x10, b:10) - SoftmaxWithLoss
        # nn_structure == (784, 'ReLU', 20, 'ReLU', 20, 'SoftmaxWithLoss, 10)
        self.std_scale = std_scale_method
        for i in range(0, len(nn_structure) // 2):
            input_num = nn_structure[2 * i]
            output_num = nn_structure[2 * i + 2]
            activation = eval(nn_structure[2 * i + 1])
            if str(std_scale_method).isalpha():
                if std_scale_method == 'Xavier':
                    self.std_scale = np.sqrt(1 / input_num)
                elif std_scale_method == 'He':
                    self.std_scale = np.sqrt(2 / input_num)
            self.layers.append(Affine(self.std_scale * np.random.randn(input_num, output_num),
                                      np.zeros(output_num), self.lr))

            if activation.__name__ not in ['SoftmaxWithLoss'] and normalization:
                print('ho', activation.__name__)
                self.layers.append(book_BatchNormalization(beta=0, gamma=1))
                # self.layers.append(BatchNormalizaition())
                # self.layers.append(BatchNormalizaition_T())

            self.layers.append(activation())

        self.lastLayer = SoftmaxWithLoss()
        self.temp_loss = None

    def predict(self, x):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return x

    # def loss(self, x, t):
    #     y = self.predict(x)
    #     loss = self.layers[-1].forward(y, t)
    #     return loss

    def loss(self, x, t):
        y = self.predict(x)
        weight_decay = 0
        loss = self.layers[-1].forward(y, t)
        for i in self.layers:
            if i.__class__.__name__ == 'Affine':
                weight_decay += 0.5 * self.weight_decay_lambda * np.sum(i.params['W'] ** 2)
        return loss + weight_decay

    def gradient_descent(self, x, t, method):
        self.temp_loss = self.loss(x,t)
        layer_lst = list(self.layers)
        layer_lst.reverse()
        dout = 1
        for layer in layer_lst:
            dout = layer.backward(dout)

        for layer in self.layers:
            if layer.__class__.__name__ == 'Affine' :
                layer.gradient_descent(method)

    def accuracy(self, x, t):
        y = self.predict(x)
        acc = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / float(y.shape[0])
        return acc
