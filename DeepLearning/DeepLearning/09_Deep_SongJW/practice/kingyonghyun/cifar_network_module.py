from practice.kingyonghyun.cifar_layer_module import *

'''
신경망을 구성하는 네트워크 클래스.
'''

class NeuralNetwork:

    def __init__(self, nn_structure, std_scale=.01, lr=.1, gd_method='SGD'):
        self.layers = []
        self.lr = lr
        # 784 - 20 - 20 - 10
        # input:100x784 - Affine(W:784x20, b:20) - 'ReLU' -  Affine(W:20x20, b:20) - 'ReLU' - Affine(W:20x10, b:10) - SoftmaxWithLoss
        # nn_structure == (784, 'ReLU', 20, 'ReLU', 20, 'SoftmaxWithLoss, 10)
        for i in range(0, len(nn_structure) // 2):
            input_num = nn_structure[2 * i]
            output_num = nn_structure[2 * i + 2]
            activation = eval(nn_structure[2 * i + 1])
            self.layers.append(Affine(std_scale * np.random.randn(input_num, output_num),
                                      np.zeros(output_num), self.lr))
            self.layers.append(activation())

        self.lastLayer = SoftmaxWithLoss()
        self.temp_loss = None

    def predict(self, x):
        for layer in self.layers[:-1]:
            x = layer.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        loss = self.layers[-1].forward(y, t)
        return loss

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
        y = np.argmax(y, axis=1)  # [[0.1, 0.05, 0.5, 0.05, 0.1, 0.0, 0.0, 0.1, 0.0, 0.1], ....] -> [2, 4, 2, 1, 9, ....]
        if t.ndim != 1: t = np.argmax(t, axis=1)  # t.ndim != 1 이면 one-hot encoding 인 경우이므로, 2차원 배열로 값이 들어온다

        accuracy = np.mean(y == t)
        return accuracy
