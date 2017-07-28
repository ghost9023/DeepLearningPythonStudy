from practice.mnist_nn.layer_module import *

'''
신경망을 구성하는 네트워크 클래스.
'''

class NeuralNetwork:

    def __init__(self, nn_structure, std_scale=.01, lr=.1):
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

    def gradient_descent(self, x, t):
        self.temp_loss = self.loss(x,t)
        layer_lst = list(self.layers)
        layer_lst.reverse()
        dout = 1
        for layer in layer_lst:
            dout = layer.backward(dout)

        for layer in self.layers:
            if layer.__class__.__name__ == 'Affine' :
                layer.gradient_descent()

    def accuracy(self, x, t):
        y = self.predict(x)
        acc = np.sum(np.argmax(y, axis=1) == np.argmax(t, axis=1)) / float(y.shape[0])
        return acc
