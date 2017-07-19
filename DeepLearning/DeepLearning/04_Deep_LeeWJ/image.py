import sys, os
import numpy as np
from mnist import load_mnist
from PIL import Image


def img_show(img):
    pil_img = Image.fromarray(np.uint8(img))
    pil_img.show()

(x_train, t_train), (x_test, t_test) = load_mnist(flatten=True,
                                                  normalize=False)  #flatten=T 설정으로, 2차원 형태의 이미지를, 1차원형태의 벡터로 변환했음.

img = x_train[0]
label = t_train[0]
print(label)

print(img.shape)  #784,     << flatten설정으로 28x28 --> 784가 된 모습
img = img.reshape(28,28)
print(img.shape)  #28,28    << reshape 설정으로 다시 28x28이 된 모습

img_show(img)


