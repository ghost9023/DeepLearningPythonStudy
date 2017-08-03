import numpy as np

def img2col(x, fh, fw, stride=1, pad=0, pad_value=0):
    x = np.pad(x, pad_width=((0, 0), (0, 0), (pad, pad), (pad, pad)), mode='constant', constant_values=pad_value)
    N, C, H, W = x.shape
    OH = (H - fh) / stride + 1
    OW = (W - fw) / stride + 1

    result = []
    for n in range(N):
        for h in range(int(OH)):
            for w in range(int(OW)):
               result.append(x[n, :, h * stride : h * stride + fh, w * stride : w * stride + fw].flatten())

    return np.array(result)


if __name__ == '__main__':
    x = np.arange(64).reshape(2,2,4,4)
    print(x)
    print(np.array(img2col(x, 3, 3, 1, 0)))

