import numpy as np

# x = np.random.rand(2, 3, 5, 5)
# print(x.shape)

y = np.arange(81).reshape(3,3,3,3)
print(y)
#
# z = np.arange(16).reshape(2,2,2,2)
# print(z)
#
# a = np.arange(8).reshape(2,2,2)
# print(a.flatten())

def im2col(x, fh, fw, stride=1, pad=0):
    oh = int((x.shape[2] + 2 * pad - fh) / stride + 1)
    ow = int((x.shape[3] + 2 * pad - fw) / stride + 1)
    lst = []

    for i in range(x.shape[0]):
        for h in range(oh):
            for w in range(ow):
                temp = x[i, 0:x.shape[1], h:h+fh, w:w+fw].flatten()
                print(temp)
                lst.append(temp)
    return np.array(lst)

b = im2col(y, 2, 2, 1, 0)
print(b.shape)
filt = np.arange(36).reshape(3,3,2,2).reshape(-1,12).T
print(filt.shape)
print(np.dot(b, filt).T.reshape(3,3,2,2))