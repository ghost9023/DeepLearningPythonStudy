# import numpy as np
# import matplotlib.pyplot as plt
#
# plt.figure(figsize=(10,10))
# x = np.linspace(-10, 10, 1000)
# y = np.linspace(-10, 10, 1000)
# X, Y = np.meshgrid(x,y)
# F1 = (X**2)/20 + Y**2 -.5
# F2 = (X**2)/20 + Y**2 -1
# F3 = (X**2)/20 + Y**2 -1.5
# F4 = (X**2)/20 + Y**2 -2
# F5 = (X**2)/20 + Y**2 -2.5
# F6 = (X**2)/20 + Y**2 -3
# F7 = (X**2)/20 + Y**2 -3.5
# plt.contour(X,Y,F1,[0])
# plt.contour(X,Y,F2,[0])
# plt.contour(X,Y,F3,[0])
# plt.contour(X,Y,F4,[0])
# plt.contour(X,Y,F5,[0])
# plt.contour(X,Y,F6,[0])
# plt.contour(X,Y,F7,[0])
# plt.show()

import matplotlib
import numpy as np
import matplotlib.cm as cm
import matplotlib.mlab as mlab
import matplotlib.pyplot as plt

matplotlib.rcParams['xtick.direction'] = 'out'
matplotlib.rcParams['ytick.direction'] = 'out'

# delta = 0.025
# x = np.arange(-3.0, 3.0, delta)
# y = np.arange(-2.0, 2.0, delta)
x = np.linspace(-40, 40, 1000)
y = np.linspace(-10, 10, 1000)
X, Y = np.meshgrid(x, y)
Z = (X **2)/20 + Y **2


# Create a simple contour plot with labels using default colors.  The
# inline argument to clabel will control whether the labels are draw
# over the line segments of the contour, removing the lines beneath
# the label
plt.figure(figsize=(16,4))
levels = np.arange(0, 80, 5)
CS = plt.contour(X, Y, Z, levels = levels)
plt.clabel(CS, inline=1, fontsize=10)
plt.title('Simplest default with labels')
plt.plot([5],[5],'bo')

plt.show()