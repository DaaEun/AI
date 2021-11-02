import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('AND GATE LEARNING 2')

plt.xlim([0, 1])    # X축의 범위: [xmin, xmax]
plt.ylim([-2, 2])   # Y축의 범위: [ymin, ymax]
x1 = np.array(range(10))
x2 = np.array(range(10))
x3 = np.array(range(10))
x4 = np.array(range(10))
x5 = np.array(range(10))

# fuction
# 0 = w1*x1 + w2*x2 + w3*theta  (w3 = -1)
# 0 = w1*x1 + w2*x2 - theta

# init W1, W2
# W1: 0.6527 / W2: 0.9047 / theta: (0.7372 + 0.9809)
# 0 = 0.6527*x1 + 0.9047*x2 - (0.7372 + 0.9809)
y1 = (0.6527*x1 - (0.7372 + 0.9809)) / (-0.9047)     # blue line

# 8 iteration W1, W2
# W1: 0.508392 / W2: 0.752939 / theta: (0.314258 + 0.557958)
# 0 = 0.508392*x1 + 0.752939*x2 - (0.314258 + 0.557958)
y2 = (0.508392*x2 - (0.314258 + 0.557958)) / (-0.752939)       # orange line

# 18 iteration W1, W2
# W1: 0.385621 / W2: 0.617712 / theta: ((-0.0461631) + 0.197537)
# 0 = 0.385621*x1 + 0.617712*x2 - ((-0.0461631) + 0.197537)
y3 = (0.385621*x3 - ((-0.0461631) + 0.197537)) / (-0.617712)       # green line

# 28 iteration W1, W2
# W1: 0.297579 / W2: 0.516765 / theta: (-0.259059 + (-0.0153589))
# 0 = 0.297579*x1 + 0.516765*x2 - (-0.259059 + (-0.0153589))
y4 = (0.297579*x4 - (-0.259059 + (-0.0153589))) / (-0.516765)       # red line

# 38 iteration W1, W2
# W1: 0.290051 / W2: 0.435845 / theta: (-0.347507 + (-0.103807))
# 0 = 0.290051*x1 + 0.435845*x2 - (-0.347507 + (-0.103807))
y5 = (0.290051*x5 - (-0.347507 + (-0.103807))) / (-0.435845)         # purple line

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
plt.show()