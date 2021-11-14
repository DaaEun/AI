import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('OR GATE LEARNING 1')

plt.xlim([0, 1])    # X축의 범위: [xmin, xmax]
plt.ylim([-1, 1])   # Y축의 범위: [ymin, ymax]
x1 = np.array(range(10))
x2 = np.array(range(10))
x3 = np.array(range(10))
x4 = np.array(range(10))
x5 = np.array(range(10))

# fuction
# 0 = w1*x1 + w2*x2 + w3*theta  (w3 = -1)
# 0 = w1*x1 + w2*x2 - theta

# init W1, W2
# W1: 0.6739  / W2: 0.057 / theta: (0.5122 + 0.106)
# 0 = 0.6739*x1 + 0.057*x2 - (0.5122 + 0.106)   
y1 = (0.6739*x1 - (0.5122 + 0.106)) / (-0.057)      # blue line

# 7 iteration W1, W2
# W1: 0.6739 / W2: 0.057 / theta: (0.429525 + 0.0233245)
# 0 = 0.6739*x1 + 0.057*x2 - (0.429525 + 0.0233245)
y2 = (0.6739*x2 - (0.429525 + 0.0233245)) / (-0.057)        # orange line

# 14 iteration W1, W2
# W1: 0.6739  / W2: 0.057 / theta: (0.362719 + (-0.0434811))
# 0 = 0.6739*x1 + 0.057*x2 - (0.362719 + (-0.0434811))
y3 = (0.6739*x3 - (0.362719 + (-0.0434811))) / (-0.057)     # green line

# 27 iteration W1, W2
# W1: 0.6739  / W2: 0.057 / theta: (0.251002 + (-0.155198))
# 0 = 0.6739*x1 + 0.057*x2 - (0.251002 + (-0.155198))
y4 = (0.6739*x4 - (0.251002 + (-0.155198))) / (-0.057)      # red line

# 34 iteration W1, W2
# W1: 0.6739  / W2: 0.057 / theta: (0.196783 + (-0.209417))
# 0 = 0.6739*x1 + 0.057*x2 - (0.196783 + (-0.209417))
y5 = (0.6739*x5 - (0.196783 + (-0.209417))) / (-0.057)      # purple line

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
plt.show()