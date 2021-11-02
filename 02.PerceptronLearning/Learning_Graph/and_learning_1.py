import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('AND GATE LEARNING 1')

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
# W1: 0.6152  / W2: 0.6284 / theta: (0.3721 + 0.504)
# 0 = 0.6152*x1 + 0.6284*x2 - (0.3721 + 0.504) 
y1 = (0.6152*x1 - (0.3721 + 0.504)) / (-0.6284)     # blue line

# 8 iteration W1, W2
# W1: 0.498199  / W2: 0.509784 / theta: (0.0424716 + 0.174372)
# 0 = 0.498199*x1 + 0.509784*x2 - (0.0424716 + 0.174372)
y2 = (0.498199*x2 - (0.0424716 + 0.174372)) / (-0.509784)       # orange line

# 13 iteration W1, W2
# W1: 0.446639  / W2: 0.457378 / theta: ((-0.0940683) + 0.0378318)
# 0 = 0.446639*x1 + 0.457378*x2 - ((-0.0940683) + 0.0378318)
y3 = (0.446639*x3 - ((-0.0940683) + 0.0378318)) / (-0.457378)       # green line

# 18 iteration W1, W2
# W1: 0.401942  / W2: 0.411931 / theta: (-0.184213 + (-0.0523131))
# 0 = 0.401942*x1 + 0.411931*x2 - (-0.184213 + (-0.0523131)) 
y4 = (0.401942*x4 - (-0.184213 + (-0.0523131))) / (-0.411931)       # red line

# 23 iteration W1, W2
# W1: 0.37016  / W2: 0.372044 / theta: (-0.255882 + (-0.123982))
# 0 = 0.37016*x1 + 0.372044*x2 - (-0.255882 + (-0.123982))
y5 = (0.37016*x5 - (-0.255882 + (-0.123982))) / (-0.372044)         # purple line

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
plt.show()