import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('OR GATE LEARNING 3')

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
# W1: 0.6955  / W2: 0.9958 / theta: (0.6389 + 0.3607)
# 0 = 0.6955*x1 + 0.9958*x2 - (0.6389 + 0.3607)   
y1 = (0.6955*x1 - (0.6389 + 0.3607)) / (-0.9958)        # blue line

# 15 iteration W1, W2
# W1: 0.6955  / W2: 0.9958 / theta: (0.450946 + 0.172746)
# 0 = 0.6955*x1 + 0.9958*x2 - (0.450946 + 0.172746)  
y2 = (0.6955*x2 - (0.450946 + 0.172746)) / (-0.9958)        # orange line

# 30 iteration W1, W2
# W1: 0.6955  / W2: 0.9958 / theta: (0.301061 + 0.0228611)
# 0 = 0.6955*x1 + 0.9958*x2 - (0.301061 + 0.0228611)   
y3 = (0.6955*x3 - (0.301061 + 0.0228611)) / (-0.9958)       # green line

# 40 iteration W1, W2
# W1: 0.6955  / W2: 0.9958 / theta: (0.213563 + (-0.0646367))
# 0 = 0.6955*x1 + 0.9958*x2 - (0.213563 + (-0.0646367))  
y4 = (0.6955*x4 - (0.213563 + (-0.0646367))) / (-0.9958)        # red line

# 50 iteration W1, W2
# W1: 0.6955  / W2: 0.9958 / theta: (0.134687 + (-0.143513))
# 0 = 0.6955*x1 + 0.9958*x2 - (0.134687 + (-0.143513))
y5 = (0.6955*x5 - (0.134687 + (-0.143513))) / (-0.9958)         # purple line

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
plt.show()