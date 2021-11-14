import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('AND GATE LEARNING 3')

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
# W1: 0.6661 / W2: 0.7814 / theta: (0.9304 + 0.0597)
# 0 = 0.6661*x1 + 0.7814*x2 - (0.9304 + 0.0597) 
y1 = (0.6661*x1 - (0.9304 + 0.0597)) / (-0.7814)     # blue line

# 8 iteration W1, W2
# W1: 0.543339 / W2: 0.653162 / theta: (0.580913 + (-0.289787))
# 0 = 0.543339*x1 + 0.653162*x2 - (0.580913 + (-0.289787))
y2 = (0.543339*x2 - (0.580913 + (-0.289787))) / (-0.653162)       # orange line

# 13 iteration W1, W2
# W1: 0.489051 / W2: 0.595484 / theta: (0.427259 + (-0.443441))
# 0 = 0.489051*x1 + 0.595484*x2 - (0.427259 + (-0.443441))
y3 = (0.489051*x3 - (0.427259 + (-0.443441))) / (-0.595484)       # green line

# 23 iteration W1, W2
# W1: 0.40161 / W2: 0.501651 / theta: (0.245986 + (-0.624714))
# 0 = 0.40161*x1 + 0.501651*x2 - (0.245986 + (-0.624714))
y4 = (0.40161*x4 - (0.245986 + (-0.624714))) / (-0.501651)       # red line

# 28 iteration W1, W2
# W1: 0.394077 / W2: 0.462556 / theta: (0.199358 + (-0.671342))
# 0 = 0.394077*x1 + 0.462556*x2 - (0.199358 + (-0.671342))
y5 = (0.394077*x5 - (0.199358 + (-0.671342))) / (-0.462556)         # purple line

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
plt.show()