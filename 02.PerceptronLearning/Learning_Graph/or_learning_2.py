import matplotlib.pyplot as plt
import numpy as np

plt.xlabel('X1')
plt.ylabel('X2')
plt.title('OR GATE LEARNING 2')

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
# W1: 0.6877  / W2: 0.7202 / theta: (0.0571 + 0.5912)
# 0 = 0.6877*x1 + 0.7202*x2 - (0.0571 + 0.5912)   
y1 = (0.6877*x1 - (0.0571 + 0.5912)) / (-0.7202)        # blue line

# 5 iteration W1, W2
# W1: 0.6877  / W2: 0.7202 / theta: ((-0.00648196) + 0.527618)
# 0 = 0.6877*x1 + 0.7202*x2 - ((-0.00648196) + 0.527618)   
y2 = (0.6877*x2 - ((-0.00648196) + 0.527618)) / (-0.7202)       # orange line

# 15 iteration W1, W2
# W1: 0.6877  / W2: 0.7202 / theta: ((-0.103872) + 0.430228)
# 0 = 0.6877*x1 + 0.7202*x2 - ((-0.103872) + 0.430228)  
y3 = (0.6877*x3 - ((-0.103872) + 0.430228)) / (-0.7202)         # green line

# 25 iteration W1, W2
# W1: 0.6877  / W2: 0.7202 / theta: ((-0.191491) + 0.342609)
# 0 = 0.6877*x1 + 0.7202*x2 - ((-0.191491) + 0.342609)   
y4 = (0.6877*x4 - ((-0.191491) + 0.342609)) / (-0.7202)         # red line

# 35 iteration W1, W2
# W1: 0.6877  / W2: 0.7202 / theta: ((-0.270473) + 0.263627)
# 0 = 0.6877*x1 + 0.7202*x2 - ((-0.270473) + 0.263627)
y5 = (0.6877*x5 - ((-0.270473) + 0.263627)) / (-0.7202)         # purple line

plt.plot(x1, y1, x2, y2, x3, y3, x4, y4, x5, y5)
plt.show()