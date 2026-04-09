import numpy as np
import matplotlib.pylab as plt
from functions import sigmoid, step_function

x = np.arange(-5.0, 5.0, 0.1)
y1 = sigmoid(x)
y2 = step_function(x)

plt.plot(x, y1)
plt.plot(x, y2, 'k--')
plt.ylim(-0.1, 1.1)     # To confine the range of y-axis
plt.show()