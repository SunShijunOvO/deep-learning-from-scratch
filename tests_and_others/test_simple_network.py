import numpy as np
from gradient import numerical_gradient
import simple_network

x = np.array([0.6, 0.9])
t = np.array([0, 0, 1])

net = simple_network.SimpleNetwork()

f = lambda w: net.loss(x, t)
d_W = numerical_gradient(f, net.W)

print(d_W)
