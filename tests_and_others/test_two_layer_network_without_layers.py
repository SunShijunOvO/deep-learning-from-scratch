import numpy as np
from two_layer_network_without_layers import TwoLayerNetworkWithoutLayers
import time

net = TwoLayerNetworkWithoutLayers(input_size=784, hidden_size=100, output_size=10)

print(net.params['W1'].shape)
print(net.params['b1'].shape)
print(net.params['W2'].shape)
print(net.params['b2'].shape)

start_time = time.time()

x = np.random.rand(100, 784)    # Input 100 samples
y = net.predict(x)
t = np.random.rand(100, 10)

predict_time = time.time()

grads = net.gradient(x, t)

print(grads['W1'].shape)
print(grads['b1'].shape)
print(grads['W2'].shape)
print(grads['b2'].shape)

gradient_time = time.time()

print("Predict time: ")
print(predict_time - start_time)
print("Gradient time: ")
print(gradient_time - predict_time)