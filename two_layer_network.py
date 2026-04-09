import numpy as np
from layers import *
from gradient import numerical_gradient
from collections import OrderedDict

class TwoLayerNetwork:
    def __init__(self, input_size, hidden_size, ouput_size, weight_init_std=0.01):
        # Initialize weights and biases
        self.params = {}
        self.params['W1'] = np.random.randn(input_size, hidden_size) * weight_init_std
        self.params['b1'] = np.zeros(hidden_size)
        self.params['W2'] = np.random.randn(hidden_size, ouput_size) * weight_init_std
        self.params['b2'] = np.random.randn(ouput_size)
        # Generate layers
        self.layers = OrderedDict()
        self.layers['affine_1'] = Affine(self.params['W1'], self.params['b1'])
        self.layers['relu_1'] = Relu()
        self.layers['affine_2'] = Affine(self.params['W2'], self.params['b2'])
        self.last_layer = SoftmaxWithLoss()

    def predict(self, x):
        for layers in self.layers.values():
            x = layers.forward(x)
        return x

    def loss(self, x, t):
        y = self.predict(x)
        return self.last_layer.forward(y, t)

    def accuracy(self, x, t):
        y = self.predict(x)
        y = np.argmax(y, axis=1)
        if t.ndim != 1 : t = np.argmax(t, axis=1)
        accu = np.sum(y == t) / float(x.shape[0])
        return accu

    def numerical_gradient(self, x, t):
        loss_W = lambda W: self.loss(x, t)
        grads = {}
        grads['W1'] = numerical_gradient(loss_W, self.params['W1'])
        grads['b1'] = numerical_gradient(loss_W, self.params['b1'])
        grads['W2'] = numerical_gradient(loss_W, self.params['W2'])
        grads['b2'] = numerical_gradient(loss_W, self.params['b2'])
        return grads

    def gradient(self, x, t):
        # Forward
        self.loss(x, t)

        # Backward
        d_out = 1
        d_out = self.last_layer.backward(d_out)
        layers = list(self.layers.values())
        layers.reverse()
        for layer in layers:
            d_out = layer.backward(d_out)
        
        # Set
        grads = {}
        grads['W1'] = self.layers['affine_1'].d_W
        grads['b1'] = self.layers['affine_1'].d_b
        grads['W2'] = self.layers['affine_2'].d_W
        grads['b2'] = self.layers['affine_2'].d_b
        return grads