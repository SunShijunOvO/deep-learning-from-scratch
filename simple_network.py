import numpy as np
from functions import softmax, cross_entropy_error
from gradient import numerical_gradient

class SimpleNetwork:
    def __init__(self):
        self.W = np.random.randn(2, 3)  # Return sample(s) from the "standard normal" distribution.

    def predict(self, x):
        return np.dot(x, self.W)
    
    def loss(self, x, t):
        z = self.predict(x)
        y = softmax(z)
        loss = cross_entropy_error(y, t)
        return loss