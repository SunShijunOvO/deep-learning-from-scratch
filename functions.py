import numpy as np

# Activation functions: step function, sigmoid function, and ReLU function,
# as well as identity function and softmax function for output layer

def step_function(x):
    y = x > 0
    return y.astype(np.int_)

def sigmoid(x):
    return 1 / (1 + np.exp(-x))

def sigmoid_grad(x):
    return (1.0 - sigmoid(x)) * sigmoid(x)

def relu(x):
    '''
    ReLU, aka Rectified Linear Unit:
    h(x) =  x, if x > 0;
            0, if x <= 0
    '''
    return np.maximum(0, x)

# For output layer

def identity_function(x):
    return x

def softmax(x):
    x = x - np.max(x, axis=-1, keepdims=True)
    return np.exp(x) / np.sum(np.exp(x), axis=-1, keepdims=True)

# Loss functions: mean squared error (MSE) and cross entropy error (CEE)

def mean_squared_error(y, t):
    return 0.5 * np.sum((y - t)**2)

def cross_entropy_error_simple(y, t):
    delta = 1e-7    # Add a number in log() to pervent -inf as y to 0
    return -np.sum(t * np.log(y + delta))

def cross_entropy_error(y, t):
    if y.ndim == 1:
        t = t.reshape(1, t.size)
        y = y.reshape(1, y.size)
    # Change one-hot data to index
    if t.size == y.size:
        t = t.argmax(axis=1)
    batch_size = y.shape[0]
    return -np.sum(np.log(y[np.arange(batch_size), t] + 1e-7)) / batch_size
