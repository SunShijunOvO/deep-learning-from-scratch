import numpy as np
import gradient

def gradient_descent(f, init_x, lr=0.01, step_num=100, history=False):
    '''
    The function will modify values of `init_x`.
    '''
    x = init_x
    if history == True:
        x_history = []
        for i in range(step_num):
            x_history.append(x.copy())
            grad = gradient.numerical_gradient(f, x)
            x -= lr * grad
        return x, np.array(x_history)
    else:
        for i in range(step_num):
            grad = gradient.numerical_gradient(f, x)
            x -= lr * grad
        return x