import numpy as np
from dataset.mnist import load_mnist

(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True, one_hot_label=True)
    # `one_hot_label=True` helps to get the `one-hot` array

print(x_train.shape)
print(t_train.shape)

# Choose 10 samples in mnist dataset randomly
train_size = x_train.shape[0]
batch_size = 10
batch_mask = np.random.choice(train_size, batch_size)
    # Choose 10 numbers in range(60000) randomly
x_batch = x_train[batch_mask]
t_batch = t_train[batch_mask]