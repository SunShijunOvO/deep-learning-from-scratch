import numpy as np
from mnist_network import get_data, init_network, predict

x, t = get_data()
network = init_network()

# Normal
accuracy_cnt = 0
for i in range(len(x)):
    y = predict(network, x[i])
    p = np.argmax(y)    # To get the index with maximum probability
    if p == t[i]:
        accuracy_cnt += 1

print("Accuracy: " + str(float(accuracy_cnt) / len(x)))

# Batch
batch_size = 100
batch_accuracy_cnt = 0

for i in range(0, len(x), batch_size):
    x_batch = x[i:i + batch_size]
    y_batch = predict(network, x_batch)
    p2 = np.argmax(y_batch, axis=1) # axis=0 means finding in a column, 1 means finding in a row
    batch_accuracy_cnt += np.sum(p2 == t[i:i + batch_size])

print("Batch accuracy: " + str(float(batch_accuracy_cnt) / len(x)))