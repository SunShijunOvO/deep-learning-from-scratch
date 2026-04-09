import matplotlib.pyplot as plt
from dataset.mnist import load_mnist
from utilities import smooth_curve
from multi_layer_network import MultiLayerNetwork
from optimizer import *

# 0. Load MNIST dataset
(x_train, t_train), (x_test, t_test) = load_mnist(normalize=True)
train_size = x_train.shape[0]
batch_size = 128
max_iterations = 2000

# 1. Setup the experiment
optimizers = {}
optimizers['SGD'] = StochasticGradientDescent()
optimizers['Momentum'] = Momentum()
optimizers['AdaGrad'] = AdaptiveGrad()
optimizers['Adam'] = Adam()
optimizers['RMSprop'] = RMSProp()

networks = {}
train_loss = {}
for key in optimizers.keys():
    networks[key] = MultiLayerNetwork(input_size=784, hidden_size_list=[100,100,100,100], output_size=10)
    train_loss[key] = []

# 2. Begin training
for i in range(max_iterations):
    batch_mask = np.random.choice(train_size, batch_size)
    x_batch = x_train[batch_mask]
    t_batch = t_train[batch_mask]

    for key in optimizers.keys():
        grads = networks[key].gradient(x_batch, t_batch)
        optimizers[key].update(networks[key].params, grads)

        loss = networks[key].loss(x_batch, t_batch)
        train_loss[key].append(loss)

    if i % 100 == 0:
        print("===========" + "iteration:" + str(i) + "===========")
        for key in optimizers.keys():
            loss = networks[key].loss(x_batch, t_batch)
            print(key + ":" + str(loss))

# 3. Plot the graph
markers = {"SGD": "o", "Momentum": "x", "AdaGrad": "s", "Adam": "D", "RMSprop": "o"}
x = np.arange(max_iterations)
for key in optimizers.keys():
    plt.plot(x, smooth_curve(train_loss[key]), marker=markers[key], markevery=100, label=key)
plt.xlabel("iterations")
plt.ylabel("loss")
plt.ylim(0, 1)
plt.legend()
plt.show()