import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# we will create a 3-5-2 neural network architecture

data = np.array([1, 2, 3])  # (1,3 Input Data)
actual = np.array([4, 5])  # (1,2 Output Data)


def init_params(layers_conf):  # initialise the weight and bias matrices based on the layers
    layers = []
    for i in range(1, len(layers_conf)):
        layers.append([
            np.random.rand(layers_conf[i-1], layers_conf[i]),
            np.zeros((1, layers_conf[i]))
        ])
    return layers


def relu(x):
    return np.max(0, x)


def forwardPass(data, layers):
    backData = [data.copy()]  # maintain a copy of the data for Backpropagation
    for i in range(len(layers)):
        data = np.matmul(data.T, layers[i][0]) + layers[i][1]  # y = w.x + b
        backData.append(data.copy())
        if i != len(layers):
            data = relu(data)
    return data, backData


def loss(actual, predicted):
    return (predicted - actual) ** 2


def backwardPass(layers, backData, grads, lr=1e-6):
    for i in range(-1, -(len(layers)+1), -1):
        if i != len(layers):
            # heaviside -> approximate derivate of ReLU Function
            grads = np.multiply(grads, np.heaviside(backData[i+1], 0))

        w_grad = backData[i].T @ grads
        b_grad = np.mean(grads, axis=0)

        layers[i][0] -= lr * w_grad
        layers[i][1] -= lr * b_grad

        grads = grads @ layers[i][0].T
    return layers


layers_conf = [3, 5, 2]  # network architecture
layers = init_params(layers_conf)
iters = 100
loss_count = []

for _ in range(iters):
    prediction, Backdata = forwardPass(data, layers)
    lossValue = loss(prediction, actual)
    loss_count.append(lossValue)

    layers = backwardPass(layers, Backdata, loss)

plt.plot((0, len(loss_count), loss_count))  # plotting loss vs iterations
plt.show()
