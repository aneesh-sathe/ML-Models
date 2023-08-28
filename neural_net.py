import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# we will create a 3-5-2 neural network architecture


def init_params(layers_conf):  # initialise the weight and bias matrices based on the layers
    layers = []
    for i in range(1, len(layers_conf)):
        layers.append([
            np.random.rand(layers_conf[i-1], layers_conf[i]),
            np.zeros((1, layers_conf[i]))
        ])
    return layers


layers_conf = [3, 5, 2]
a = init_params(layers_conf)


def relu(x):
    return max(0, x)


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


def backwardPass(layers, backData, grads, lr=1e-4):
    for i in range(-1, -(len(layers)+1), -1):
        if i != len(layers):
            grad = np.multiply(grad, np.heaviside(backData[i+1], 0))

        w_grad = backData[i].T @ grad
        b_grad = np.mean(grad, axis=0)

        layers[i][0] -= lr * w_grad
        layers[i][1] -= lr * b_grad

        grad = grad @ layers[i][0].T
