import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

''' Reading Data, Converting to Numpy Arrays'''

data = pd.read_csv('/data/mnist.csv')
Y = data.iloc[:, 0]
X = (data.iloc[:, 1:]).T
X = X.to_numpy()
Y = Y.to_numpy()


def normalize(x):
    x = x/255
    return x


X = normalize(X)

''' 784 - 120 - 10 Architecture Neural Network for MNIST Classification'''


def init_weights():
    W1 = np.random.rand(120, 784) * 0.2
    b1 = np.random.rand(120, 1) * 0.2
    W2 = np.random.rand(10, 120) * 0.2
    b2 = np.random.rand(10, 1) * 0.2
    return W1, b1, W2, b2


def relu(z):
    return np.maximum(0, z)


def grad_relu(z):
    return 1 * (z > 0)


def softmax(x):
    max_val = np.max(x, axis=0, keepdims=True)
    e_x = np.exp(x - max_val)
    return e_x / np.sum(e_x, axis=0, keepdims=True)


def forward(X, W1, b1, W2, b2):
    Z1 = np.dot(W1, X) + b1
    A1 = relu(Z1)
    Z2 = np.dot(W2, A1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


def one_hot(Y):
    m = Y.shape[0]
    y = np.zeros((m, Y.max()+1))
    y[np.arange(m), Y] = 1
    y = y.T
    return y


def loss(y_true, y_pred):
    return np.mean((y_true - y_pred)**2)


def backward(Z1, A1, Z2, A2, W2, y):
    m = y.shape[1]
    dZ2 = 2*(A2 - y)
    dW2 = (1 / m) * np.dot(dZ2, A1.T)
    db2 = (1 / m) * np.sum(dZ2, axis=1, keepdims=True)
    dA1 = np.dot(W2.T, dZ2)
    dZ1 = dA1 * (Z1 > 0)
    dW1 = (1 / m) * np.dot(dZ1, X.T)
    db1 = (1 / m) * np.sum(dZ1, axis=1, keepdims=True)

    '''Gradient Clipping to prevent Exploading Gradients'''
    max_gradient = 1.0
    if np.linalg.norm(dW1) > max_gradient:
        dW1 = (max_gradient / np.linalg.norm(dW1)) * dW1
    if np.linalg.norm(dW2) > max_gradient:
        dW2 = (max_gradient / np.linalg.norm(dW2)) * dW2

    return dW1, db1, dW2, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, alpha):
    W1 -= (alpha * dW1)
    b1 -= (alpha * db1)
    W2 -= (alpha * dW2)
    b2 -= (alpha * db2)

    return W1, b1, W2, b2


def gradient_descent(X, Y, iters, alpha=0.01):
    W1, b1, W2, b2 = init_weights()
    y = one_hot(Y)
    loss_count = []
    for _ in range(iters):
        Z1, A1, Z2, A2 = forward(X, W1, b1, W2, b2)
        dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, W2, y)
        W1, b1, W2, b2 = update_params(
            W1, b1, W2, b2, dW1, db1, dW2, db2, alpha)

        if _ % 100 == 0:
            loss_val = loss(A2, y)
            loss_count.append(loss_val)
            print(f"Iteration : {_}, Loss : {loss_val}")
            # print(f"Accuracy : {accuracy(A2, y)}")

    plt.plot(np.arange(0, iters, 100), loss_count)  # plot iters vs loss
    plt.show()
    return Z1, A1, Z2, A2


_, _, _, A2 = gradient_descent(X, Y, 1000)
