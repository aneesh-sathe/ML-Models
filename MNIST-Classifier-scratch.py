import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

data = pd.read_csv('train.csv')
data.head()
X = data.iloc[:, 1:]
Y = data.iloc[:, 0]
X = X.to_numpy()
Y = Y.to_numpy()
X = X.T
X = X/255

''' Classification Neural Network will be 784-120-10 Architecture'''


def init_params():
    W1 = np.random.rand(784, 120) - 0.5
    b1 = np.random.rand(120, 1) - 0.5
    W2 = np.random.rand(120, 10) - 0.5
    b2 = np.random.rand(10, 1) - 0.5
    return W1, b1, W2, b2


W1, b1, W2, b2 = init_params()


def relu(z):
    return np.maximum(0, z)


def relu_grad(z):
    return z > 0


def softmax(Z):
    return np.exp(Z) / np.sum(np.exp(Z))


def forward(W1, b1, W2, b2, X):
    Z1 = W1.T.dot(X) + b1
    A1 = relu(Z1)
    Z2 = W2.T.dot(Z1) + b2
    A2 = softmax(Z2)
    return Z1, A1, Z2, A2


Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)


def loss(y_true, y_hat):
    return np.mean((np.argmax(y_hat, 0) - y_true)**2)


def one_hot(Y):
    m = Y.shape[0]
    one_hot = np.zeros((m, Y.max()+1))
    one_hot[np.arange(m), Y] = 1
    one_hot = one_hot.T
    return one_hot


def backward(Z1, A1, Z2, A2, Y):
    m = Y.shape[0]
    y = one_hot(Y)
    dL_dA2 = 2 * (A2 - y)
    dA2_dZ2 = A2 * (1 - A2)
    dL_dZ2 = dL_dA2 * dA2_dZ2
    dL_dW2 = (1 / m) * np.dot(dL_dZ2, A1.T)
    dL_db2 = (1 / m) * np.sum(dL_dZ2, axis=1, keepdims=True)

    dZ2_dA1 = W2
    dL_dA1 = np.dot(dZ2_dA1, dL_dZ2)
    dA1_dZ1 = np.where(A1 > 0, 1, 0)
    dL_dZ1 = dL_dA1 * dA1_dZ1
    dL_dW1 = (1 / m) * np.dot(dL_dZ1, X.T)
    dL_db1 = (1 / m) * np.sum(dL_dZ1, axis=1, keepdims=True)
    return dW1.T, db1, dW2.T, db2


def update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr):
    W1 = W1 - lr*dW1
    W2 = W2 - lr*dW2
    b1 = b1 - lr*db1
    b2 = b2 - lr*db2
    return W1, b1, W2, b2


def gradient_descent(X, Y, iters, lr=0.001):
    W1, b1, W2, b2 = init_params()
    loss_array = []
    y = one_hot(Y)
    for _ in range(iters):
        Z1, A1, Z2, A2 = forward(W1, b1, W2, b2, X)
        dW1, db1, dW2, db2 = backward(Z1, A1, Z2, A2, Y)
        W1, b1, W2, b2 = update_params(W1, b1, W2, b2, dW1, db1, dW2, db2, lr)
        if _ % 100 == 0:
            print('Iteration :', _)
            print('Loss :', loss(y, A2))
            loss_array.append(loss(y, A2))

    plt.plot(np.arange(0, iters, 100), loss_array)
    return W1, b1, W2, b2


gradient_descent(X, Y, 2000, lr=0.0001)
