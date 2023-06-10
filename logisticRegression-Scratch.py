import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Visualising and Preprocessing the Data
df = pd.read_csv("heart.csv")
X = df[["Age", "RestingBP", "MaxHR"]].to_numpy()
Y = df["HeartDisease"].to_numpy()
samples = X.shape[0]
costs = []

# Sigmoid Function to return Probability


def sigmoid(z):
    return 1 / (1 + np.exp(-z))

# Gradient Descent Algorithm to minimize the Loss Function


def gradientDescent(iters, alpha=0.0001):
    num_samples, num_features = X.shape
    w = np.zeros(num_features)
    b = 0
    costs = []
    epsilon = 1e-8

    for _ in range(iters):
        linear_combination = np.dot(X, w) + b
        yhat = sigmoid(linear_combination)

        dw = (1 / num_samples) * np.dot(X.T, (yhat - Y))
        db = (1 / num_samples) * np.sum(yhat - Y)

        w -= alpha * dw
        b -= alpha * db

        cost = (-1 / num_samples) * np.sum(Y * np.log(yhat +
                                                      epsilon) + (1 - Y) * np.log(1 - yhat + epsilon))
        costs.append(cost)

    return w, b, costs

# Function to make Predictions on New Data


def predict(x):
    z = np.dot(x, w) + b
    yhat = sigmoid(z) > 0.5  # Decision Boundary = 0.5
    return yhat


iters = 10000
w, b, costs = gradientDescent(iters)

# Predicting on New Data

print(predict([90, 300, 230]))

# Plot Cost vs No. of Iters

plt.plot(range(len(costs)), costs)
plt.xlabel("Iters")
plt.ylabel("Cost")
plt.title("Cost vs. Iters")
plt.show()

# Plot Classification Boundary
print(Y.shape)
print(X.T[0].shape)
plt.scatter(X.T[0], Y)
plt.plot(X.T[0], sigmoid(X.T[0]), "r")
plt.show()
