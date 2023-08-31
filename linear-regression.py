# Importing Libraries

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Visualising and Preprocessing the Data

df = pd.read_csv("data/salary.csv")
X = df[["YearsExperience"]].to_numpy()
Y = df["Salary"].to_numpy()
costs = []


# Gradient Descent Algorithm to minimize the Loss Function

def gradientDescent(iters, alpha=0.001):
    n = X.shape[0]
    w = np.zeros(X.shape[1])
    b = 0
    costs = []

    for _ in range(iters):
        yhat = np.dot(X, w) + b

        dw = (1 / n) * np.dot(X.T, (yhat - Y))
        db = (1 / n) * np.sum(yhat - Y)

        w = w - (alpha * dw)
        b = b - (alpha * db)

        cost = np.mean((yhat - Y) ** 2)
        costs.append(cost)

    return w, b, costs

# Function to make Predictions on New Data


def predict(x):
    yhat = np.dot(x, w) + b
    return yhat


iters = 10000
w, b, costs = gradientDescent(iters)
print(w)
print(b)

# Predicting on New Data

print(predict(20))

# Plot Cost vs No. of Iters

plt.plot(range(len(costs)), costs)
plt.xlabel("Iters")
plt.ylabel("Cost")
plt.title("Cost vs. Iters")
plt.show()

# Plot Regression Line

plt.scatter(X, Y)
plt.plot(X, predict(X), "r")
plt.show()
