import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Visualising and Preprocessing the Data
df = pd.read_csv("diabetes.csv")
X = df[["Pregnancies", "Glucose", "BloodPressure", "SkinThickness",
        "Insulin", "BMI", "DiabetesPedigreeFunction", "Age"]]

# Normalising the Data


def min_max(x, xmax, xmin):
    return ((x-xmax) / (xmax-xmin))


for c in X.columns:
    xmax = X[c].max()
    xmin = X[c].min()
    X[c] = X[c].apply(lambda x: min_max(x, xmax, xmin))


X = X.to_numpy()
Y = df["Outcome"].to_numpy()
samples = X.shape[0]
costs = []

# Sigmoid Function to return Probability


def sigmoid(z):
    return (1 / (1 + (np.exp(-z))))

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


iters = 100000
w, b, costs = gradientDescent(iters)
print(w, b)

# Predicting on New Data

print(predict([6, 148, 72, 35, 0, 33.6, 0.627, 50]))

# Plot Cost vs No. of Iters

plt.plot(range(len(costs)), costs)
plt.xlabel("Iters")
plt.ylabel("Cost")
plt.title("Cost vs. Iters")
plt.show()

# Plot Classification Boundary
# print(Y.shape)
# print(X.T[0].shape)
plt.scatter(X.T[0], Y)
plt.plot(len(range(2)), sigmoid(10),  "r")
plt.show()
