
# Import Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# Load & Visualize Data
df = pd.read_csv("data/housing.csv")
X = df[["housing_median_age", "total_rooms", "median_income"]]
y = df["median_house_value"]


# Feature Scaling
def min_max(x, xmax, xmin):
    return ((x-xmax) / (xmax-xmin))


for c in X.columns:
    xmax = X[c].max()
    xmin = X[c].min()
    X[c] = X[c].apply(lambda x: min_max(x, xmax, xmin))


X = X.to_numpy()
y = y.to_numpy()
n, features = X.shape
print(X.shape)
print(X[0].shape)


def gradientDescent(iters, alpha=0.01):
    w = np.zeros(features)
    b = 0
    costs = []
    for _ in range(iters):
        yhat = np.dot(X, w)
        dw = (1 / n) * np.dot(X.T, (yhat - y))
        db = (1 / n) * np.sum(yhat - y)

        w -= (alpha*dw)
        b -= (alpha*db)

        cost = np.mean((yhat - y) ** 2)
        costs.append(cost)
    return w, b, costs


def predict(x):
    return np.dot(x, w) + b


iters = 100000
w, b, costs = gradientDescent(iters)

# Prediction
x = np.array([-0.21568627, -0.97766926, -0.46033158])
yhat = predict(x)
print(yhat)

# Plotting Learning Curve
plt.plot(range(iters), costs)
plt.xlabel("No. of Iters")
plt.ylabel("Cost")
plt.title("No of Iters vs Cost")
plt.show()
