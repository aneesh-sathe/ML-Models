# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("diabetes.csv")
X = df.iloc[:, :8]
Y = df.iloc[:, -1:]

# Cleaning the Data
for column in X.columns[1:]:
    X[column] = X[column].replace(0, np.NaN)
    replace_val = int(X[column].mean(skipna=True))
    X[column] = X[column].replace(np.NaN, replace_val)

# Normalizing Data


def zNorm(x, mu, sigma):
    return ((x - mu) / sigma)


for column in X.columns:
    mu = X[column].mean()
    sigma = X[column].std()
    X[column] = X[column].apply(lambda x: zNorm(x, mu, sigma))


X['Label'] = Y
X = X.to_numpy()

# Distance Function


def euclidDist(p, q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2))

# KNN Classifier


def KNN(trainData, testData, k=3):
    distances = []
    for i in trainData:
        d = euclidDist(i, testData)
        distances.append([i, d])
    distances = sorted(distances, key=lambda x: x[1])
    neighbours = [distances[:k][0][0][-1]]
    prediction = Counter(neighbours).most_common(1)[0][0]
    return prediction


predictedLabel = KNN(X, [0.63953049, 0.86469014, -0.0319691,  0.67020577, -0.00330798, 0.16713124,
                         0.46818687,  1.42506672, 1])

# Predicting Class Label for New Data
print(predictedLabel)
