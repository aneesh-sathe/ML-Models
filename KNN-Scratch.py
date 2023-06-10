# Importing Libraries
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from collections import Counter

df = pd.read_csv("diabetes.csv")
X = df.iloc[:, :8]
Y = df.iloc[:, 9:]


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

def euclidDist(p,q):
    return np.sqrt(np.sum((np.array(p) - np.array(q)) ** 2 ))

def KNN(data, k=3):
    
