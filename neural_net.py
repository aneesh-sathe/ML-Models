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
print(a)
