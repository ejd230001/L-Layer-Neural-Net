import numpy as np

def relu(Z):
     return np.maximum(Z, 0)

def sigmoid(Z):
     return 1 / (1 + np.exp(-Z))

def relu_derivative(Z):
    dZ = np.where(Z > 0, 1, 0)
    return dZ


def sigmoid_derivative(Z):
     A = sigmoid(Z)
     return np.multiply(A, (1 - A))