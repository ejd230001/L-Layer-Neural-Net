import numpy as np
from activations import relu, sigmoid, relu_derivative, sigmoid_derivative

class Model:
    def __init__(self, dims):
        self.parameters = {}
        self.L = len(dims)
        self.caches = []
        self.costs = []
        self.initialize_parameters(dims)

    def initialize_parameters(self, dims):
        self.parameters = {}
        for i in range(1, len(dims)):
            self.parameters['W' + str(i)] = np.random.randn(dims[i], dims[i - 1]) / np.sqrt(dims[i-1])
            self.parameters['b' + str(i)] = np.zeros((dims[i], 1))

    def single_forward(self, A_prev, iteration, activation):
        cache = {}
        W = self.parameters['W' + str(iteration)]
        b = self.parameters['b' + str(iteration)]
        Z = np.dot(W, A_prev) + b
        A = relu(Z) if activation == 'relu' else sigmoid(Z)

        cache['A_prev'] = A_prev
        cache['Z'] = Z
        cache['W'] = W

        return A, cache
        
    def single_backward(self, dA, cache, activation):
        Z, A_prev, W = cache['Z'], cache['A_prev'], cache['W']
        m = A_prev.shape[1]
        dAdZ = relu_derivative(Z) if activation == "relu" else sigmoid_derivative(Z)
        dZ = np.multiply(dA, dAdZ)
        dW = np.dot(dZ, A_prev.T) / m
        dB = np.sum(dZ, axis=1, keepdims=True) / m
        dA_prev = np.dot(W.T, dZ)

        return dA_prev, dW, dB

    def forward_propagate(self, X):
        self.caches = []
        A_prev = X
        for i in range(1, self.L - 1):
            A, cache = self.single_forward(A_prev, i, 'relu')
            self.caches.append(cache)
            A_prev = A
        AL, cache = self.single_forward(A_prev, self.L-1, "sigmoid")
        self.caches.append(cache)

        return AL

    def calculate_cost(self, AL, Y):
        m = Y.shape[1]
        cost = np.sum(np.multiply(Y, np.log(AL)) + np.multiply(1 - Y, np.log(1 - AL))) / -m
        cost = np.squeeze(cost)
        
        return cost
    
    def backward_propagate(self, AL, Y):
        dA = - (np.divide(Y, AL) - np.divide(1 - Y, 1 - AL))
        grads = {}
        
        dA_prev, dW, db = self.single_backward(dA, self.caches[len(self.caches) - 1], 'sigmoid')
        first = str(len(self.caches) - 1 + 1)
        grads['dW' + first] = dW
        grads['db' + first] = db

        nextdA = dA_prev

        for i in range(len(self.caches) - 2, -1, -1):
            dA_prev, dW, db = self.single_backward(nextdA, self.caches[i], 'relu')
            grads['dW' + str(i + 1)] = dW
            grads['db' + str(i + 1)] = db
            nextdA = dA_prev
        return grads


    def update_parameters(self, grads, learning_rate):
        for i in range(1, self.L):
            self.parameters['W' + str(i)] -= learning_rate * grads['dW' + str(i)]
            self.parameters['b' + str(i)] -= learning_rate * grads['db' + str(i)]

    def train_model(self, X, Y, learning_rate, num_iterations):
        for i in range(num_iterations):
            AL = self.forward_propagate(X)
            cost = self.calculate_cost(AL, Y)
            if i % 100 == 0 or i == num_iterations - 1:
                print("Cost after iteration " + str(i) + " = " + str(cost))
            grads = self.backward_propagate(AL, Y)
            self.update_parameters(grads, learning_rate)
    
    def predict(self, x):
        result = np.squeeze(self.forward_propagate(x))
        if result >= .5:
            return 1
        else:
            return 0
        

    
    

