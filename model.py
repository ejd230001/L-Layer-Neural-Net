import numpy as np

class Model:

    def __init__(self, dims):
        self.parameters = {}
        self.caches = []
        self.costs = []
        self.initialize_parameters(dims)

    def initialize_parameters(self, dims):
        print("Hello")

    def single_forward(self):
        print("Hello")
    
    def single_backward(self):
        print("Hello")

    def forward_propagate(self, X):
        print("Hello")

    def calculate_cost(self, AL, Y):
        print("Hello")
    
    def backward_propagate(self, AL):
        print("Hello")

    def update_parameters(self, grads, learning_rate):
        print("Hello")

    def train_model(self, X, Y, learning_rate, num_iterations):
        print("Hello")
    
    def predict(self, x):
        print("Hello")

    
    

