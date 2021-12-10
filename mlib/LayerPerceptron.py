from .ActivationFunction import *
import numpy as np

class LayerPerceptron:
    def __init__(self, size, input_size, activation_function):
        self.size = size
        self.input_size = input_size
        self.weights = np.random.randn(size, input_size)
        self.biases = np.random.randn(size)
        if activation_function == 'Relu':
            self.activation_function = Relu()
        else:
            self.activation_function = Sigmoid()

    def forward(self, X):
        return self.activation(self.aggregation(X))

    def aggregation(self, x):
        return np.dot(self.weights, x) + self.biases

    def activation(self, x):
        return self.activation_function.func(x)

    def activation_prime(self, x):
        return self.activation_function.prime(x)

    def update_weights(self, gradient, learning_rate):
        self.weights -= learning_rate * gradient

    def update_biases(self, gradient, learning_rate):
        self.biases -= learning_rate * gradient


