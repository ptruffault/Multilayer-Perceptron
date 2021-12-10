import numpy as np

class Sigmoid:
    def func(self, X):
        return 1 / (1 + np.exp(np.negative(X)))

    def prime(self, X):
        return self.func(X) * (1 - self.func(X))

class Relu:
    def func(self, X):
        return np.array([x if x > 0 else 0 for x in X])

    def prime(self, X):
        return np.array([1 if x > 0 else 0 for x in X])






