import numpy as np


class MeanSquaredError:
    def __init__(self, mlp):
        self.mlp = mlp

    def func(self, X, Y):
        loss = 0
        for x, y in zip(X, Y):
            output = self.mlp.feedforward(x)
            loss += np.sum((output - [1 if i == y else 0 for i in range(self.mlp.nb_outputs)]) ** 2)
        loss *= 1 / (2 * len(Y))
        return loss

    def prime(self, X, Y):
        expected = [1 if i == Y else 0 for i in range(self.mlp.nb_outputs)]
        return self.mlp.feedforward(X) - expected


class CrossEntropy:
    def __init__(self, mlp):
        self.mlp = mlp

    def func(self, X, Y):
        loss = 0
        for x, y in zip(X, Y):
            output = self.mlp.feedforward(x)
            expected = np.array([1 if i == y else 0 for i in range(self.mlp.nb_outputs)])
            loss += np.sum(expected * np.log(output) + (1 - expected) * np.log(1 - output))
        loss *= -1 / (2 * len(Y))
        return loss

    def prime(self, X, Y):
        return self.mlp.feedforward(X) - [1 if i == Y else 0 for i in range(self.mlp.nb_outputs)]
