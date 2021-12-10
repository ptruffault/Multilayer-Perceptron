import matplotlib.pyplot as plt
from tqdm import tqdm
import json
import numpy as np
from mlib.LayerPerceptron import LayerPerceptron as Layer
from mlib.LossFunction import *
import random


class MultipleLayerPerceptron:
    def __init__(self, dataset, filepath=None, structure=None, loss="MeanSquaredError", activations=None):
        self.dataset = dataset
        if filepath is not None:
            self.load(filepath)
        else:
            self.nb_inputs = dataset.nb_features
            self.nb_hidden = len(structure)
            self.nb_outputs = dataset.nb_classes
            self.structure = [self.nb_inputs] + structure + [self.nb_outputs]
            self.size = len(self.structure)
            self.learning_curve = {'accuracy': [], 'loss': []}
            self.activations = ["Sigmoid"] * self.size if activations is None else activations
            self.build(loss)
            self.accuracy = 0
            self.loss = 1
            self.dataset.scale()
            self.dataset.train_test_split(80)
            self.X = dataset.X
            self.Y = dataset.Y
            self.X_train = self.dataset.X_train
            self.Y_train = self.dataset.Y_train
            self.X_test = self.dataset.X_test
            self.Y_test = self.dataset.Y_test
        self.A = None
        self.Z = None

    def build(self, loss_function=None):
        if loss_function == "MeanSquaredError":
            self.loss_function = MeanSquaredError(self)
        else:
            self.loss_function = CrossEntropy(self)
        self.layers = []
        for i in range(self.size):
            self.layers.append(Layer(self.structure[i], self.nb_inputs if not i else self.layers[i - 1].size , self.activations[i]))

    def predict(self, x):
        return np.argmax(self.feedforward(x))

    def predictions(self, X):
        return [self.predict(x) for x in X]

    def get_loss(self):
        self.loss = self.loss_function.func(self.X_test, self.Y_test)
        self.learning_curve['loss'].append(self.loss)

    def get_accuracy(self):
        self.accuracy = sum([1 if self.predict(x) == y else 0 for (x, y) in zip(self.X_test, self.Y_test)]) / self.Y_test.size * 100
        self.learning_curve['accuracy'].append(self.accuracy)

    def feedforward(self, X):
        a = X
        self.A = [a]
        self.Z = []
        for layer in self.layers:
            z = layer.aggregation(a)
            a = layer.activation(z)
            self.A.append(a)
            self.Z.append(z)
        return a

    def feedbackward(self, x, y):
        # initialise error
        delta = self.loss_function.prime(x, y)
        deltas = [delta]
        # gradiant back propagation
        for l in reversed(range(self.size - 1)):
            delta = self.layers[l].activation_prime(self.Z[l]) * np.dot(self.layers[l + 1].weights.transpose(), delta)
            deltas.append(delta)
        deltas = list(reversed(deltas))
        weight_gradient = [np.outer(deltas[l], self.A[l]) for l in range(self.size)]
        bias_gradient = [deltas[l] for l in range(self.size)]
        return weight_gradient, bias_gradient

    # train a batch of exemple
    def train_batch(self, X, Y, learning_rate):
        weight_gradient = [np.zeros(layer.weights.shape) for layer in self.layers]
        bias_gradient = [np.zeros(layer.biases.shape) for layer in self.layers]
        for (x, y) in zip(X, Y):
            new_weight_gradient, new_bias_gradient = self.feedbackward(x, y)
            weight_gradient = [wg + nwg for wg, nwg in zip(weight_gradient, new_weight_gradient)]
            bias_gradient = [bg + nbg for bg, nbg in zip(bias_gradient, new_bias_gradient)]
        avg_weight_gradient = [wg / Y.size for wg in weight_gradient]
        avg_bias_gradient = [bg / Y.size for bg in bias_gradient]
        # update weights and bias
        for layer, weight_gradient, bias_gradient in zip(self.layers, avg_weight_gradient, avg_bias_gradient):
            layer.update_weights(weight_gradient, learning_rate)
            layer.update_biases(bias_gradient, learning_rate)

    def train(self, iterations=100, learning_rate=0.1, batch_size=4, plot=False):
        if plot:
            fig, (axs) = plt.subplots(3)
            self.show(fig, axs[0])
        for epoch in tqdm(range(iterations)):
            self.dataset.shuffle(self.X_train, self.Y_train)
            for batch_start in range(0, self.Y_train.size, batch_size):
                self.train_batch(self.X_train[batch_start:batch_start + batch_size], self.Y_train[batch_start:batch_start + batch_size], learning_rate)
            if not epoch % 5:
                self.get_accuracy()
                self.get_loss()
                if plot:
                    self.train_plot(fig, axs)


    def train_plot(self, fig, axs):
        axs[1].set_title('Accuracy')
        axs[1].plot(self.learning_curve['accuracy'])
        axs[2].set_title('Loss')
        axs[2].plot(self.learning_curve['loss'])
        plt.pause(0.01)

    def show(self, fig, ax):
        radius = 20
        space = 2 * radius + 5
        ax.clear()
        ax.set_title('Layers structure')
        maxLayerSize = None
        for l in self.layers:
            maxLayerSize = l.size if maxLayerSize is None or l.size > maxLayerSize else maxLayerSize
        for i in range(self.size):
            margin = (maxLayerSize - self.layers[i].size) * (radius + space) / 2
            for j in range(self.layers[i].size):
                x = margin + j * (radius + space)
                y = i * (radius + space)
                circle = plt.Circle((x, y), radius=radius, fill=False)
                ax.add_patch(circle)
                if i > 0:
                    for k in range(self.layers[i - 1].size):
                        x2 = prev_margin + k * (radius + space)
                        y2 = (i - 1) * (radius + space)
                        line = plt.Line2D((x, x2), (y, y2), linewidth=0.05)
                        ax.add_patch(line)
            prev_margin = margin
        ax.set_aspect('equal')
        ax.axis('scaled')
        ax.autoscale_view()
        ax.axis('off')


    def save(self, filepath, dataset):
        with open(filepath, 'w+') as jsonfile:
            json.dump({
                'nb_inputs':        self.nb_inputs,
                'nb_hidden':        self.nb_hidden,
                'nb_outputs':       self.nb_outputs,
                'structure':        self.structure,
                'activations':      self.activations,
                'loss_function':    self.loss_function.__class__.__name__,
                'weights':          [layer.weights.tolist() for layer in self.layers],
                'biases':           [layer.biases.tolist() for layer in self.layers],
                'learning_curve':   self.learning_curve,
                'accuracy':         self.accuracy,
                'loss':             self.loss,
                'minmax':           dataset.minmax
            }, jsonfile)

    def load(self, filepath):
        with open(filepath) as jsonfile:
            data = json.load(jsonfile)
            self.nb_inputs = data['nb_inputs']
            self.nb_hidden = data['nb_hidden']
            self.nb_outputs = data['nb_outputs']
            self.structure = data['structure']
            self.size = self.nb_hidden + 2
            self.activations = data['activations']
            self.learning_curve = data['learning_curve']
            self.build(data['loss_function'])
            for layer, weights, biases in zip(self.layers, data['weights'], data['biases']):
                layer.weights = np.array(weights)
                layer.biases = np.array(biases)
            self.dataset.scale(data['minmax'])
            self.X = self.dataset.X
            self.Y = self.dataset.Y
            self.dataset.train_test_split(80)
            self.X_train = self.dataset.X_train
            self.Y_train = self.dataset.Y_train
            self.X_test = self.dataset.X_test
            self.Y_test = self.dataset.Y_test

