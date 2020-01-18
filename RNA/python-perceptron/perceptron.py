import numpy as np
from classifier import Classifier


class Perceptron(Classifier):

    def __init__(self, no_of_inputs, epochs=100, learning_rate=0.01):
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros(no_of_inputs + 1)

    def calculate_class(self, inputs):
        summation = np.dot(inputs, self.weights[1:]) + self.weights[0]
        if summation > 0:
            prediction = 1
        else:
            prediction = 0
        return prediction

    def train(self, training_inputs, labels):
        # dictionary to store sum of errors in each epoch
        dict = {}
        for i in range(self.epochs):
            error = 0
            for inputs, label in zip(training_inputs, labels):
                prediction = self.calculate_class(inputs)
                self.weights[1:] += self.learning_rate * (label - prediction) * inputs
                # update bias
                self.weights[0] += self.learning_rate * (label - prediction)
                error += abs(label - prediction)
            dict[i + 1] = error
        return dict
