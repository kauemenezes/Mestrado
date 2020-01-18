from Classifier import Classifier
import numpy as np
import numpy.matlib
import math


class BayesParzenWinClassifier(Classifier):
    priorProbabilities = {}
    separated = {}
    h = 0.05

    def fit(self, train_x, train_y, h):
        self.calculate_attributes(train_x, train_y)
        self.h = h

    def calculate_attributes(self, train_x, train_y):
        # identity_matrix = np.identity(np.array(train_x).shape[1])
        # dirty_matrix = 0.00001 * identity_matrix
        self.separated = self.separate_by_class(train_x, train_y)
        for classValue, instances in self.separated.items():
            self.priorProbabilities[int(classValue)] = len(instances) / float(len(train_x))

    def calculate_probability(self, test_vector, data):
        # np.dot product of two arrays
        N, components = data.shape
        conditional_probability = np.sum(
            (1 / (math.pow(2 * math.pi, components / 2)) * math.pow(self.h, components)) *
            np.exp(-np.diag(np.dot((data - np.matlib.repmat(test_vector, N, 1)),
                              (data - np.matlib.repmat(test_vector, N, 1)).T)) / (2 * math.pow(self.h, 2)))) / N
        return conditional_probability

    def calculate_class_probabilities(self, input_vector):
        probabilities = {}
        for classValue, priorProbability in self.priorProbabilities.items():
            probabilities[classValue] = 1
            probabilities[classValue] *= priorProbability * self.calculate_probability(input_vector,
                                                                                       np.array(
                                                                                           self.separated[classValue]))

        return probabilities
