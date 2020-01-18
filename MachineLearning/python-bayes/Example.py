from Classifier import Classifier
import numpy as np
import math


class Example(Classifier):
    priorProbabilities = {}
    means = {}
    determinants = {}
    covInvMatrices = {}

    def fit(self, train_x, train_y):
        self.calculate_attributes(train_x, train_y)

    def calculate_attributes(self, train_x, train_y):
        identity_matrix = np.identity(np.array(train_x).shape[1])
        dirty_matrix = 0.00001 * identity_matrix
        separated = self.separate_by_class(train_x, train_y)
        for classValue, instances in separated.items():
            self.priorProbabilities[int(classValue)] = len(instances) / float(len(train_x))
            self.means[int(classValue)] = np.mean(np.array(instances), axis=0)
            cov_matrix = np.cov(np.array(instances).T)
            cov_matrix = cov_matrix + dirty_matrix
            self.determinants[int(classValue)] = np.linalg.det(cov_matrix)
            self.covInvMatrices[int(classValue)] = np.linalg.inv(cov_matrix)

    def calculate_probability(self, prior_probability, components, means, determinant, cov_inv_matrix, input_vector):
        # np.dot product of two arrays
        exponent = math.exp(-1 / 2 * np.dot(np.dot((input_vector - means).reshape(1, -1), cov_inv_matrix),
                                            (input_vector - means).reshape(-1, 1)))
        conditional_probability = exponent / (math.pow(2 * math.pi, components / 2) * math.sqrt(determinant))
        return conditional_probability * prior_probability

    def calculate_class_probabilities(self, input_vector):
        probabilities = {}
        for classValue, priorProbability in self.priorProbabilities.items():
            probabilities[classValue] = 1
            probabilities[classValue] *= self.calculate_probability(priorProbability, len(input_vector),
                                                                    self.means[classValue],
                                                                    self.determinants[classValue],
                                                                    self.covInvMatrices[classValue], input_vector)
        return probabilities
