from Classifier import Classifier
import numpy as np
import math


class BayesLDAClassifier(Classifier):
    priorProbabilities = {}
    means = {}
    covInvMatrix = []

    def fit(self, train_x, train_y):
        self.calculate_attributes(train_x, train_y)

    def calculate_attributes(self, train_x, train_y):
        # identity_matrix = np.identity(np.array(train_x).shape[1])
        # dirty_matrix = 0.00001 * identity_matrix
        separated = self.separate_by_class(train_x, train_y)
        for classValue, instances in separated.items():
            self.priorProbabilities[int(classValue)] = len(instances) / float(len(train_x))
            self.means[int(classValue)] = np.mean(np.array(instances), axis=0)

        cov_matrix = np.cov(np.array(train_x).T)
        # cov_matrix = cov_matrix + dirty_matrix
        self.covInvMatrix = np.linalg.inv(cov_matrix)

    def calculate_probability(self, prior_probability, means, input_vector):
        # np.dot product of two arrays
        probability = - 0.5 * (np.dot(np.dot((input_vector - means).reshape(1, -1), self.covInvMatrix),
                                      (input_vector - means).reshape(-1, 1))) + math.log(prior_probability)

        return probability

    def calculate_class_probabilities(self, input_vector):
        probabilities = {}
        for classValue, priorProbability in self.priorProbabilities.items():
            probabilities[classValue] = 1
            probabilities[classValue] *= self.calculate_probability(priorProbability, self.means[classValue],
                                                                    input_vector)
        return probabilities
