from Classifier import Classifier
import numpy as np
import math


class BayesRejClassifier(Classifier):
    priorProbabilities = {}
    means = {}
    determinants = {}
    covInvMatrices = {}
    wr = 1

    def fit(self, train_x, train_y):
        self.calculate_attributes(train_x, train_y)

    def calculate_attributes(self, train_x, train_y):
        separated = self.separate_by_class(train_x, train_y)
        for classValue, instances in separated.items():
            self.priorProbabilities[int(classValue)] = len(instances) / float(len(train_x))
            self.means[int(classValue)] = np.mean(np.array(instances), axis=0)
            cov_matrix = np.cov(np.array(instances).T)
            self.determinants[int(classValue)] = np.linalg.det(cov_matrix)
            self.covInvMatrices[int(classValue)] = np.linalg.inv(cov_matrix)

    def calculate_probability(self, prior_probability, components, means, determinant, cov_inv_matrix, input_vector):
        numerator = self.calculate_conditional_probability(components, means, determinant,
                                                           cov_inv_matrix, input_vector) * prior_probability
        summation = 0
        for classValue, prior in self.priorProbabilities.items():
            summation += prior * self.calculate_conditional_probability(len(input_vector),
                                                                        self.means[classValue],
                                                                        self.determinants[classValue],
                                                                        self.covInvMatrices[classValue],
                                                                        input_vector)
        return numerator/float(summation)

    def calculate_conditional_probability(self, components, means, determinant, cov_inv_matrix, input_vector):
        # np.dot product of two arrays
        exponent = math.exp(-1 / 2 * np.dot(np.dot((input_vector - means).reshape(1, -1), cov_inv_matrix),
                                            (input_vector - means).reshape(-1, 1)))
        conditional_probability = exponent / (math.pow(2 * math.pi, components / 2) * math.sqrt(determinant))
        return conditional_probability

    def calculate_class_probabilities(self, input_vector):
        probabilities = {}
        for classValue, priorProbability in self.priorProbabilities.items():
            probabilities[classValue] = 1
            probabilities[classValue] *= self.calculate_probability(priorProbability, len(input_vector),
                                                                    self.means[classValue],
                                                                    self.determinants[classValue],
                                                                    self.covInvMatrices[classValue], input_vector)
        return probabilities

    def calculate_class(self, input_vector):
        probabilities = self.calculate_class_probabilities(input_vector)
        best_label, best_prob = None, -1
        for classValue, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = classValue
        if best_prob < (1-self.wr):
            best_label = 0

        return best_label

    def set_wr(self, value):
        self.wr = value

    def evaluate(self, test_y, predictions):
        correct = 0
        for i in range(len(test_y)):
            if int(test_y[i]) == predictions[i]:
                correct += 1
        total = len(np.array(test_y)[np.array(predictions) != 0])
        total = total if total > 0 else len(test_y)
        rejected = len(np.array(test_y)[np.array(predictions) == 0])
        accuracy = correct / float(total)
        rejection = rejected/float(len(test_y))
        return accuracy, rejection
