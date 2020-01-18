from Classifier import Classifier
import numpy as np
import numpy.matlib
import math
from scipy.spatial.distance import cdist


class LSSVM(Classifier):
    priorProbabilities = {}
    separated = {}
    h = 0.05
    params = {}
    train_x = []
    train_y = []
    bias = []
    alphas = []

    def fit(self, train_x, train_y, params):
        # self.calculate_attributes(train_x, train_y)
        self.train_x = train_x
        self.train_y = train_y
        self.params = params

    def train(self):
        # getting A.x = b problem
        A = np.multiply(np.dot(self.train_x, self.train_x.T), np.dot(self.train_y.reshape(-1, 1),
                                                                     self.train_y.reshape(1, -1)))
        A = A + np.multiply((1 / self.params['gamma']), np.identity(self.train_x.shape[0]))
        A = np.concatenate((self.train_y.reshape(1, -1), A), axis=0)
        A = np.concatenate((np.concatenate(([0], self.train_y), axis=0).reshape(-1, 1), A), axis=1)
        aux = np.ones(len(self.train_y))
        b = np.concatenate(([0], aux), axis=0)

        # make lssvm classifier
        x = np.linalg.lstsq(A, b.reshape(-1, 1))
        self.bias.append(x[0][0])
        self.alphas = x[0][1:]

    def classify(self, test_x):
        K = cdist(test_x.reshape(1, -1), self.train_x)
        K = np.exp(-(K ** 2) / self.params['sigma'] ** 2)

        # real = np.sum(np.multiply(K, np.multiply(self.alphas.T, self.train_y.reshape(1, -1)))) + self.bias[0]
        instance_class = numpy.sign(np.sum(np.multiply(K, np.multiply(self.alphas.T, self.train_y.reshape(1, -1)))) +
                                    self.bias[0])
        return instance_class
