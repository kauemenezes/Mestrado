from scipy import *
from scipy._lib.six import xrange
from scipy.linalg import norm, pinv
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np


class RBF:

    def __init__(self, output_layer, attributes, num_centers=16, sigma=12):
        self.attributes = attributes
        self.output_layer = output_layer
        self.num_centers = num_centers
        # self.centers = [random.uniform(-1, 1, attributes) for i in xrange(num_centers)]
        a = 0
        b = 1
        self.centers = (b - a) * np.random.random_sample((num_centers, self.attributes + 1)) + a
        self.sigma = sigma
        self.W = random.random((self.num_centers, self.output_layer))

    def basis_function(self, c, d):
        return math.exp(-math.pow(norm(c - d), 2) / math.pow(2 * self.sigma, 2))

    def activation_function(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.num_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self.basis_function(c, x)
        return G

    def train(self, X, Y):
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rnd_idx]

        # calculate activations of RBFs
        G = self.activation_function(X)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def predict(self, X):

        G = self.activation_function(X)
        Y = dot(G, self.W)
        return Y

    def evaluate(self, test_y, predictions):
        squared_error = 0.0
        for i in range(len(test_y)):
            error = test_y[i] - predictions[i]
            squared_error += error ** 2

        mse = squared_error / len(test_y)
        rmse = np.sqrt(mse)
        return mse, rmse

    # return the best number of hidden neurons for each arrangement of the data
    @staticmethod
    def model_training(no_of_classes, no_of_attributes, X, y):
        X = np.array(X)
        y = np.array(y)
        hidden_neurons = {}
        grid_accuracy = np.zeros((6, 6))
        grid_range = range(10, 22, 2)
        for index_s, sigma in enumerate(grid_range):  # range from 10 to 20
            print("hidden neurons: {}".format(n))
            for index_c, center in enumerate(grid_range):
                kf = KFold(n_splits=5)
                hit_rates = []
                for train_index, test_index in kf.split(X):
                    rbf = RBF(no_of_classes, no_of_attributes, center, sigma)
                    train_X, test_X = X[train_index], X[test_index]
                    train_y, test_y = y[train_index], y[test_index]

                    rbf.train(train_X, train_y)
                    predictions = rbf.predict(test_X)
                    hit_rates.append(rbf.evaluate(predictions, test_y))
                grid_accuracy[index_s, index_c] = np.mean(hit_rates)

        hidden_neurons = sorted(hidden_neurons.items(), key=lambda p: p[1], reverse=True)
        chosen_neurons = next(iter(hidden_neurons))[0]
        return chosen_neurons

    @staticmethod
    def plot_decision_boundaries_one(train_X, train_y, test_X, test_y, model, figure_index):
        h = .02  # step size in the mesh
        train_X = train_X[:, 1:]
        test_X = test_X[:, 1:]

        x_min, x_max = np.array(train_X).min() - 1, np.array(train_X).max() + 1
        x = np.arange(x_min, x_max, h)
        x = x.reshape(-1, 1)
        no_rows = x.shape[0]
        x = np.c_[-1 * np.ones(no_rows), x]
        y = model.predict(x)

        plt.figure(figure_index)
        fig, ax = plt.subplots()
        plt.title("Artificial I")
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        x = x[:, 1:]
        ax.scatter(x, y, label='Predict', color='#FFAAAA')
        ax.scatter(train_X, train_y, label='Train Data', color='#AAFFAA')
        ax.scatter(test_X, test_y, label='Test Data', color='#AAAAFF')

        ax.legend()
        ax.grid(True)

    @staticmethod
    def show_plot_decision_boundaries():
        plt.show()
