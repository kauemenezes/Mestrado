import numpy as np
import math
from mpl_toolkits import mplot3d
import matplotlib.pyplot as plt
from classifier import Classifier
from mpl_toolkits.mplot3d import Axes3D


class Adaline(Classifier):
    def __init__(self, learning_rate=0.001, epochs=100):
        self.learning_rate = learning_rate
        self.epochs = epochs

    def fit(self, no_of_inputs, X, y):
        self.weight_ = np.zeros(no_of_inputs + 1)
        self.error_ = []

        cost = 0
        for _ in range(self.epochs):

            output = self.activation_function(X)
            error = y.reshape(-1, 1) - output

            self.weight_[0] += self.learning_rate * sum(error)
            self.weight_[1:] += self.learning_rate * np.ravel(np.dot(X.T, error))

            cost = 1./2 * sum((error**2))
            self.error_.append(cost)

        return self

    def activation_function(self, X):
        return np.dot(X, np.array(self.weight_[1:]).reshape(-1, 1)) + self.weight_[0]

    def calculate_error(self, inputs, y):
        output = self.activation_function(inputs)
        error = y.reshape(-1, 1) - output
        self.mse_ = 1./2 * sum((error**2))
        self.rmse_ = math.sqrt(self.mse_)

    def calculate_outputs(self, X):
        return self.activation_function(X)

    def plot_decision_boundaries_one(self, train_X, train_y, test_X, test_y, figure_index):
        h = .02  # step size in the mesh
        x_min, x_max = np.array(train_X).min() - 1, np.array(train_X).max() + 1
        x = np.arange(x_min, x_max, h)
        y = self.predict(x.reshape(-1, 1))

        plt.figure(figure_index)
        fig, ax = plt.subplots()
        plt.title('Artificial I')
        plt.xlabel('Eixo X')
        plt.ylabel('Eixo y')

        ax.scatter(x, y, label='Predict', color='#FFAAAA')
        ax.scatter(train_X, train_y, label='Train Data', color='#AAFFAA')
        ax.scatter(test_X, test_y, label='Test Data', color='#AAAAFF')

        ax.legend()
        ax.grid(True)

    def plot_decision_boundaries_two(self, train_X, train_y, test_X, test_y, figure_index, dataset):
        X = dataset[:, 0]
        Y = dataset[:, 1]
        data = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
        Z = self.predict(data)

        plt3d = plt.figure().gca(projection='3d')
        # plt3d.hold(True)
        plt3d.scatter(np.array(train_X)[:, 0], np.array(train_X)[:, 1], train_y, label='Train Data', color='#00FF00')
        plt3d.scatter(np.array(test_X)[:, 0], np.array(test_X)[:, 1], test_y, label='Test Data', color='#0000FF')
        plt3d.plot_surface(X, Y, Z, linewidth=0.2, antialiased=True, color='#FFAAAA')
        plt.show()

    # def plot_decision_boundaries_two(self, train_X, train_y, test_X, test_y, figure_index, dataset):
    #     X = dataset[:, 0]
    #     Y = dataset[:, 1]
    #     data = np.concatenate([X.reshape(-1, 1), Y.reshape(-1, 1)], axis=1)
    #     Z = self.predict(data)
    #
    #     plt.figure(figure_index)
    #     fig = plt.figure()
    #     ax = Axes3D(fig)
    #     ax.plot_trisurf(X, Y, Z.ravel(), linewidth=0.2, antialiased=True, color='#FFAAAA')
    #     plt.title('Artificial II')
    #
    #     ax.scatter(np.array(train_X)[:, 0], np.array(train_X)[:, 1], train_y, label='Train Data', color='#00FF00')
    #     ax.scatter(np.array(test_X)[:, 0], np.array(test_X)[:, 1], test_y, label='Test Data', color='#0000FF')
    #
    #     ax.set_xlabel('X')
    #     ax.set_ylabel('Y')
    #     ax.set_zlabel('Z')

    @staticmethod
    def show_plot_decision_boundaries():
        plt.show()
