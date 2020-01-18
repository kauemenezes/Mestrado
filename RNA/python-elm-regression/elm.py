from scipy import *
from scipy.linalg import norm, pinv
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class ELM:

    def __init__(self, output_layer, attributes, hidden_units=100):
        self.attributes = attributes
        self.output_layer = output_layer
        self.hidden_units = hidden_units
        self.Win = np.random.normal(size=(self.attributes + 1, self.hidden_units))
        self.Wout = []

    @staticmethod
    def model_training(no_of_classes, no_of_attributes, X, y):
        X = np.array(X)
        y = np.array(y)
        hidden_neurons = {}
        for n in range(50, 110, 10):  # range from 10 to 50
            # print("hidden neurons: {}".format(n))
            kf = KFold(n_splits=5)
            accuracies = []
            for train_index, test_index in kf.split(X):
                perceptron = ELM(no_of_classes, no_of_attributes, n)
                train_X, test_X = X[train_index], X[test_index]
                train_y, test_y = y[train_index], y[test_index]

                perceptron.train(train_X, train_y)
                predictions = perceptron.predict(test_X)
                accuracies.append(perceptron.evaluate(predictions, test_y))
            hidden_neurons[n] = np.mean(accuracies)

        hidden_neurons = sorted(hidden_neurons.items(), key=lambda p: p[1], reverse=True)
        chosen_neurons = next(iter(hidden_neurons))[0]
        return chosen_neurons

    def input_to_hidden(self, x):
        a = np.dot(x, self.Win)
        a = np.maximum(a, 0, a)
        return a

    def activation_function(self, u):
        value = np.amax(u)
        y = np.where(u == value, 1, 0)
        return y

    def predict(self, x):
        x = self.input_to_hidden(x)
        Y = np.dot(x, self.Wout)

        return Y

    def train(self, X, Y):
        X = self.input_to_hidden(X)
        Xt = np.transpose(X)
        self.Wout = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, Y))

    def evaluate(self, test_y, predictions):
        squared_error = 0.0
        for i in range(len(test_y)):
            error = test_y[i] - predictions[i]
            squared_error += error ** 2

        mse = squared_error / len(test_y)
        rmse = np.sqrt(mse)
        return mse, rmse

    def confusion_matrix(self, test_y, predictions):
        amount = np.array(test_y).shape[1]
        confusion_matrix = np.zeros((amount, amount), dtype=int)
        for i in range(len(predictions)):
            row = np.where(np.array(predictions[i]) == 1)[0][0]
            col = np.where(np.array(test_y[i]) == 1)[0][0]
            confusion_matrix[row, col] += 1

        return confusion_matrix

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
