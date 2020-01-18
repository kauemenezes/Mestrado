from scipy import *
from scipy._lib.six import xrange
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

        for index, y in enumerate(Y):
            y_new = self.activation_function(y)
            Y[index, :] = y_new

        return Y

    def train(self, X, Y):
        X = self.input_to_hidden(X)
        Xt = np.transpose(X)
        self.Wout = np.dot(np.linalg.pinv(np.dot(Xt, X)), np.dot(Xt, Y))

    def evaluate(self, test_y, predictions):
        correct = 0
        for i in range(len(test_y)):
            if np.array_equal(test_y[i], predictions[i]):
                correct += 1
        return correct / float(len(test_y))  # * 100.0

    def confusion_matrix(self, test_y, predictions):
        amount = np.array(test_y).shape[1]
        confusion_matrix = np.zeros((amount, amount), dtype=int)
        for i in range(len(predictions)):
            row = np.where(np.array(predictions[i]) == 1)[0][0]
            col = np.where(np.array(test_y[i]) == 1)[0][0]
            confusion_matrix[row, col] += 1

        return confusion_matrix

    @staticmethod
    def plot_decision_boundaries(train_X, train_y, test_X, test_y, model, figure_index):
        h = .02  # step size in the mesh
        train_X = train_X[:, 1:]
        test_X = test_X[:, 1:]

        # we only take two features.
        # train_X = np.array(train_X)[:, [0, 1]]
        # test_X = np.array(test_X)[:, [0, 1]]
        # train_X = np.array(train_X)[:, [2, 3]]
        # test_X = np.array(test_X)[:, [2, 3]]

        # no_rows = train_X.shape[0]
        # train_X = np.c_[-1 * np.ones(no_rows), train_X]
        # perceptron = Perceptron(no_of_classes, no_of_attributes, hidden_neurons, 'logistic')
        # perceptron.train(train_X, train_y)

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Create legend labels
        # labels = ['Iris Setosa (treino)', 'Iris Versicolor (treino)', 'Iris Virginica (treino)',
        #           'Iris Setosa (teste)', 'Iris Versicolor (teste)', 'Iris Virginica (teste)']
        labels = ['classe 1 (treino)', 'classe 2 (treino)',
                  'classe 1 (teste)', 'classe 2 (teste)']

        # Create axis labels
        # axis_labels = ['comprimento da pétala', 'largura da pétala']
        axis_labels = ['atributo 1', 'atributo 2']

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = np.array(test_X)[:, 0].min() - 1, np.array(test_X)[:, 0].max() + 1
        y_min, y_max = np.array(test_X)[:, 1].min() - 1, np.array(test_X)[:, 1].max() + 1

        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()]
        no_rows = Z.shape[0]
        Z = np.c_[-1 * np.ones(no_rows), Z]

        Z = model.predict(Z)
        # class_two = np.array([np.array_equal(row, [0, 0, 1]) for row in Z])
        # class_one = np.array([np.array_equal(row, [0, 1, 0]) for row in Z])
        # class_zero = np.array([np.array_equal(row, [1, 0, 0]) for row in Z])
        # Z[class_zero] = 0
        # Z[class_one] = 1
        # Z[class_two] = 2

        class_one = np.array([np.array_equal(row, [0, 1]) for row in Z])
        class_zero = np.array([np.array_equal(row, [1, 0]) for row in Z])
        Z[class_zero] = 0
        Z[class_one] = 1

        Z = np.array(Z, dtype=int)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figure_index)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        # train_two = np.array([np.array_equal(row, [0, 0, 1]) for row in train_y])
        # train_one = np.array([np.array_equal(row, [0, 1, 0]) for row in train_y])
        # train_zero = np.array([np.array_equal(row, [1, 0, 0]) for row in train_y])
        #
        # test_two = np.array([np.array_equal(row, [0, 0, 1]) for row in test_y])
        # test_one = np.array([np.array_equal(row, [0, 1, 0]) for row in test_y])
        # test_zero = np.array([np.array_equal(row, [1, 0, 0]) for row in test_y])

        train_one = np.array([np.array_equal(row, [0, 1]) for row in train_y])
        train_zero = np.array([np.array_equal(row, [1, 0]) for row in train_y])

        test_one = np.array([np.array_equal(row, [0, 1]) for row in test_y])
        test_zero = np.array([np.array_equal(row, [1, 0]) for row in test_y])

        # Remove bias column
        # train_X = train_X[:, 1:]

        # Plot also the training points
        plt.scatter(train_X[train_zero, 0], train_X[train_zero, 1],
                    label=labels[0], c='purple', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(train_X[train_one, 0], train_X[train_one, 1],
                    label=labels[1], c='lightgreen', cmap=cmap_bold, edgecolor='k', s=20)
        # plt.scatter(train_X[train_two, 0], train_X[train_two, 1],
        #             label=labels[2], c='blue', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(test_X[test_zero, 0], test_X[test_zero, 1],
                    label=labels[2], c='gray', cmap=cmap_bold, edgecolor='k', s=50)
        plt.scatter(test_X[test_one, 0], test_X[test_one, 1],
                    label=labels[3], c='orange', cmap=cmap_bold, edgecolor='k', s=50)
        # plt.scatter(test_X[test_two, 0], test_X[test_two, 1],
        #             label=labels[5], c='brown', cmap=cmap_bold, edgecolor='k', s=50)

        # plt.title("Número de neurônios ocultos: {}".format(hidden_neurons))
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.legend()
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

    @staticmethod
    def show_plot_decision_boundaries():
        plt.show()
