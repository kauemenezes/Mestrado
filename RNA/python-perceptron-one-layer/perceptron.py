import numpy as np
from classifier import Classifier
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap


class Perceptron(Classifier):

    def __init__(self, no_of_classes, no_of_inputs, activation, epochs=100, learning_rate=0.01):
        self.activation = activation
        self.epochs = epochs
        self.learning_rate = learning_rate
        self.weights = np.zeros([no_of_inputs + 1, no_of_classes], dtype=object)

    def calculate_class(self, inputs):
        summation = np.dot(inputs.reshape(1, -1), self.weights[1:]) + self.weights[0]
        prediction, y_ = self.activation_function(summation.ravel())

        return prediction, y_

    def train(self, training_inputs, labels):
        # dictionary to store sum of errors in each epoch
        for i in range(self.epochs):
            for inputs, label in zip(training_inputs, labels):
                prediction, y_ = self.calculate_class(inputs)
                error = label - prediction
                self.weights[1:] += self.learning_rate * inputs.reshape(-1, 1) * (y_ * error).reshape(1, -1)
                # update bias
                self.weights[0] += self.learning_rate * y_ * error
                error += abs(label - prediction)

    def activation_function(self, u):
        if self.activation == 'step':
            index = np.amax(u)
            y = np.where(u == index, 1, 0)
            y_ = 1
            return y, y_

        elif self.activation == 'logistic':
            u = 1.0 / (1.0 + np.exp(np.array(-u, dtype=float)))
            value = np.nanmax(u)
            y = np.where(u == value, 1, 0)
            y_ = u * (1.0 - u)
            return y, y_

        elif self.activation == 'tanh':
            u = (np.exp(u) - np.exp(-u)) / (np.exp(u) + np.exp(-u))
            value = np.nanmax(u)
            y = np.where(u == value, 1, -1)
            y_ = 0.5 * (1.0 - (u * u))
            return y, y_

    @staticmethod
    def plot_decision_boundaries(train_X, train_y, test_X, test_y, figure_index):
        h = .02  # step size in the mesh

        # we only take two features.
        # train_X = np.array(train_X)[:, [0, 1]]
        # test_X = np.array(test_X)[:, [0, 1]]
        train_X = np.array(train_X)[:, [2, 3]]
        test_X = np.array(test_X)[:, [2, 3]]
        no_of_attributes = np.array(train_X).shape[1]
        no_of_classes = len(train_y[0])
        perceptron = Perceptron(no_of_classes, no_of_attributes, 'logistic')
        perceptron.train(train_X, train_y)

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Create legend labels
        labels = ['Iris Setosa (treino)', 'Iris Versicolor (treino)', 'Iris Virginica (treino)',
                  'Iris Setosa (teste)', 'Iris Versicolor (teste)', 'Iris Virginica (teste)']
        # labels = ['classe 1 (treino)', 'classe 2 (treino)', 'classe 3 (treino)',
        #           'classe 1 (teste)', 'classe 2 (teste)', 'classe 3 (teste)']

        # Create axis labels
        axis_labels = ['comprimento da pétala', 'largura da pétala']
        # axis_labels = ['atributo 1', 'atributo 2']

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = np.array(test_X)[:, 0].min() - 1, np.array(test_X)[:, 0].max() + 1
        y_min, y_max = np.array(test_X)[:, 1].min() - 1, np.array(test_X)[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        Z = np.c_[xx.ravel(), yy.ravel()]

        Z = perceptron.predict(Z)
        class_zero = np.array([np.array_equal(row, [0, 0, 1]) for row in Z])
        class_one = np.array([np.array_equal(row, [0, 1, 0]) for row in Z])
        class_two = np.array([np.array_equal(row, [1, 0, 0]) for row in Z])
        Z[class_zero] = 0
        Z[class_one] = 1
        Z[class_two] = 2
        Z = np.array(Z, dtype=int)

        # Put the result into a color plot
        Z = Z.reshape(xx.shape)
        plt.figure(figure_index)
        plt.pcolormesh(xx, yy, Z, cmap=cmap_light)

        train_zero = np.array([np.array_equal(row, [0, 0, 1]) for row in train_y])
        train_one = np.array([np.array_equal(row, [0, 1, 0]) for row in train_y])
        train_two = np.array([np.array_equal(row, [1, 0, 0]) for row in train_y])

        test_zero = np.array([np.array_equal(row, [0, 0, 1]) for row in test_y])
        test_one = np.array([np.array_equal(row, [0, 1, 0]) for row in test_y])
        test_two = np.array([np.array_equal(row, [1, 0, 0]) for row in test_y])

        # Plot also the training points
        plt.scatter(train_X[train_zero, 0], train_X[train_zero, 1],
                    label=labels[0], c='purple', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(train_X[train_one, 0], train_X[train_one, 1],
                    label=labels[1], c='lightgreen', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(train_X[train_two, 0], train_X[train_two, 1],
                    label=labels[2], c='blue', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(test_X[test_zero, 0], test_X[test_zero, 1],
                    label=labels[3], c='gray', cmap=cmap_bold, edgecolor='k', s=50)
        plt.scatter(test_X[test_one, 0], test_X[test_one, 1],
                    label=labels[4], c='orange', cmap=cmap_bold, edgecolor='k', s=50)
        plt.scatter(test_X[test_two, 0], test_X[test_two, 1],
                    label=labels[5], c='brown', cmap=cmap_bold, edgecolor='k', s=50)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.legend()
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

    @staticmethod
    def show_plot_decision_boundaries():
        plt.show()
