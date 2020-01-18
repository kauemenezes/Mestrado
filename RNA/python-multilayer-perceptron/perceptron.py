import numpy as np
from classifier import Classifier
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap


class Perceptron(Classifier):

    def __init__(self, no_of_classes, no_of_inputs, no_hidden_neurons, activation, epochs=500, eta=0.1):
        self.precision = 10 ** (-7)
        self.activation = activation
        self.epochs = epochs
        self.eta = eta
        # self.ww = 0.1 * np.random.rand(no_of_inputs + 1, no_hidden_neurons)
        # self.mm = 0.1 * np.random.rand(no_hidden_neurons + 1, no_of_classes)
        aux_1 = -1.5
        aux_2 = 1.5
        self.ww = (aux_2 - aux_1) * np.random.random_sample((no_of_inputs + 1, no_hidden_neurons)) + aux_1
        self.mm = (aux_2 - aux_1) * np.random.random_sample((no_hidden_neurons + 1, no_of_classes)) + aux_1

    def calculate_class(self, test_x):
        u_h = np.dot(test_x.reshape(-1, 1).T, self.ww)
        H = 1.0 / (1.0 + np.exp(-u_h))
        H = np.c_[-1, H]

        u_y = np.dot(H, self.mm)
        Y = 1.0 / (1.0 + np.exp(-u_y))

        y = self.activation_function(Y)
        return y

    def activate_hidden_neurons(self, inputs):
        u = np.dot(inputs.reshape(-1, 1).T, self.ww)
        H = 1.0 / (1.0 + np.exp(-u))
        H_ = H * (1.0 - H)

        return H, H_

    def activate_output_neurons(self, H):
        u = np.dot(H, self.mm)
        Y = 1.0 / (1.0 + np.exp(-u))
        Y_ = Y * (1.0 - Y)

        return Y, Y_

    def updateEta(self, epoch):
        eta_start = 0.1
        eta_end = 0.05
        eta = eta_start * ((eta_end / eta_start) ** (epoch / self.epochs))
        self.eta = eta

    def train(self, training_inputs, labels):
        old_error = 0
        for i in range(self.epochs):
            self.updateEta(i)
            epoch_error = 0
            for inputs, label in zip(training_inputs, labels):
                H, H_ = self.activate_hidden_neurons(inputs)
                H = np.c_[-1, H]
                Y, Y_ = self.activate_output_neurons(H)

                error = label - Y
                epoch_error += np.sum(error ** 2)

                # output layer
                output_gradient = error * Y_
                output_aux = self.eta * output_gradient
                self.mm += np.dot(H.T, output_aux)

                # hidden layer
                hidden_gradient = np.sum(np.dot(self.mm, output_gradient.T)) * H_
                hidden_multiplier = self.eta * hidden_gradient
                self.ww += np.dot(inputs.reshape(-1, 1), hidden_multiplier)

            if abs(epoch_error - old_error) <= self.precision:
                print('Stop Precision: {}'.format(abs(epoch_error - old_error)))
                break

            old_error = epoch_error

    def activation_function(self, u):
        if self.activation == 'step':
            index = np.amax(u)
            y = np.where(u == index, 1, 0)
            y_ = 1
            return y, y_

        elif self.activation == 'logistic':
            # u = 1.0 / (1.0 + np.exp(np.array(-u, dtype=float)))
            value = np.nanmax(u)
            y = np.where(u == value, 1, 0)
            # y_ = u * (1.0 - u)
            return y.ravel()

        elif self.activation == 'tanh':
            u = (np.exp(u) - np.exp(-u)) / (np.exp(u) + np.exp(-u))
            value = np.nanmax(u)
            y = np.where(u == value, 1, -1)
            y_ = 0.5 * (1.0 - (u * u))
            return y, y_

    # return the best number of hidden neurons for each arrangement of the data
    @staticmethod
    def model_training(no_of_classes, no_of_attributes, X, y):
        X = np.array(X)
        y = np.array(y)
        hidden_neurons = {}
        for n in range(4, 14, 2):  # range from 3 to 25
            print("hidden neurons: {}".format(n))
            kf = KFold(n_splits=5)
            accuracies = []
            for train_index, test_index in kf.split(X):
                perceptron = Perceptron(no_of_classes, no_of_attributes, n, 'logistic')
                train_X, test_X = X[train_index], X[test_index]
                train_y, test_y = y[train_index], y[test_index]

                perceptron.train(train_X, train_y)
                predictions = perceptron.predict(test_X)
                accuracies.append(perceptron.evaluate(predictions, test_y))
            hidden_neurons[n] = np.mean(accuracies)

        hidden_neurons = sorted(hidden_neurons.items(), key=lambda p: p[1], reverse=True)
        chosen_neurons = next(iter(hidden_neurons))[0]
        return chosen_neurons

    @staticmethod
    def plot_decision_boundaries(train_X, train_y, test_X, test_y, model, hidden_neurons, figure_index):
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
