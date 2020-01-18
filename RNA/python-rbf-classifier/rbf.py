from scipy import *
from scipy.linalg import norm, pinv
from sklearn.model_selection import KFold
from matplotlib import pyplot as plt
import numpy as np
from matplotlib.colors import ListedColormap


class RBF:

    def __init__(self, output_layer, attributes, num_centers=10, sigma=10):
        self.attributes = attributes
        self.output_layer = output_layer
        self.num_centers = num_centers
        a = 0
        b = 1
        self.centers = (b - a) * np.random.random_sample((num_centers, self.attributes + 1)) + a
        self.sigma = sigma
        self.W = random.random((self.num_centers, self.output_layer))

    def basis_function(self, c, d):
        # assert len(d) == self.attributes
        # return exp(-self.beta * norm(c - d) ** 2)
        # return np.exp(-0.5 * np.dot(aux, aux.T) / (width^2))
        return math.exp(-math.pow(norm(c - d), 2) / math.pow(2 * self.sigma, 2))

    def activation_calc(self, X):
        # calculate activations of RBFs
        G = zeros((X.shape[0], self.num_centers), float)
        for ci, c in enumerate(self.centers):
            for xi, x in enumerate(X):
                G[xi, ci] = self.basis_function(c, x)
        return G

    def activation_function(self, u):
        value = np.amax(u)
        y = np.where(u == value, 1, 0)
        return y

    def train(self, X, Y):
        # choose random center vectors from training set
        rnd_idx = random.permutation(X.shape[0])[:self.num_centers]
        self.centers = [X[i, :] for i in rnd_idx]

        # calculate activations of RBFs
        G = self.activation_calc(X)

        # calculate output weights (pseudoinverse)
        self.W = dot(pinv(G), Y)

    def predict(self, X):
        G = self.activation_calc(X)
        Y = dot(G, self.W)

        for index, y in enumerate(Y):
            y_new = self.activation_function(y)
            Y[index, :] = y_new

        return Y

    def evaluate(self, test_y, predictions):
        correct = 0
        for i in range(len(test_y)):
            if np.array_equal(test_y[i], predictions[i]):
                correct += 1
        return correct / float(len(test_y))  # * 100.0

    # return the best number of hidden neurons for each arrangement of the data
    @staticmethod
    def model_training(no_of_classes, no_of_attributes, X, y):
        X = np.array(X)
        y = np.array(y)
        centers = [10, 12, 14, 16, 18, 20]
        sigmas = [10, 12, 14, 16, 18, 20]
        grid_accuracy = np.zeros((6, 6))
        for index_s, sigma in enumerate(sigmas):  # range from 10 to 20
            # print("hidden neurons: {}".format(sigma))
            for index_c, center in enumerate(centers):
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

        ind = np.unravel_index(np.argmax(grid_accuracy, axis=None), grid_accuracy.shape)
        return sigmas[ind[0]], centers[ind[1]]

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


if __name__ == '__main__':

    # ----- 1D Example ------------------------------------------------
    n = 100

    x = mgrid[-1:1:complex(0, n)].reshape(n, 1)
    # set y and add random noise
    y = sin(3 * (x + 0.5) ** 3 - 1)
    # y += random.normal(0, 0.1, y.shape)

    # rbf regression
    rbf = RBF(1, 10, 1)
    rbf.train(x, y)
    z = rbf.test(x)

    # plot original data
    plt.figure(figsize=(12, 8))
    plt.plot(x, y, 'k-')

    # plot learned model
    plt.plot(x, z, 'r-', linewidth=2)

    # plot rbfs
    plt.plot(rbf.centers, zeros(rbf.num_centers), 'gs')

    for c in rbf.centers:
        # RF prediction lines
        cx = arange(c - 0.7, c + 0.7, 0.01)
        cy = [rbf._basisfunc(array([cx_]), array([c])) for cx_ in cx]
        plt.plot(cx, cy, '-', color='gray', linewidth=0.2)

    plt.xlim(-1.2, 1.2)
    plt.show()
