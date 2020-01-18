import numpy as np
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from abc import ABC, abstractmethod


class Classifier(ABC):

    def train_test_split(self, dataset, split_ratio=0.2):
        rows, columns = np.shape(dataset)
        columns = columns - 1

        # shuffle matrix
        indices = np.arange(rows)
        np.random.shuffle(indices)
        dataset = dataset[indices, :]

        # calculate split ratio
        test_ratio = int(len(dataset) * split_ratio)
        train_ratio = len(dataset) - test_ratio

        # separation of training and test data
        train_data = dataset[:train_ratio]
        test_data = dataset[train_ratio:]

        train_x = [data[:columns] for data in train_data]
        train_y = [data[columns] for data in train_data]
        test_x = [data[:columns] for data in test_data]
        test_y = [data[columns] for data in test_data]

        return train_x, train_y, test_x, test_y

    # @abstractmethod
    def fit(self, X, iterations, number_of_sources):
        pass

    def confusion_matrix(self, test_y, predictions):
        amount = len(np.unique(test_y))
        confusion_matrix = np.zeros((amount, amount), dtype=int)
        for i in range(len(predictions)):
            row = int(predictions[i])
            col = int(test_y[i])
            confusion_matrix[row, col] += 1

        return confusion_matrix

    def plot_decision_boundaries(self, train_X, train_y, test_X, test_y, figure_index):
        h = .04  # step size in the mesh

        # we only take two features.
        train_X = np.array(train_X)[:, [2, 3]]
        test_X = np.array(test_X)[:, [2, 3]]
        # train_X = np.array(train_X)[:, [0, 1]]
        # test_X = np.array(test_X)[:, [0, 1]]

        # Create color maps
        cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
        cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

        # Create legend labels
        labels = ['Iris Setosa (treino)', 'Iris Versicolor (treino)', 'Iris Virginica (treino)',
                  'Iris Setosa (teste)', 'Iris Versicolor (teste)', 'Iris Virginica (teste)']
        # labels = ['Hérnia de Disco (treino)', 'Espondilolistese (treino)', 'Normal (treino)',
        #           'Hérnia de Disco (teste)', 'Espondilolistese (teste)', 'Normal (teste)']
        # labels = ['classe 1 (treino)', 'classe 2 (treino)', 'classe 1 (teste)', 'classe 2 (teste)']

        # Create axis labels
        axis_labels = ['comprimento da pétala', 'largura da pétala']
        # axis_labels = ['incidência pélvica', 'inclinação pélvica']
        # axis_labels = ['atributo 1', 'atributo 2']

        # Plot the decision boundary. For that, we will assign a color to each
        # point in the mesh [x_min, x_max]x[y_min, y_max].
        x_min, x_max = np.array(test_X)[:, 0].min() - 1, np.array(test_X)[:, 0].max() + 1
        y_min, y_max = np.array(test_X)[:, 1].min() - 1, np.array(test_X)[:, 1].max() + 1
        xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                             np.arange(y_min, y_max, h))
        z = np.c_[xx.ravel(), yy.ravel()]

        separated_data = self.separate_by_class(train_X, train_y)
        model = {}
        for classValue, classData in separated_data.items():
            model[classValue] = []
            model[classValue] = (self.train_model(np.array(classData), 3, 50, 1))

        predictions = self.predict(z, model)

        # Put the result into a color plot
        predictions = np.array(predictions).reshape(xx.shape)
        plt.figure(figure_index)
        plt.pcolormesh(xx, yy, predictions, cmap=cmap_light)

        # Plot also the training points
        plt.scatter(train_X[np.array(train_y) == 0.0, 0], train_X[np.array(train_y) == 0.0, 1], label=labels[0],
                    c='purple', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(train_X[np.array(train_y) == 1.0, 0], train_X[np.array(train_y) == 1.0, 1], label=labels[1],
                    c='lightgreen', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(train_X[np.array(train_y) == 2.0, 0], train_X[np.array(train_y) == 2.0, 1], label=labels[2],
                    c='blue', cmap=cmap_bold, edgecolor='k', s=20)
        plt.scatter(test_X[np.array(test_y) == 0.0, 0], test_X[np.array(test_y) == 0.0, 1], label=labels[3],
                    c='gray', cmap=cmap_bold, edgecolor='k', s=50)
        plt.scatter(test_X[np.array(test_y) == 1.0, 0], test_X[np.array(test_y) == 1.0, 1], label=labels[4],
                    c='orange', cmap=cmap_bold, edgecolor='k', s=50)
        plt.scatter(test_X[np.array(test_y) == 2.0, 0], test_X[np.array(test_y) == 2.0, 1], label=labels[5],
                    c='brown', cmap=cmap_bold, edgecolor='k', s=50)
        plt.xlim(xx.min(), xx.max())
        plt.ylim(yy.min(), yy.max())
        plt.legend()
        plt.xlabel(axis_labels[0])
        plt.ylabel(axis_labels[1])

    def show_plot_decision_boundaries(self):
        plt.show()

    def separate_by_class(self, train_x, train_y):
        separated = {}
        for i in range(len(train_x)):
            vector = train_x[i]
            if train_y[i] not in separated:
                separated[train_y[i]] = []
            separated[train_y[i]].append(vector)
        return separated

    @abstractmethod
    def train_model(self, X, n_clusters, n_epochs, threshold):
        pass

    @abstractmethod
    def calculate_class_probabilities(self, input_vector, model):
        pass

    def calculate_class(self, input_vector, model):
        probabilities = self.calculate_class_probabilities(input_vector, model)
        best_label, best_prob = None, -1
        for classValue, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = classValue
        return best_label

    def predict(self, test_x, model):
        predictions = []
        for i in range(len(test_x)):
            result = self.calculate_class(test_x[i], model)
            predictions.append(result)
        return predictions

    def evaluate(self, test_y, predictions):
        correct = 0
        for i in range(len(test_y)):
            if int(test_y[i]) == predictions[i]:
                correct += 1
        return correct / float(len(test_y))  # * 100.0

    def min_max_normalizer(self, x, min_value, max_value):
        return (x - min_value) / (max_value - min_value) if (max_value - min_value) else 0  # avoid division by 0

    def normalize(self, data, min_data, max_data):
        norm_data = [self.min_max_normalizer(data[i], min_data[i], max_data[i]) for i in range(len(data))]
        return norm_data

    # data normalization
    def normalize_dataset(self, dataset):
        columns = np.shape(dataset)[1]
        columns = columns - 1
        min_data = dataset[:, 0:columns].min(axis=0)
        max_data = dataset[:, 0:columns].max(axis=0)
        normalized_features = [self.normalize(data, min_data, max_data) for data in dataset[:, 0:columns]]
        classes = dataset[:, columns].reshape(-1, 1)  # matrix transpose
        dataset = np.concatenate([normalized_features, classes], axis=1)

        return dataset
