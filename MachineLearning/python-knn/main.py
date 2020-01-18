import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import KFold
from matplotlib.colors import ListedColormap
import random


def train_test_split(dataset, split_ratio=0.2):
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


def euclidean_distance(X, y):
    return np.sqrt(np.sum((X-y)**2, axis=1))


def k_nearest_neighbors(train_set, test_instance, k=5):
    distances = euclidean_distance(train_set, test_instance).tolist()
    return sorted(range(len(distances)), key=lambda i: distances[i])[:k]


def predict(train_y, k_neighbors):
    knn_labels = [train_y[i] for i in k_neighbors]

    return max(set(knn_labels), key=knn_labels.count)


def evaluate(predictions, test_y):
    accuracy = sum(i == j for i, j in zip(
        predictions, test_y)) / len(test_y)

    return accuracy


def min_max_normalizer(x, min, max):
    return (x - min) / (max - min) if (max - min) else 0  # avoid division by 0


def normalize(data, minData, maxData):
    norm_data = [min_max_normalizer(data[i], minData[i], maxData[i]) for i in range(len(data))]
    return norm_data


# return the best k value for each arrangement of the data
def model_training(X, y):
    X = np.array(X)
    y = np.array(y)
    k_values = {}
    for k in range(3, 27, 2):  # range from 3 to 25
        kf = KFold(n_splits=5)
        accuracies = []
        for train_index, test_index in kf.split(X):
            train_X, test_X = X[train_index], X[test_index]
            train_y, test_y = y[train_index], y[test_index]

            predictions = []
            for instance_test in test_X:
                k_neighbors = k_nearest_neighbors(train_X, instance_test, k)
                predictions.append(predict(train_y, k_neighbors))

            accuracies.append(evaluate(predictions, test_y))
        k_values[k] = np.mean(accuracies)

    k_values = sorted(k_values.items(), key=lambda p: p[1], reverse=True)
    chosen_k = next(iter(k_values))[0]
    return chosen_k


def confusion_matrix(test_y, predictions):
    amount = len(np.unique(test_y))
    confusion_matrix = np.zeros((amount, amount), dtype=int)
    for i in range(len(predictions)):
        row = int(predictions[i])
        col = int(test_y[i])
        confusion_matrix[row, col] += 1

    return confusion_matrix


def plot_decision_boundaries(train_X, train_y, test_X, test_y, k, figure_indice):
    predictions = []
    h = .02  # step size in the mesh
    # we only take two features.
    train_X = np.array(train_X)[:, [2, 3]]
    test_X = np.array(test_X)[:, [2, 3]]

    # Create color maps
    cmap_light = ListedColormap(['#FFAAAA', '#AAFFAA', '#AAAAFF'])
    cmap_bold = ListedColormap(['#FF0000', '#00FF00', '#0000FF'])

    # Create legend labels
    labels = ['Iris Setosa (treino)', 'Iris Versicolor (treino)', 'Iris Virginica (treino)',
              'Iris Setosa (teste)', 'Iris Versicolor (teste)', 'Iris Virginica (teste)']
    # labels = ['Hérnia de Disco (treino)', 'Espondilolistese (treino)', 'Normal (treino)',
    #           'Hérnia de Disco (teste)', 'Espondilolistese (teste)', 'Normal (teste)']
    # labels = ['classe 1 (treino)', 'classe 2 (treino)']

    # Create axis labels
    axis_labels = ['comprimento da pétala', 'largura da pétala']
    # axis_labels = ['incidência pélvica', 'inclinação pélvica']

    # Plot the decision boundary. For that, we will assign a color to each
    # point in the mesh [x_min, x_max]x[y_min, y_max].
    x_min, x_max = np.array(test_X)[:, 0].min() - 1, np.array(test_X)[:, 0].max() + 1
    y_min, y_max = np.array(test_X)[:, 1].min() - 1, np.array(test_X)[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()]

    for instance_test in Z:
        k_neighbors = k_nearest_neighbors(train_X, instance_test, k)
        predictions.append(predict(train_y, k_neighbors))

    # Put the result into a color plot
    predictions = np.array(predictions).reshape(xx.shape)
    plt.figure(figure_indice)
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
    plt.title("3-Class classification (k = %i)"
              % k)
    plt.xlabel(axis_labels[0])
    plt.ylabel(axis_labels[1])


if __name__ == '__main__':
    # define column names
    names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    # names = ['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope',
    #          'pelvic radius', 'grade of spondylolisthesis', 'class']
    # names = ['id', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'marginal_adhesion', 'epithelial_size',
    #          'bare_nucleoli', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

    # loading data
    dataset = pd.read_csv('iris.data.txt', names=names).values
    # dataset = pd.read_csv('column_3C.dat.txt', names=names).values
    # dataframe = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)
    # dataframe = dataframe.drop(['id', 'bare_nucleoli'], axis=1)
    # dataframe = pd.read_csv('dermatology.csv')
    # dataframe = dataframe.drop(['age'], axis=1)
    # dataset = dataframe.values

    # data normalization
    columns = np.shape(dataset)[1]
    columns = columns - 1
    minData = dataset[:, 0:columns].min(axis=0)
    maxData = dataset[:, 0:columns].max(axis=0)
    normalized_features = [normalize(data, minData, maxData) for data in dataset[:, 0:columns]]
    classes = dataset[:, columns].reshape(-1, 1)  # matrix transpose
    dataset = np.concatenate([normalized_features, classes], axis=1)

    # artificial dataset
    # features_1 = np.array([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0] for _ in range(50)])
    # features_2 = np.array([[random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1), 0] for _ in range(50)])
    # features_3 = np.array([[random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1), 0] for _ in range(50)])
    # features_4 = np.array([[random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), 1] for _ in range(50)])
    # dataset = np.concatenate([features_1, features_2, features_3, features_4], axis=0)

    accuracies = []
    k_values = []
    for j in range(0, 20):
        print("realization %d" % j)
        train_X, train_y, test_X, test_y = train_test_split(dataset)

        k = model_training(train_X, train_y)
        print("k value: %d" % k)
        k_values.append(k)
        predictions = []

        for instance_test in test_X:
            k_neighbors = k_nearest_neighbors(train_X, instance_test, k)
            predictions.append(predict(train_y, k_neighbors))

        plot_decision_boundaries(train_X, train_y, test_X, test_y, k, j)
        print(confusion_matrix(test_y, predictions))
        accuracies.append(evaluate(predictions, test_y))

    print('accuracies: {}'.format(accuracies))
    print('mean: {}'.format(np.mean(accuracies)))
    print('std: {}'.format(np.std(accuracies)))
    print('k values: {}'.format(k_values))
    plt.show()
