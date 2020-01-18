import pandas as pd
import numpy as np
import math
import matplotlib.pyplot as plt
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


def calculate_centroids(train_X, train_y):
    centroids = []

    classes = np.array(train_y).reshape(-1, 1)  # matrix transpose
    train_dataset = np.concatenate([np.array(train_X), classes], axis=1)
    columns = train_dataset.shape[1] - 1

    dictionary = {classes: train_dataset[train_dataset[:, columns] == classes] for classes in np.unique(train_dataset[:, columns])}
    for key in dictionary:
        centroid = np.mean(dictionary[key][:, 0:columns], axis=0)
        centroid = np.append(centroid, key)
        centroids.append(centroid)

    return centroids


def min_max_normalizer(x, min, max):
    return (x - min) / (max - min) if (max - min) else 0  # avoid division by 0


def normalize(data, minData, maxData):
    norm_data = [min_max_normalizer(data[i], minData[i], maxData[i]) for i in range(len(data))]
    return norm_data


def euclidean_distance(point_p, point_q):
    square_of_diffs = lambda p_i, q_i: (float(q_i) - float(p_i)) ** 2
    distance = list(map(square_of_diffs, point_p, point_q))

    return math.sqrt(sum(distance))


def predict(centroids, test_instance):
    columns = centroids.shape[1] - 1
    distances = [euclidean_distance(X, test_instance) for X in centroids[:, 0:columns]]
    indice_min_distance = sorted(range(len(distances)), key=lambda i: distances[i])[:1]
    return centroids[indice_min_distance[0], columns]


def evaluate(predictions, test_y):
    accuracy = sum(i == j for i, j in zip(
        predictions, test_y)) / len(test_y)

    return accuracy


def confusion_matrix(test_y, predictions):
    amount = len(np.unique(test_y))
    confusion_matrix = np.zeros((amount, amount), dtype=int)
    for i in range(len(predictions)):
        row = int(test_y[i])
        col = int(predictions[i])
        confusion_matrix[row, col] += 1

    return confusion_matrix


def plot_decision_boundaries(train_X, train_y, test_X, test_y, figure_indice):
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
    # labels = ['classe 1 (treino)', 'classe 2 (treino)', 'classe 1 (teste)', 'classe 2 (teste)']

    # Create axis labels
    # axis_labels = ['comprimento da pétala', 'largura da pétala']
    axis_labels = ['incidência pélvica', 'inclinação pélvica']

    x_min, x_max = test_X[:, 0].min() - 1, test_X[:, 0].max() + 1
    y_min, y_max = test_X[:, 1].min() - 1, test_X[:, 1].max() + 1
    xx, yy = np.meshgrid(np.arange(x_min, x_max, h),
                         np.arange(y_min, y_max, h))
    Z = np.c_[xx.ravel(), yy.ravel()]

    centroids = calculate_centroids(train_X, train_y)
    centroids = np.array(centroids)

    for instance_test in Z:
        predictions.append(predict(centroids, instance_test))

    # Put the result into a color plot
    predictions = np.array(predictions).reshape(xx.shape)
    plt.figure(figure_indice)
    plt.pcolormesh(xx, yy, predictions, cmap=cmap_light)

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
    plt.scatter(centroids[:, 0], centroids[:, 1], s=50, c='red', label='Centroids')
    plt.xlim(xx.min(), xx.max())
    plt.ylim(yy.min(), yy.max())
    plt.legend()
    plt.title("3-Class classification")
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
    dataset = pd.read_csv('../../iris.data.txt', names=names).values
    # dataset = pd.read_csv('../../column_3C.dat.txt', names=names).values
    # dataframe = pd.read_csv('../../breast-cancer-wisconsin.data.csv', names=names)
    # dataframe = dataframe.drop(['id', 'bare_nucleoli'], axis=1)
    # dataframe = pd.read_csv('../../dermatology.csv')
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
    for j in range(0, 20):
        print("realization %d" % j)
        train_X, train_y, test_X, test_y = train_test_split(dataset)

        centroids = calculate_centroids(train_X, train_y)
        centroids = np.array(centroids)

        predictions = []

        plot_decision_boundaries(train_X, train_y, test_X, test_y, j)
        for instance_test in test_X:
            predictions.append(predict(centroids, instance_test))

        print(confusion_matrix(test_y, predictions))
        accuracies.append(evaluate(predictions, test_y))

    print('accuracies: {}'.format(accuracies))
    print('mean: {}'.format(np.mean(accuracies)))
    print('std: {}'.format(np.std(accuracies)))
    plt.show()
