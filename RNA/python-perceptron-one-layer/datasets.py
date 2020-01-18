import numpy as np
import random
import pandas as pd
from classifier import Classifier


def get_iris_dataset():
    names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
    # loading data
    dataset = pd.read_csv('iris.data.txt', names=names).values
    dataset = Classifier.normalize_dataset(dataset)
    dataset = np.array(dataset, dtype=object)
    col = dataset.shape[1] - 1

    for i in range(len(dataset)):
        if dataset[i, col] == 0:
            dataset[i, col] = [1, 0, 0]
        elif dataset[i, col] == 1:
            dataset[i, col] = [0, 1, 0]
        else:
            dataset[i, col] = [0, 0, 1]

    return dataset


def get_artificial():
    features_1 = np.array([[random.uniform(0.4, 0.6), random.uniform(0.4, 0.6), [0, 0, 1]] for _ in range(50)])
    features_2 = np.array([[random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), [0, 1, 0]] for _ in range(50)])
    features_3 = np.array([[random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1), [1, 0, 0]] for _ in range(50)])
    dataset = np.concatenate([features_1, features_2, features_3], axis=0)

    return dataset


def get_dermatology_dataset():
    dataframe = pd.read_csv('dermatology.csv')
    dataframe = dataframe.drop(['age'], axis=1)
    dataset = dataframe.values
    dataset = Classifier.normalize_dataset(dataset)
    dataset = np.array(dataset, dtype=object)
    col = dataset.shape[1] - 1

    for i in range(len(dataset)):
        if dataset[i, col] == 0:
            dataset[i, col] = [1, 0, 0, 0, 0, 0]
        elif dataset[i, col] == 1:
            dataset[i, col] = [0, 1, 0, 0, 0, 0]
        elif dataset[i, col] == 2:
            dataset[i, col] = [0, 0, 1, 0, 0, 0]
        elif dataset[i, col] == 3:
            dataset[i, col] = [0, 0, 0, 1, 0, 0]
        elif dataset[i, col] == 4:
            dataset[i, col] = [0, 0, 0, 0, 1, 0]
        else:
            dataset[i, col] = [0, 0, 0, 0, 0, 1]

    return dataset


def get_vertebral_column_dataset():
    names = ['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope',
             'pelvic radius', 'grade of spondylolisthesis', 'class']
    dataset = pd.read_csv('column_3C.dat.txt', names=names).values
    dataset = Classifier.normalize_dataset(dataset)
    dataset = np.array(dataset, dtype=object)
    col = dataset.shape[1] - 1

    for i in range(len(dataset)):
        if dataset[i, col] == 0:
            dataset[i, col] = [1, 0, 0]
        elif dataset[i, col] == 1:
            dataset[i, col] = [0, 1, 0]
        else:
            dataset[i, col] = [0, 0, 1]

    return dataset

