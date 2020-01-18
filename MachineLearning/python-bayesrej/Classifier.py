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

    @abstractmethod
    def fit(self, train_x, train_y):
        pass

    def separate_by_class(self, train_x, train_y):
        separated = {}
        for i in range(len(train_x)):
            vector = train_x[i]
            if train_y[i] not in separated:
                separated[train_y[i]] = []
            separated[train_y[i]].append(vector)
        return separated

    @abstractmethod
    def calculate_attributes(self, train_x, train_y):
        pass

    @abstractmethod
    def calculate_class_probabilities(self, input_vector):
        pass

    def calculate_class(self, input_vector):
        probabilities = self.calculate_class_probabilities(input_vector)
        best_label, best_prob = None, -1
        for classValue, probability in probabilities.items():
            if best_label is None or probability > best_prob:
                best_prob = probability
                best_label = classValue
        return best_label

    def predict(self, test_x):
        predictions = []
        for i in range(len(test_x)):
            result = self.calculate_class(test_x[i])
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
