import numpy as np
from abc import ABC, abstractmethod


class Classifier(ABC):

    @staticmethod
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

    # # @abstractmethod
    # def fit(self, X, iterations, number_of_sources):
    #     pass

    def confusion_matrix(self, test_y, predictions):
        amount = np.array(test_y).shape[1]
        confusion_matrix = np.zeros((amount, amount), dtype=int)
        for i in range(len(predictions)):
            row = np.where(np.array(predictions[i]) == 1)[0][0]
            col = np.where(np.array(test_y[i]) == 1)[0][0]
            confusion_matrix[row, col] += 1

        return confusion_matrix

    @abstractmethod
    def calculate_class(self, test_x):
        pass

    def predict(self, test_x):
        predictions = np.zeros(len(test_x), dtype=object)
        for i in range(len(test_x)):
            predictions[i] = self.calculate_class(test_x[i])
        return predictions

    def evaluate(self, test_y, predictions):
        correct = 0
        for i in range(len(test_y)):
            if np.array_equal(test_y[i], predictions[i]):
                correct += 1
        return correct / float(len(test_y))  # * 100.0

    @staticmethod
    def min_max_normalizer(x, min_value, max_value):
        return (x - min_value) / (max_value - min_value) if (max_value - min_value) else 0  # avoid division by 0

    @staticmethod
    def normalize(data, min_data, max_data):
        norm_data = [Classifier.min_max_normalizer(data[i], min_data[i], max_data[i]) for i in range(len(data))]
        return norm_data

    # data normalization
    @staticmethod
    def normalize_dataset(dataset):
        columns = np.shape(dataset)[1]
        columns = columns - 1
        min_data = dataset[:, 0:columns].min(axis=0)
        max_data = dataset[:, 0:columns].max(axis=0)
        normalized_features = [Classifier.normalize(data, min_data, max_data) for data in dataset[:, 0:columns]]
        classes = dataset[:, columns].reshape(-1, 1)  # matrix transpose
        dataset = np.concatenate([normalized_features, classes], axis=1)

        return dataset
