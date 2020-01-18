from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from sklearn.neural_network import MLPClassifier
from util import datasets
from sklearn.metrics import confusion_matrix
from Extractor import Extractor

import numpy as np


def evaluate(predictions, test_y):
    accuracy = sum(i == j for i, j in zip(
        predictions, test_y)) / len(test_y)

    return accuracy


n_realizations = 20
hit_rates = []
X, y = datasets.get_car_numbers_dataset()
# X = Extractor.lbp_extraction(X, 24, 8)
# X = Extractor.hu_extraction(X)
X = Extractor.glcm_extraction(X)

# MLP classifier
model = MLPClassifier(hidden_layer_sizes=(64, 128, 64), learning_rate='adaptive', learning_rate_init=0.005)

for _ in range(n_realizations):
    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)
    model.fit(X_train, y_train)
    predictions = model.predict(X_test)

    hit_rates.append(evaluate(predictions, y_test))
    print(confusion_matrix(y_test, predictions))
    print("\n\n")

print("hit rates: {}".format(hit_rates))
print('accuracy: {}'.format(np.mean(hit_rates)))
print('std: {}'.format(np.std(hit_rates)))

