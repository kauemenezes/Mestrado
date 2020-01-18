import pandas as pd
import numpy as np
from perceptron import Perceptron
import matplotlib.pyplot as plt
from classifier import Classifier
import random

# define column names
# names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# names = ['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope',
#          'pelvic radius', 'grade of spondylolisthesis', 'class']

# loading data
# dataset = pd.read_csv('iris.data.txt', names=names).values

# artificial dataset
features_1 = np.array([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0] for _ in range(50)])
features_2 = np.array([[random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1), 1] for _ in range(50)])
features_3 = np.array([[random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1), 1] for _ in range(50)])
features_4 = np.array([[random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), 0] for _ in range(50)])
dataset = np.concatenate([features_1, features_2, features_3, features_4], axis=0)

# bias = np.ones(len(dataset)) * -1
# dataset = np.concatenate([bias.reshape(-1, 1), dataset], axis=1)
dataset = Classifier.normalize_dataset(dataset)

# for i in range(len(dataset)):
#     if dataset[i, 4] == 1:
#         dataset[i, 4] = 0
#
# for i in range(len(dataset)):
#     if dataset[i, 4] == 2:
#         dataset[i, 4] = 1

hit_rates = []

for j in range(0, 1):
    print("realization %d" % j)
    perceptron = Perceptron(2)
    train_X, train_y, test_X, test_y = perceptron.train_test_split(dataset)
    # Generating convergence chart
    dict = perceptron.train(train_X, train_y)
    # plt.figure(j)
    # plt.plot(np.array(list(dict.keys())), np.array(list(dict.values())), marker="^")
    # plt.ylabel('sum of errors')
    # plt.xlabel('epochs')

    predictions = perceptron.predict(test_X)
    hit_rates.append(perceptron.evaluate(test_y, predictions))
    print(perceptron.confusion_matrix(test_y, predictions))
    perceptron.plot_decision_boundaries(train_X, train_y, test_X, test_y, j)

print('hit rates: {}'.format(hit_rates))
print('accuracy: {}'.format(np.mean(hit_rates)))
print('std: {}'.format(np.std(hit_rates)))
# plt.show()
Classifier.show_plot_decision_boundaries()
