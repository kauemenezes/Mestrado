from BayesParzenWin import BayesParzenWinClassifier
import pandas as pd
import numpy as np
import random

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

# artificial dataset
# features_1 = np.array([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 0] for _ in range(50)])
# features_2 = np.array([[random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1), 0] for _ in range(50)])
# features_3 = np.array([[random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1), 0] for _ in range(50)])
# features_4 = np.array([[random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), 1] for _ in range(50)])
# dataset = np.concatenate([features_1, features_2, features_3, features_4], axis=0)

bayesClassifier = BayesParzenWinClassifier()
dataset = bayesClassifier.normalize_dataset(dataset)
accuracies = []
h_values = []

for j in range(0, 20):
    print("realization %d" % j)
    train_X, train_y, test_X, test_y = bayesClassifier.train_test_split(dataset)

    h = bayesClassifier.model_training(train_X, train_y)
    bayesClassifier.fit(train_X, train_y, h)
    h_values.append(h)

    predictions = bayesClassifier.predict(test_X)
    accuracies.append(bayesClassifier.evaluate(test_y, predictions))
    print(bayesClassifier.confusion_matrix(test_y, predictions))
    bayesClassifier.plot_decision_boundaries(train_X, train_y, test_X, test_y, j, h)

print('accuracies: {}'.format(accuracies))
print('mean: {}'.format(np.mean(accuracies)))
print('std: {}'.format(np.std(accuracies)))
print('h values: {}'.format(h_values))
bayesClassifier.show_plot_decision_boundaries()
