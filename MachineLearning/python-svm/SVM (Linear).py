from sklearn.svm import SVC
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.model_selection import GridSearchCV
import pandas as pd
from Classifier import Classifier
import numpy as np
import random

# names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# names = ['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope',
#          'pelvic radius', 'grade of spondylolisthesis', 'class']
names = ['id', 'clump_thickness', 'size_uniformity', 'shape_uniformity', 'marginal_adhesion', 'epithelial_size',
         'bare_nucleoli', 'bland_chromatin', 'normal_nucleoli', 'mitoses', 'class']

# dataset = pd.read_csv('iris-binary.data.txt', names=names).values
# dataset = pd.read_csv('column_2C.dat.txt', names=names).values
dataframe = pd.read_csv('breast-cancer-wisconsin.data.csv', names=names)
dataframe = dataframe.drop(['id', 'bare_nucleoli'], axis=1)
# dataframe = pd.read_csv('dermatology.csv')
# dataframe = dataframe.drop(['age'], axis=1)
dataset = dataframe.values

# artificial dataset
# features_1 = np.array([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), -1] for _ in range(50)])
# features_2 = np.array([[random.uniform(-0.1, 0.1), random.uniform(0.9, 1.1), -1] for _ in range(50)])
# features_3 = np.array([[random.uniform(0.9, 1.1), random.uniform(-0.1, 0.1), -1] for _ in range(50)])
# features_4 = np.array([[random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), 1] for _ in range(50)])
# dataset = np.concatenate([features_1, features_2, features_3, features_4], axis=0)

clf = Classifier()
dataset = clf.normalize_dataset(dataset)

accuracies = []
params_values = []

for j in range(0, 1):
    print("realization %d" % j)
    train_X, train_y, test_X, test_y = clf.train_test_split(dataset)

    C_range = np.logspace(-2, 10, 5)
    param_grid = dict(C=C_range)
    cv = StratifiedShuffleSplit(n_splits=5, test_size=0.2, random_state=42)
    grid = GridSearchCV(SVC(kernel='linear', gamma="scale"), param_grid=param_grid, cv=cv)
    grid.fit(train_X, np.array(train_y))
    params_values.append(grid.best_params_)

    svc = SVC(kernel='linear', C=grid.best_params_['C'], gamma="scale")
    svc.fit(train_X, train_y)
    predictions = svc.predict(test_X)

    accuracies.append(clf.evaluate(test_y, predictions))
    print(clf.confusion_matrix(test_y, predictions))
    # clf.plot_decision_boundaries(train_X, train_y, test_X, test_y, j, grid.best_params_, 'linear')

print('accuracies: {}'.format(accuracies))
print('mean: {}'.format(np.mean(accuracies)))
print('std: {}'.format(np.std(accuracies)))
print('params values: {}'.format(params_values))
# clf.show_plot_decision_boundaries()
