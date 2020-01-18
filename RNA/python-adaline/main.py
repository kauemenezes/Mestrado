import numpy as np
from classifier import Classifier
from adaline import Adaline

hit_rates = []

# X = np.random.uniform(0, 1, 100)
# dataset = Classifier.generate_dataset_one(3, 5, X)
# no_of_inputs = 1

X1 = np.random.uniform(0, 1, 100)
X2 = np.random.uniform(0, 1, 100)
dataset = Classifier.generate_dataset_two(3, 5, 7, X1, X2)
no_of_inputs = 2

dictionary = {}
dictionary['mse'] = []
dictionary['rmse'] = []

for j in range(0, 1):
    print("realization %d" % j)
    adaline = Adaline(0.01)
    train_X, train_y, test_X, test_y = adaline.train_test_split(dataset)
    adaline.fit(no_of_inputs, np.array(train_X), np.array(train_y))

    adaline.calculate_error(np.array(test_X), np.array(test_y))
    dictionary['mse'].append(adaline.mse_)
    dictionary['rmse'].append(adaline.rmse_)
    # adaline.plot_decision_boundaries_one(train_X, train_y, test_X, test_y, j)
    adaline.plot_decision_boundaries_two(train_X, train_y, test_X, test_y, j, dataset)
#
print('mean square error: {}'.format(dictionary['mse']))
print('root mean square error: {}'.format(dictionary['rmse']))
print('mean mse: {}'.format(np.mean(dictionary['mse'])))
print('mean rmse: {}'.format(np.mean(dictionary['rmse'])))
print('std mse: {}'.format(np.std(dictionary['mse'])))
print('std rmse: {}'.format(np.std(dictionary['rmse'])))
Adaline.show_plot_decision_boundaries()
