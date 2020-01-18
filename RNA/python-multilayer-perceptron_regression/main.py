import numpy as np
from perceptron import MLP
import datasets


dataset = datasets.get_artificial_one()

no_of_attributes = dataset.shape[1] - 1
no_of_classes = 1
# no_of_classes = np.unique(dataset[:, no_of_attributes]).size

# insert bias
no_rows = dataset.shape[0]
dataset = np.c_[-1 * np.ones(no_rows), dataset]

dictionary = {}
dictionary['mse'] = []
dictionary['rmse'] = []

for j in range(0, 10):
    print("realization %d" % j)
    train_X, train_y, test_X, test_y = MLP.train_test_split(dataset)
    train_X = np.array(train_X, dtype=float)
    test_X = np.array(test_X, dtype=float)

    hidden_neurons = MLP.model_training(no_of_classes, no_of_attributes, train_X, train_y)
    print("\nHidden neurons: {}".format(hidden_neurons))
    perceptron = MLP(no_of_classes, no_of_attributes, hidden_neurons, 'logistic')
    perceptron.train(train_X, train_y)
    predictions = perceptron.predict(test_X)
    mse, rmse = perceptron.evaluate(test_y, predictions)
    dictionary['mse'].append(mse)
    dictionary['rmse'].append(rmse)
    # MLP.plot_decision_boundaries_one(train_X, train_y, test_X, test_y, perceptron, j)

print('mean square error: {}'.format(dictionary['mse']))
print('root mean square error: {}'.format(dictionary['rmse']))
print('mean mse: {}'.format(np.mean(dictionary['mse'])))
print('mean rmse: {}'.format(np.mean(dictionary['rmse'])))
print('std mse: {}'.format(np.std(dictionary['mse'])))
print('std rmse: {}'.format(np.std(dictionary['rmse'])))
# MLP.show_plot_decision_boundaries()
