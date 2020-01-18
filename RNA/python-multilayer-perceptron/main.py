import numpy as np
from perceptron import Perceptron
import datasets


dataset = datasets.get_artificial_xor()

hit_rates = []
no_of_attributes = dataset.shape[1] - 1
no_of_classes = len(dataset[0, no_of_attributes])

# insert bias
no_rows = dataset.shape[0]
dataset = np.c_[-1 * np.ones(no_rows), dataset]

# perceptron = Perceptron(no_of_classes, no_of_attributes, 5, 'logistic')

for j in range(0, 5):
    print("realization %d" % j)
    train_X, train_y, test_X, test_y = Perceptron.train_test_split(dataset)
    train_X = np.array(train_X, dtype=float)
    test_X = np.array(test_X, dtype=float)

    # hidden_neurons = Perceptron.model_training(no_of_classes, no_of_attributes, train_X, train_y)
    # print("\nHidden neurons: {}".format(hidden_neurons))
    perceptron = Perceptron(no_of_classes, no_of_attributes, 13, 'logistic')
    perceptron.train(train_X, train_y)
    predictions = perceptron.predict(test_X)
    hit_rates.append(perceptron.evaluate(test_y, predictions))
    print(perceptron.confusion_matrix(test_y, predictions))
    Perceptron.plot_decision_boundaries(train_X, train_y, test_X, test_y, perceptron, 13, j)

print('hit rates: {}'.format(hit_rates))
print('accuracy: {}'.format(np.mean(hit_rates)))
print('std: {}'.format(np.std(hit_rates)))
Perceptron.show_plot_decision_boundaries()
