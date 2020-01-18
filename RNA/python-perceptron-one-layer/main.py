import numpy as np
from perceptron import Perceptron
import datasets


dataset = datasets.get_dermatology_dataset()

hit_rates = []
no_of_attributes = dataset.shape[1] - 1
no_of_classes = len(dataset[0, no_of_attributes])
perceptron = Perceptron(no_of_classes, no_of_attributes, 'logistic')

for j in range(0, 20):
    print("realization %d" % j)
    train_X, train_y, test_X, test_y = perceptron.train_test_split(dataset)
    perceptron.train(train_X, train_y)

    predictions = perceptron.predict(test_X)
    hit_rates.append(perceptron.evaluate(test_y, predictions))
    print(perceptron.confusion_matrix(test_y, predictions))
    # Perceptron.plot_decision_boundaries(train_X, train_y, test_X, test_y, j)

print('hit rates: {}'.format(hit_rates))
print('accuracy: {}'.format(np.mean(hit_rates)))
print('std: {}'.format(np.std(hit_rates)))
# Perceptron.show_plot_decision_boundaries()
