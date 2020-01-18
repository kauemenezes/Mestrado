import numpy as np
import datasets
from classifier import Classifier
from rbf import RBF


dataset = datasets.get_breast_cancer_dataset()

hit_rates = []
no_of_attributes = dataset.shape[1] - 1
no_of_classes = len(dataset[0, no_of_attributes])

# insert bias
no_rows = dataset.shape[0]
dataset = np.c_[-1 * np.ones(no_rows), dataset]

# perceptron = Perceptron(no_of_classes, no_of_attributes, 5, 'logistic')

for j in range(0, 20):
    print("realization %d" % j)
    train_X, train_y, test_X, test_y = Classifier.train_test_split(dataset)
    train_X = np.array(train_X, dtype=float)
    test_X = np.array(test_X, dtype=float)

    sigma, center = RBF.model_training(no_of_classes, no_of_attributes, train_X, train_y)
    rbf = RBF(no_of_classes, no_of_attributes, center, sigma)
    rbf.train(train_X, train_y)
    predictions = rbf.predict(test_X)
    hit_rates.append(rbf.evaluate(test_y, predictions))
    print(rbf.confusion_matrix(test_y, predictions))
    # rbf.plot_decision_boundaries(train_X, train_y, test_X, test_y, rbf, j)

print('hit rates: {}'.format(hit_rates))
print('accuracy: {}'.format(np.mean(hit_rates)))
print('std: {}'.format(np.std(hit_rates)))
# RBF.show_plot_decision_boundaries()
