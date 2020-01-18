from BayesRej import BayesRejClassifier
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import random

# define column names
names = ['sepal_length', 'sepal_width', 'petal_length', 'petal_width', 'class']
# names = ['pelvic incidence', 'pelvic tilt', 'lumbar lordosis angle', 'sacral slope',
#          'pelvic radius', 'grade of spondylolisthesis', 'class']

# loading data
dataset = pd.read_csv('iris-binary.data.txt', names=names).values
# dataset = pd.read_csv('column_2C.dat.txt', names=names).values

# artificial dataset
# features_1 = np.array([[random.uniform(-0.1, 0.1), random.uniform(-0.1, 0.1), 1] for _ in range(50)])
# features_2 = np.array([[random.uniform(0.9, 1.1), random.uniform(0.9, 1.1), 2] for _ in range(50)])
# dataset = np.concatenate([features_1, features_2], axis=0)

bayesClassifier = BayesRejClassifier()
dataset = bayesClassifier.normalize_dataset(dataset)
wr = [0.04, 0.12, 0.24, 0.36, 0.48]
accuracies = []
std_accuracies = []
rejections = []
std_rejections = []

for i in range(0, len(wr)):
    bayesClassifier.set_wr(wr[i])
    evaluates = []
    rejected = []
    for j in range(0, 20):
        print("realization %d" % j)
        train_X, train_y, test_X, test_y = bayesClassifier.train_test_split(dataset)
        bayesClassifier.fit(train_X, train_y)

        predictions = bayesClassifier.predict(test_X)
        accuracy, rejection = bayesClassifier.evaluate(test_y, predictions)
        evaluates.append(accuracy)
        rejected.append(rejection)

    accuracies.append(np.mean(evaluates))
    std_accuracies.append(np.std(evaluates))
    rejections.append(np.mean(rejected))
    std_rejections.append(np.std(rejected))

print('accuracies: {}'.format(accuracies))
print('std_accuracies: {}'.format(std_accuracies))
print('rejections: {}'.format(rejections))
print('std_rejections: {}'.format(std_rejections))
plt.plot(rejections, accuracies)
plt.title("Iris")
plt.xlabel("Taxa de Rejeição (x 100%)")
plt.ylabel("Taxa de Acerto (x 100%)")
plt.show()