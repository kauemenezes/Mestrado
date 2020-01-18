from sklearn.neighbors import KNeighborsClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.model_selection import train_test_split
from util import datasets
from Extractor import Extractor
from sklearn.metrics import confusion_matrix
import numpy as np


def evaluate(predictions, test_y):
    accuracy = sum(i == j for i, j in zip(
        predictions, test_y)) / len(test_y)

    return accuracy


n_realizations = 20
hit_rates = []
X, y = datasets.get_car_numbers_dataset()
# X = Extractor.lbp_extraction(X , 24, 8)
# X = Extractor.hu_extraction(X)
X = Extractor.glcm_extraction(X)

# create new a knn model
knn = KNeighborsClassifier()

for _ in range(n_realizations):
    # split dataset into train and test data
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y)

    # create a dictionary of all values we want to test for n_neighbors
    param_grid = {'n_neighbors': np.arange(3, 20, 2)}

    # use gridsearch to test all values for n_neighbors
    knn_gscv = GridSearchCV(knn, param_grid, cv=5)

    # fit model to data
    knn_gscv.fit(X_train, y_train)

    # check top performing n_neighbors value
    # knn_gscv.best_params_

    # check mean score for the top performing value of n_neighbors
    # knn_gscv.best_score_

    # Create KNN classifier
    knn = KNeighborsClassifier(n_neighbors=knn_gscv.best_params_['n_neighbors'])

    # Fit the classifier to the data
    knn.fit(X_train, y_train)
    predictions = knn.predict(X_test)

    hit_rates.append(evaluate(predictions, y_test))
    print(confusion_matrix(y_test, predictions))
    print("\n\n")


print("hit rates: {}".format(hit_rates))
print("accuracy: {}".format(np.mean(hit_rates)))
print('std: {}'.format(np.std(hit_rates)))

