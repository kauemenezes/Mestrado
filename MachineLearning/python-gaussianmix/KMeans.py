import numpy as np
import random


class K_Means:
    centroids = {}
    classifications = {}

    def __init__(self, k=2, tol=0.001, max_iter=300):
        self.k = k
        self.tol = tol
        self.max_iter = max_iter

    def fit(self, data):

        for i in range(self.k):
            self.centroids[i] = data[i]
        # ranges = [(min([row[i] for row in data]),
        #            max([row[i] for row in data]))
        #           for i in range(len(data[0]))]
        #
        # # Cria K centroides aleatórios
        # # Cria uma lista contendo os K centroides em posições aleatorias.
        # # No nosso caso serão 3
        # clusters = [[random.random() * (ranges[i][1] - ranges[i][0]) + ranges[i][0]
        #             for i in range(len(data[0]))] for j in range(self.k)]
        # for i in range(self.k):
        #     self.centroids[i] = clusters[i]

        for iteration in range(self.max_iter):

            for j in range(self.k):
                self.classifications[j] = []

            for featureset in data:
                distances = [np.linalg.norm(self.euclidean_distance(featureset, self.centroids[centroid]))
                             for centroid in self.centroids]
                classification = distances.index(min(distances))
                self.classifications[classification].append(featureset)

            prev_centroids = dict(self.centroids)

            for classification in self.classifications:
                self.centroids[classification] = np.average(self.classifications[classification], axis=0)

            optimized = True

            for c in self.centroids:
                original_centroid = prev_centroids[c]
                current_centroid = self.centroids[c]
                if np.sum((current_centroid-original_centroid)/original_centroid*100.0) > self.tol:
                    print(np.sum((current_centroid-original_centroid)/original_centroid*100.0))
                    optimized = False

            if optimized:
                print("iteration {}".format(iteration))
                break

    def euclidean_distance(self, x, y):
        return np.sqrt(np.sum((x - y) ** 2))

    def predict(self, data):
        distances = [np.linalg.norm(self.euclidean_distance(data, self.centroids[centroid]))
                     for centroid in self.centroids]
        classification = distances.index(min(distances))
        return classification
