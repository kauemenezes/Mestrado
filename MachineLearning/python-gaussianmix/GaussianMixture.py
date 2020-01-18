import numpy as np
from sklearn.cluster import KMeans
import math
from Classifier import Classifier


class GMM(Classifier):

    def gaussian(self, X, mu, cov):
        n = X.shape[1]
        diff = (X - mu).T
        return np.diagonal(1 / ((2 * np.pi) ** (n / 2) * np.linalg.det(cov) ** 0.5) * np.exp(
            -0.5 * np.dot(np.dot(diff.T, np.linalg.inv(cov)), diff))).reshape(-1, 1)

    def calculate_class_probabilities(self, input_vector, model):
        classes_probabilities = {}
        for classValue, clusters in model.items():
            probabilities = []
            for cluster in clusters:
                pi_k = cluster['pi_k']
                mu_k = cluster['mu_k']
                cov_k = cluster['cov_k']
                probabilities.append(pi_k * self.gaussian(input_vector.reshape(1, -1), mu_k, cov_k))

            classes_probabilities[classValue] = sum(probabilities)
        return classes_probabilities

    def initialize_clusters(self, X, n_clusters):
        clusters = []

        kmeans = KMeans().fit(X)
        mu_k = kmeans.cluster_centers_

        for i in range(n_clusters):
            clusters.append({
                'pi_k': 1.0 / n_clusters,
                'mu_k': mu_k[i],
                'cov_k': np.identity(X.shape[1], dtype=np.float64)
            })

        return clusters

    def expectation_step(self, X, clusters):
        totals = np.zeros((X.shape[0], 1), dtype=np.float64)

        for cluster in clusters:
            pi_k = cluster['pi_k']
            mu_k = cluster['mu_k']
            cov_k = cluster['cov_k']

            gamma_nk = (pi_k * self.gaussian(X, mu_k, cov_k)).astype(np.float64)

            for i in range(X.shape[0]):
                totals[i] += gamma_nk[i]

            cluster['gamma_nk'] = gamma_nk.reshape(-1, 1)
            cluster['totals'] = totals

        for cluster in clusters:
            cluster['gamma_nk'] /= cluster['totals']

    def maximization_step(self, X, clusters):
        N = float(X.shape[0])

        for cluster in clusters:
            gamma_nk = cluster['gamma_nk']
            cov_k = np.zeros((X.shape[1], X.shape[1]))

            N_k = np.sum(gamma_nk, axis=0)

            pi_k = N_k / N
            mu_k = np.sum(gamma_nk * X, axis=0) / N_k

            for j in range(X.shape[0]):
                diff = (X[j] - mu_k).reshape(-1, 1)
                cov_k += gamma_nk[j] * np.dot(diff, diff.T)

            cov_k /= N_k

            cluster['pi_k'] = pi_k
            cluster['mu_k'] = mu_k
            identity_matrix = np.identity(X.shape[1])
            dirty_matrix = 0.001 * identity_matrix
            cluster['cov_k'] = cov_k + dirty_matrix

    def get_likelihood(self, clusters):
        sample_likelihoods = np.log(np.array([cluster['totals'] for cluster in clusters]))
        return np.sum(sample_likelihoods)

    def train_model(self, X, n_clusters, n_epochs, threshold):
        clusters = self.initialize_clusters(X, n_clusters)
        likelihoods = np.zeros((n_epochs,))
        old_likelihood = 0

        for i in range(n_epochs):
            self.expectation_step(X, clusters)
            self.maximization_step(X, clusters)

            likelihood = self.get_likelihood(clusters)
            likelihoods[i] = likelihood
            if abs(old_likelihood - likelihood) < threshold:
                # print('GMM converged on epoch: ', i + 1, 'Likelihood: ', likelihood)
                break
            old_likelihood = likelihood
            # print('GMM converged on epoch: ', i + 1, 'Likelihood: ', likelihood)

        return clusters
