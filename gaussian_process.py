# Combines Gaussian Processes with the Tensor Train Decomposition
#
#
#

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

COVARIANCE_FUNCTIONS = ['RBF']

class GaussianProcess:
    def __init__(self, covariance_type):
        if covariance_type == 'RBF':
            self.covariance = lambda x, y: np.exp(-0.5 * la.norm(x - y) ** 2)

    def build(self, X, y, sigma_n):
        '''
        NOTE: currently not acccounting for noise!


        :param X: Multidimensional inputs
        :param y: Targets
        :return:
        '''
        self.X = X
        K = np.zeros((len(X), len(X)))

        for i in range(len(X)):
            for j in range(len(X)):
                K[i, j] = self.covariance(X[i], X[j])

        self.L = la.cholesky(K)
        ly = la.solve(self.L, y)
        self.alpha = la.solve(self.L.transpose(), ly)

    def query(self, pt):
        '''
        Queries the mean and variance at a given point from the GP.

        :param pt: Test point
        :return: Mean, variance of the provided test points
        '''

        # Compute the covariance of the test point with all of the other points

        kst = np.zeros(len(self.X))
        for i in range(len(self.X)):
            kst[i] = self.covariance(pt, self.X[i])

        mean = np.dot(kst.transpose(), self.alpha)

        return mean

def test_univariate():
    gp = GaussianProcess('RBF')

    X = np.array([i * 1 for i in range(0, 10)])
    y = np.sin(X)

    gp.build(X, y, 0)

    X_pred = np.array([i * 0.1 for i in range(0, 100)])
    y_pred = np.array([gp.query(x) for x in X_pred])

    plt.plot(X, y)
    plt.plot(X_pred, y_pred)

    plt.show()

if __name__ == '__main__':
    test_univariate()
