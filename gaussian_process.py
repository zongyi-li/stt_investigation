# Combines Gaussian Processes with the Tensor Train Decomposition
#

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

from tests import *

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
        v = la.solve(self.L, kst)
        variance = np.abs(self.covariance(pt, pt) - np.dot(v, v))

        return mean, variance

def test_univariate():
    gp = GaussianProcess('RBF')

    X = np.array([i * 1.0 for i in range(0, 10)])
    y = np.sin(X)

    gp.build(X, y, 0)

    X_pred = np.array([i * 0.1 for i in range(0, 100)])
    y_pred = np.array([gp.query(x)[0] for x in X_pred])


    variances = np.array([gp.query(x)[1] for x in X_pred])
    upperBound = y_pred + np.sqrt(variances) * 2
    lowerBound = y_pred - np.sqrt(variances) * 2

    plt.plot(X, y)
    plt.plot(X_pred, y_pred)

    plt.plot(X_pred, upperBound)
    plt.plot(X_pred, lowerBound)

    plt.show()

def test_multivariate():
    gp = GaussianProcess('RBF')

    nDims = 2
    gridRange = 10
    subDivs = 5

    X = getEquispaceGrid(nDims, gridRange, subDivs).getPointArray()
    y = np.array(X)

    for i in range(len(y)):
        y[i] = polytest2(X[i])

    print("Building approximation...")
    gp.build(X, y, 0)

    # Compute an average RMSE by taking 10,000 data points randomly from the simulation box,
    # computing polynomial, evaluating error

    num_test_points = 1000
    rmse = 0

    for i in range(num_test_points):
        x = np.random.rand(nDims) * gridRange
        y = polytest2(x)
        print(gp.query(x)[0])
        rmse += (y - gp.query(x)[0]) ** 2

    rmse = np.sqrt(rmse / num_test_points)

    print(rmse)

if __name__ == '__main__':
    test_univariate()
    # test_multivariate()