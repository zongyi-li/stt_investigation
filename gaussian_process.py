# Combines Gaussian Processes with the Tensor Train Decomposition
#

import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import tensorly as tl
import itertools

import TensorToolbox.multilinalg as mla
import TensorToolbox as DT
from TensorToolbox.core import STT
import SpectralToolbox

from tests import *

COVARIANCE_FUNCTIONS = ['RBF']

class GaussianProcess:
    def __init__(self, covariance_type):
        if covariance_type == 'RBF':
            self.covariance = lambda x, y: np.exp(-0.5 * la.norm(x - y) ** 2)

    def alpha_function(self, X):
        res = 0
        for i in range(len(self.grid.gridIndices)):
            res *= len(self.grid.gridIndices[i])
            res += X[i]
        return self.alpha[int(res)]

    def build(self, X, y, sigma_n, grid):
        '''
        NOTE: currently not acccounting for noise!

        :param X: Multidimensional inputs
        :param y: Targets
        :return:
        '''
        self.grid = grid
        self.X = X
        self.y = y
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

    # plt.plot(X, y)
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

    # Compute an average RMSE by taking 1,000 data points randomly from the simulation box,
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


def conjugate_gradient_solve(A, b):
    '''
    Implements the conjugate gradient technique to solve the system Ax = b for x.
    :return:
    '''
    x = np.zeros((len(b), 1))
    r = np.matmul(A, x) - b
    p = -r
    rsold = (r.transpose() * r)[0, 0]

    count = 0
    for i in range(len(b)):
        count += 1
        Ap = np.matmul(A, p)
        alpha = rsold / (p.transpose() * Ap)[0, 0]

        x = x + alpha * p
        r = r + alpha * Ap
        rsnew = (r.transpose() * r)[0, 0]
        if np.sqrt(rsnew) < 1e-2:
            break
        p =  (rsnew / rsold) * p - r
        rsold = rsnew
        print(rsnew)

    print(count)

    return x, rsnew

if __name__ == '__main__':
    size = 100
    A = np.array(np.random.rand(size, size))
    for i in range(len(A)):
        for j in range(len(A[0])):
            if j < i:
                A[i][j] = A[j][i]

    print(A)

    b = np.matrix(np.random.rand(size)).transpose()
    x, rsnew = conjugate_gradient_solve(A, b)
    print(rsnew)
    # test_univariate()
    # test_multivariate()
    # test_covariance_approximation()