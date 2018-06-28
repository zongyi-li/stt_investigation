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

# Testing w/ my own TT-wrapper
from tensor_train import tensor_train

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

def pairwise_covariance_function(X):
    return np.exp(-0.5 * la.norm(X[0:4] - X[4:8]) ** 2)

def test_covariance_approximation():
    '''
    Evaluate the ability of TT-DMRG-cross to approximate the RBF covariance matrix.
    :return:
    '''

    # Test the capabilities of the TT-DMRG algorithm
    nDims = 2
    gridRange = 1
    subDivs = 15

    # TODO: Need to figure out what these parameters do!
    maxvoleps = 1e-3
    eps = 0.01

    X = getEquispaceGrid(nDims, gridRange, subDivs).gridIndices
    TW = DT.TensorWrapper(lambda X, params: pairwise_covariance_function(X), X, None)
    TTapprox = DT.TTvec(TW)

    print("Building tensor-train approximation...")
    TTapprox.build(method='ttdmrg', eps=eps, mv_eps=maxvoleps)
    print("Approximation complete!")

    # Evaluate the error at set # of randomly-selected test points
    rmse = 0
    numTest = 1000

    for i in range(numTest):
        pt = np.array(np.random.randint(subDivs, size=nDims))
        gridPoint = np.array([X[i][pt[i]] for i in range(len(pt))])
        rmse += (pairwise_covariance_function(gridPoint) - TTapprox.__getitem__(pt)) ** 2
        print("{}, {}".format(pairwise_covariance_function(gridPoint), TTapprox.__getitem__(pt)))
    print(np.sqrt(rmse / numTest))

def test_alpha_approximation():
    gp = GaussianProcess('RBF')

    nDims = 4
    gridRange = 6
    subDivs = gridRange

    maxvoleps = 1e-5
    eps = 1e-8

    grid = getEquispaceGrid(nDims, gridRange, subDivs)
    X = grid.getPointArray()
    y = np.zeros(len(X))

    for i in range(len(y)):
        y[i] = polytest3(X[i])

    print("Building Gaussian Process...")
    gp.build(X, y, 0, grid)                  # 0 indicates the prior is making no assumption of noise

    # Construct a tensor-train approximation to the alpha vector...
    TW = DT.TensorWrapper(lambda x, params: gp.alpha_function(x), grid.gridIndices, None)
    TTapprox = DT.TTvec(TW)

    print("Building TT-approximation of alpha vector...")
    TTapprox.build(method='ttdmrgcross', eps=eps, mv_eps=maxvoleps)
    print("Approximation complete!")

    rmse = 0
    numTest = 100

    # See how good the approximation was:
    for i in range(numTest):
        pt = np.array(np.random.randint(subDivs, size=nDims))
        rmse += (gp.alpha_function(pt) - TTapprox.__getitem__(pt)) ** 2
        print("{}, {}".format(gp.alpha_function(pt), TTapprox.__getitem__(pt)))

    print(np.sqrt(rmse / numTest))


def conjugate_gradient_inverter():
    '''
    Implements the conjugate gradient technique for inverting a matrix.
    :return:
    '''
    pass

if __name__ == '__main__':
    test_alpha_approximation()
    # test_univariate()
    # test_multivariate()
    # test_covariance_approximation()