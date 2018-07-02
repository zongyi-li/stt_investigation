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

def conjugate_grad(A, b, x=None):
    """
    Description
    -----------
    Solve a linear equation Ax = b with conjugate gradient method.
    Parameters
    ----------
    A: 2d numpy.array of positive semi-definite (symmetric) matrix
    b: 1d numpy.array
    x: 1d numpy.array of initial point
    Returns
    -------
    1d numpy.array x such that Ax = b
    """
    n = len(b)
    if not x:
        x = np.ones(n)
    r = np.dot(A, x) - b
    p = - r
    r_k_norm = np.dot(r, r)
    for i in range(2*n):
        Ap = np.dot(A, p)
        alpha = r_k_norm / np.dot(p, Ap)
        x += alpha * p
        r += alpha * Ap
        r_kplus1_norm = np.dot(r, r)
        beta = r_kplus1_norm / r_k_norm
        r_k_norm = r_kplus1_norm
        if r_kplus1_norm < 1e-3:
            print('Itr:', i)
            break
        p = beta * p - r

    return x

class GaussianProcess:
    def __init__(self, covariance_type):
        if covariance_type == 'RBF':
            self.covariance = lambda x, y: np.exp(-0.5 / 3.0 * la.norm(x - y) ** 2)
            self.approx_method = None

    def build_sd(self, X, y, sigma_n):
        '''
        Uses a simple random subset-of-data approximation

        :param X: Multidimensional inputs
        :param y: Targets
        :return:
        '''
        self.approx_method = 'SD'
        self.X = X
        self.y = y
        K = np.zeros((len(X), len(X)))

        for i in range(len(X)):
            for j in range(len(X)):
                K[i, j] = self.covariance(X[i], X[j])
                if i == j:
                    K[i, j] += sigma_n ** 2

        # self.L = la.cholesky(K)
        # ly = la.solve(self.L, y)
        # self.alpha = la.solve(self.L.transpose(), ly)
        self.alpha = conjugate_grad(K, np.ravel(y))

    def build_ii(self, X, y, sigma, M, ranges):
        '''
        :param X:
        :param y:
        :param sigma_n:
        :param M: The number of induced inputs
        :param ranges: Ranges in each coordinate axis to randomly-initialize the pseudo-input locations
        :return:
        '''
        self.approx_method = 'II'

        # Initialize the locations of the pseudo-inputs randomly
        self.M = M
        self.Xbar = np.array(np.random.rand(M, len(X[0])))

        for i in range(M):
            for j in range(len(X[0])):
                self.Xbar[i, j] = (ranges[j][1] - ranges[j][0]) * self.Xbar[i, j] + ranges[j][0]

        N = len(X)

        # Perform gradient ascent to determine pseudo-input locations
        # TODO: Not implemented yet!

        # Get the mean predictor given the pseudo-input values
        Km = np.zeros((M, M))
        Knm = np.zeros((N, M))
        kn = np.zeros((M, 1))
        for i in range(M):
            for j in range(M):
                Km[i, j] = self.covariance(self.Xbar[i], self.Xbar[j])

            for j in range(N):
                Knm[j, i] = self.covariance(self.Xbar[i], X[j])

        ldiag = np.zeros(N)
        for i in range(N):
            kn = Knm[i:i+1]
            ldiag[i] = self.covariance(X[i], X[i]) - np.matmul(np.matmul(kn, la.inv(Km)), kn.transpose())[0, 0] + sigma
            # Invert the value
            ldiag[i] = 1.0 / ldiag[i]

        kmnc_inv = np.matmul(Knm.transpose(), np.diag(ldiag))
        # rhs = np.matmul(kmnc_inv, y.transpose())

        Qm = Km + np.matmul(kmnc_inv, Knm)
        Qminv = la.inv(Qm)
        # self.L = la.cholesky(Qm)
        # ly = la.solve(self.L, rhs)
        # self.alpha = la.solve(self.L.transpose(), ly)
        self.alpha = np.matmul(np.matmul(Qminv, kmnc_inv), y.transpose())

    def query(self, pt):
        '''
        Queries the mean and variance at a given point from the GP.

        :param pt: Test point
        :return: Mean, variance of the provided test points
        '''

        mean = None
        variance = None

        # Subset of data approximation
        if self.approx_method == 'SD':
            # Compute the covariance of the test point with all of the other points
            kst = np.zeros(len(self.X))
            for i in range(len(self.X)):
                kst[i] = self.covariance(pt, self.X[i])

            mean = np.dot(kst.transpose(), self.alpha)
            # v = la.solve(self.L, kst)
            # variance = np.abs(self.covariance(pt, pt) - np.dot(v, v))

        # Induced input approximation
        elif self.approx_method == 'II':
            kst = np.zeros(self.M)
            for i in range(self.M):
                kst[i] = self.covariance(pt, self.Xbar[i])

            mean = np.dot(kst.transpose(), self.alpha)[0]

        return mean, #, variance

def test_univariate():
    gp = GaussianProcess('RBF')

    X = np.array([[i * 0.1] for i in range(0, 150)])
    y = np.sin(X.transpose()) + np.random.normal(0, scale=0.15, size=len(X))


    gp.build_ii(X, y, 0.001, 8, [[0, 15]])


    X_pred = np.array([i * 0.1 for i in range(-50, 200)])
    y_pred = np.array([gp.query(x) for x in X_pred]) # Add indexing [0] after adding in variance computation

    # variances = np.array([gp.query(x)[1] for x in X_pred])
    # upperBound = y_pred + np.sqrt(variances) * 2
    # lowerBound = y_pred - np.sqrt(variances) * 2

    plt.scatter(np.ravel(X), np.ravel(y), s=0.4)
    plt.plot(X_pred, y_pred)

    # plt.plot(X_pred, upperBound)
    # plt.plot(X_pred, lowerBound)

    plt.show()

if __name__ == '__main__':
    test_univariate()
    # test_multivariate()
    # test_covariance_approximation()