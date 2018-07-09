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
import progressbar

from scipy.optimize import minimize

from multiprocessing import Process, Value, Array

from tests import *

COVARIANCE_FUNCTIONS = ['RBF']

def compute_likelihood(hp, covariance, X, y, noise, approx_method, Xbar=None):
    if approx_method == 'SD':
        R = np.zeros((len(X), len(X)))
        for i in range(len(X)):
            for j in range(len(X)):
                R[i, j] = covariance(X[i], X[j], hp)
                if i == j:
                    R[i, j] += noise ** 2

        L = la.cholesky(R)
        t = la.solve(L, y)
        rinv_y = la.solve(L.transpose(), t)
        return -(-0.5 * np.log(la.det(R)) - 0.5 * np.matmul(y.transpose(), rinv_y) - len(X) / 2 * np.log(2 * np.pi))

    elif approx_method == 'II':
        M = len(Xbar)
        N = len(X)

        # Get the mean predictor given the pseudo-input values
        Km = np.zeros((M, M))
        Knm = np.zeros((N, M))
        kn = np.zeros((M, 1))

        for i in range(M):
            for j in range(M):
                Km[i, j] = covariance(Xbar[i], Xbar[j], hp)

            for j in range(N):
                Knm[j, i] = covariance(Xbar[i], X[j], hp)

        ldiag = np.zeros(N)
        Kminv = la.inv(Km)

        for i in range(N):
            kn = Knm[i:i + 1]
            ldiag[i] = covariance(X[i], X[i], hp) - np.matmul(np.matmul(kn, Kminv), kn.transpose())[0, 0]
            # Invert the value

        R = np.matmul(np.matmul(Knm, Kminv), Knm.transpose())

        for i in range(len(X)):
            R[i, i] += ldiag[i] + noise ** 2

        L = la.cholesky(R)
        t = la.solve(L, y)
        rinv_y = la.solve(L.transpose(), t)
        return -(-0.5 * np.log(la.det(R)) - 0.5 * np.matmul(y.transpose(), rinv_y) - len(X) / 2 * np.log(2 * np.pi))


def lloydKMeans(inputs, numClusters, numIter):

    clusters = np.array([inputs[np.random.randint(len(inputs))] for i in range(numClusters)])
    for i in progressbar.progressbar(range(numIter)):

        newClusters = np.zeros(np.shape(clusters))
        clusterCounts = np.zeros(len(clusters))

        for j in range(len(inputs)):
            closestCluster = 0
            closestDist = np.inf
            for k in range(len(clusters)):
                if la.norm(inputs[j] - clusters[k]) < closestDist:
                    closestCluster = k
                    closestDist = la.norm(inputs[j] - clusters[k])

            newClusters[closestCluster] += inputs[j]
            clusterCounts[closestCluster] += 1

        for k in range(len(clusters)):
            if clusterCounts[k] == 0:
                newClusters[k] = clusters[k]
            else:
                newClusters[k] /= clusterCounts[k]

        clusters = newClusters

    return clusters

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
    def __init__(self, covariance_type, hp, y_noise):
        self.hp = hp
        self.sigma_n = y_noise
        self.covariance_type = covariance_type

        if covariance_type == 'RBF':
            # Hyperparameters:
            # hp[0]: sigma_rbf (value in covariance)

            self.covariance = lambda x, y, hp: np.exp(-0.5 / (hp[0] ** 2) * la.norm(x - y) ** 2)

            self.hyperparam_derivs = [ # Analytic derivatives of covariance w.r.t. hyperparams, excluding sigma_n
                lambda x, y: -2 * self.covariance(x, y) * la.norm(x - y) / self.hp[0] ** 3 # Derivative w.r.t. sigma_rbf
            ]
            self.approx_method = None

    def compute_likelihood(self, X, y, Xbar=None):
        return -compute_likelihood(self.hp, self.covariance, X, y, self.sigma_n, self.approx_method, Xbar=Xbar)

    def learn_hyperparameters(self, X, y, approx_type, centers=None):
        res = minimize(compute_likelihood, self.hp, args=(self.covariance, X, y, self.sigma_n, approx_type, centers), method='CG', jac=False,
                       options={"maxiter": 30, "disp": True})
        self.hp = res.x

    def build_multigp(self):
        pass

    def build_sd(self, X, y):
        '''
        Uses a simple random subset-of-data approximation

        :param X: Multidimensional inputs
        :param y: Targets
        :return:
        '''
        self.approx_method = 'SD'
        self.X = X
        self.y = y
        R = np.zeros((len(X), len(X)))

        for i in range(len(X)):
            for j in range(len(X)):
                R[i, j] = self.covariance(X[i], X[j], self.hp)
                if i == j:
                    R[i, j] += self.sigma_n ** 2

        Rinv = la.inv(R)

        # self.L = la.cholesky(K)
        # ly = la.solve(self.L, y)
        # self.alpha = la.solve(self.L.transpose(), ly)
        self.alpha = conjugate_grad(R, np.ravel(y))

    def build_ii(self, X, y, M, ranges, Xbar=None):
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

        if Xbar is None:
            self.Xbar = np.array(np.random.rand(M, len(X[0])))
            for i in range(M):
                for j in range(len(X[0])):
                    self.Xbar[i, j] = (ranges[j][1] - ranges[j][0]) * self.Xbar[i, j] + ranges[j][0]
        else:
            self.Xbar = Xbar

        N = len(X)

        # Get the mean predictor given the pseudo-input values
        Km = np.zeros((M, M))
        Knm = np.zeros((N, M))
        kn = np.zeros((M, 1))

        for i in range(M):
            for j in range(M):
                Km[i, j] = self.covariance(self.Xbar[i], self.Xbar[j], self.hp)

            for j in range(N):
                Knm[j, i] = self.covariance(self.Xbar[i], X[j], self.hp)

        ldiag = np.zeros(N)
        Kminv = la.inv(Km)
        for i in range(N):
            kn = Knm[i:i+1]
            ldiag[i] = self.covariance(X[i], X[i], self.hp) - np.matmul(np.matmul(kn, Kminv), kn.transpose())[0, 0] + self.sigma_n ** 2
            # Invert the value
            ldiag[i] = 1.0 / ldiag[i]

        kmnc_inv = Knm.transpose()
        # kmnc_inv = np.matmul(Knm.transpose(), np.diag(ldiag))

        for i in range(len(ldiag)):
            kmnc_inv[:, i] *= ldiag[i]

        Qm = Km + np.matmul(kmnc_inv, Knm)
        Qminv = la.inv(Qm)
        self.alpha = np.matmul(np.matmul(Qminv, kmnc_inv), y.transpose())

    def query(self, pt):
        '''
        Queries the mean and variance at a given point from the GP.

        :param pt: Test point
        :return: 2Mean, variance of the provided test points
        '''

        mean = None
        variance = None

        # Subset of data approximation
        if self.approx_method == 'SD':
            # Compute the covariance of the test point with all of the other points
            kst = np.zeros(len(self.X))
            for i in range(len(self.X)):
                kst[i] = self.covariance(pt, self.X[i], self.hp)

            mean = np.dot(kst.transpose(), self.alpha)
            # v = la.solve(self.L, kst)
            # variance = np.abs(self.covariance(pt, pt) - np.dot(v, v))

        # Induced input approximation
        elif self.approx_method == 'II':
            kst = np.zeros(self.M)
            for i in range(self.M):
                kst[i] = self.covariance(pt, self.Xbar[i], self.hp)

            mean = np.dot(kst.transpose(), self.alpha) # [0]
            # variance = self.covariance(pt, pt) - np.matmul(np.matmul(kst.transpose(), self.kq_inv), kst.transpose()) + self.sigma ** 2

        return mean, variance

def test_kmeans():
    size = 10
    X = np.array(np.random.rand(size, 2)) * 5
    datapoints = []
    for i in range(size * 10000):
        center = X[np.random.randint(size)]
        datapoints.append(center + np.random.normal(center, 0.4))

    datapoints = np.array(datapoints)
    centers = lloydKMeans(datapoints, 100, 20)
    plt.scatter(datapoints[:, 0], datapoints[:, 1], s=0.04)
    plt.scatter(centers[:, 0], centers[:, 1], s=3.0)
    plt.show()

def test_univariate():
    gp = GaussianProcess('RBF', [3.0], 0.7)

    X = np.array([[i * 0.1] for i in range(0, 150)])
    y = np.sin(X.transpose()) + np.random.normal(0, scale=0.15, size=len(X))
    y = y[0]

    centers = lloydKMeans(X[:, 0:2], 10, 30)

    gp.approx_method='II'
    print(gp.compute_likelihood(X, y, centers))
    gp.learn_hyperparameters(X[:, 0:2], y, 'II', centers)
    print(gp.compute_likelihood(X, y, centers))

    gp.build_ii(X, y, len(centers), None, Xbar=centers)
    X_pred = np.array([i * 0.01 for i in range(-50, 1500)])
    y_pred = np.array([gp.query(x)[0] for x in X_pred]) # Add indexing [0] after adding in variance computation

    # variances = np.array([gp.query(x)[1] for x in X_pred])
    # upperBound = y_pred + np.sqrt(variances) * 2
    # lowerBound = y_pred - np.sqrt(variances) * 2

    plt.scatter(np.ravel(X), np.ravel(y), s=0.4)
    plt.plot(X_pred, y_pred)

    # plt.plot(X_pred, upperBound)
    # plt.plot(X_pred, lowerBound)

    plt.show()

def test_multivariate():
    # Construction of ground-truth
    # ================================================
    def fcn(X):
        return np.sin((X[0] + X[1]) / 300) * 3

    testRange = np.array([[0, 1000], [0, 3000]])
    numTest = 100
    noiseScale = 0.8
    arr = np.zeros((numTest, 3))

    for i in range(numTest):
        for j in range(2):
            arr[i, j] = np.random.rand() * (testRange[j, 1] - testRange[j, 0]) + testRange[j, 0]

        arr[i, 2] = fcn(arr[i, 0:2]) + np.random.normal(0, noiseScale)

    plt.figure()
    plt.scatter(arr[:, 0], arr[:, 1], c=arr[:, 2])
    plt.title("Ground-truth")
    plt.colorbar()

    # ================================================


    # centers = lloydKMeans(arr[:, 0:2], 50, 30)

    gp = GaussianProcess('RBF', [4.0], 0.2)
    gp.learn_hyperparameters(arr[:, 0:2], arr[:, 2], 'SD')

    gp.build_sd(arr[:, 0:2], arr[:, 2])

    predictions = np.zeros(numTest)
    for i in range(numTest):
        predictions[i] = gp.query(arr[i, 0:2])[0]

    plt.figure()
    # plt.plot(centers[:, 0], centers[:, 1], 'ro')
    plt.scatter(arr[:, 0], arr[:, 1], c=predictions)
    plt.title("Predictions")
    plt.colorbar()

    plt.show()

if __name__ == '__main__':
    # test_kmeans()
    test_univariate()
    # test_multivariate()
    # test_multivariate()
    # test_covariance_approximation()