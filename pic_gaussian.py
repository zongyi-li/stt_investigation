import numpy as np
import matplotlib.pyplot as plt
import numpy.linalg as la
import scipy.linalg as sla
from progressbar import progressbar
from scipy.spatial import KDTree
from joblib import Parallel, delayed
from updatable_pq import UpdatablePQ

def sq_dist(a, b):
    a_min_b = a - b
    return np.einsum('i,i->', a_min_b, a_min_b)

def blockDiagMult(blocks, other, left=True):
    '''
    Multiplies the block diagonal matrix specified in the given array by the

    :param blocks: List of square blocks
    :param rhs:
    :param left: Which side to multiply on. Left -> blocks * other. right -> other * blocks
    :return:
    '''
    idx = 0
    N = np.sum([np.shape(blocks[i])[0] for i in range(len(blocks))])
    res = None

    if left:
        M = np.shape(other)[1]
        res = np.zeros((N, M))

        for i in range(len(blocks)):
            res[idx : idx + len(blocks[i])] = blocks[i] * other[idx : idx + len(blocks[i])]
            idx += len(blocks[i])
    else:
        M = np.shape(other)[0]
        res = np.zeros((M, N))

        for i in range(len(blocks)):
            res[:, idx : idx + len(blocks[i])] = other[:, idx : idx + len(blocks[i])] * blocks[i]
            idx += len(blocks[i])

    return np.matrix(res)


class PICGaussian:
    def __init__(self, covariance_type, hp, y_noise):
        self.hp = hp
        self.sigma_n = y_noise
        self.covariance_type = covariance_type

        # Related to PITC blocking
        self.centers = None

        self.center_tree = None
        self.clusterDist = None

        self.clusterStarts = None
        self.clusterLengths = None

        self.trained = False

        # Related to GP training
        self.Km = None

        if covariance_type == 'RBF':
            # Hyperparameters:
            # hp[0]: sigma_rbf (value in covariance)
            self.covariance = lambda x, y: np.exp(-0.5 / (hp[0] * hp[0]) * sq_dist(x, y))

    def cluster(self, X, numClusters):
        return self.FPC_cluster(X, numClusters)

    def randomCluster(self, X, numClusters):
        return X[np.random.choice(np.array(range(len(X))), replace=False, size=numClusters)]

    def FPC_cluster(self, X, numClusters):
        '''
        Performs furthest-point clustering. Greedy algorithm is within a factor of 2 of the optimal.

        :param X: The set of points to cluster
        :param numClusters: The number of clusters
        :return: The set of cluster centers
        '''
        print("Performing clustering...")

        clusters = [X[np.random.randint(len(X))]] # Random seed point
        clusterDists = np.array([np.inf for _ in range(len(X))])

        dim = np.shape(X)[1]

        for i in progressbar(range(numClusters)):
            maxIdx = 0
            for j in range(len(X)):
                cc = clusters[-1]
                # Take the square of the Euclidean Distance
                latestDist = 0
                for k in range(dim):
                    latestDist += (cc[k] - X[j][k]) * (cc[k] - X[j][k])

                if latestDist < clusterDists[j]:
                    clusterDists[j] = latestDist

                if clusterDists[j] > clusterDists[maxIdx]:
                    maxIdx = j

            clusters.append(X[maxIdx])

        print("Clustering complete!")
        return clusters

    def sp_cluster(self, X, numClusters):
        pass

    def optimizedFPC(self, X, numClusters):
        '''
        Performs optimized furthest-point clustering using a priority queue and a K-D tree for fast lookups.
        :param X: The points to cluster
        :param numClusters: Number of clusters
        :return: The clustered points
        '''
        print("Performing clustering...")
        tree = KDTree(X, 10)
        pq = UpdatablePQ()

        clusters = [X[np.random.randint(len(X))]]  # Random seed point

        maxsqDist = max(sq_dist(clusters[0], x) for x in X)

        for i in range(numClusters):
            neighbor_idx = tree.query_ball_point(clusters[-1], np.sqrt(maxsqDist))
            for neighbor in neighbor_idx:
                newDist = sq_dist(neighbor, clusters[-1])
                if newDist < -pq.entry_finder(neighbor)[0]:
                    pq.add_task()

        print("Clustering complete!")

    def getClosestBlockIdx(self, x):
        '''
        Returns the index of the block that this point belongs to.

        :param x:
        :return:
        '''
        # Assuming there is at least 1 center
        _, loc = self.center_tree.query([x])
        return loc[0]

    def getBlocks(self, X, y, numBlocks):
        '''
        Place the input points into groups.

        :param X: The input points
        :return: Blocked off points
        '''
        self.centers = self.cluster(X, numBlocks)
        self.clusterLengths = np.zeros(len(self.centers), dtype=int)

        # Place the clusters in a K-D tree for easy access
        self.center_tree = KDTree(self.centers)

        clusterPoints = []
        clusterTargets = []
        for i in range(len(self.centers)):
            clusterPoints.append([])
            clusterTargets.append([])

        # Now rearrange the data to correspond with the blocks

        print("Finding Closest Clusters for Datapoints...")
        for i in progressbar(range(len(X))):
            cc = self.getClosestBlockIdx(X[i])

            clusterPoints[cc].append(X[i])
            clusterTargets[cc].append(y[i])
            self.clusterLengths[cc] += 1

        X = np.concatenate(clusterPoints)
        y = np.concatenate(clusterTargets)
        self.clusterStarts = np.concatenate(([0], np.cumsum(self.clusterLengths, dtype=int)))

        print("Points blocked off!")

        return X, y

    def computePIC(self, X, y, induced):
        '''
        Use Woodbury's matrix inversion lemma to compute the low-rank inverse, implementing PITC
        for use in PIC

        :param X: Input x points
        :param y: Targets
        :param induced: Inducing input locations

        :return: p, defined as in the paper
        '''
        self.X = X
        self.induced = induced

        self.Knm = np.zeros((len(X), len(induced)))
        self.Km = np.zeros((len(induced), len(induced)))

        for i in range(len(self.Km)):
            for j in range(len(self.Km)):
                self.Km[i, j] = self.covariance(induced[i], induced[j])

        for i in range(len(X)):
            for j in range(len(induced)):
                self.Knm[i, j] = self.covariance(X[i], induced[j])

        self.Knm = np.matrix(self.Knm)
        self.Km = np.matrix(self.Km)
        self.Kminv = la.inv(self.Km)

        blocks = [np.zeros((self.clusterLengths[i], self.clusterLengths[i])) for i in range(len(self.clusterLengths))]

        # Compute the blocks
        print("Computing block inverses...")
        for i in progressbar(range(len(self.clusterStarts)-1)):
            # Compute the block produced by Kn
            for j in range(self.clusterStarts[i], self.clusterStarts[i+1]):
                for k in range(self.clusterStarts[i], self.clusterStarts[i + 1]):
                    blocks[i][j - self.clusterStarts[i], k - self.clusterStarts[i]] = self.covariance(X[j], X[k])

            # Subtract off the block Qn
            partial = np.matrix(self.Knm[self.clusterStarts[i]: self.clusterStarts[i + 1]])
            blocks[i] -= partial * self.Kminv * partial.transpose()

            # Add regularization terms
            for j in range(self.clusterLengths[i]):
                blocks[i][j, j] += self.sigma_n ** 2

            # Invert each of the blocks to get A^{-1}... hooray
            blocks[i] = la.inv(blocks[i])

        # Now perform the PITC matrix inversion using Woodbury's Lemma
        print("Computing low-rank inverse with Woodbury's Lemma")

        invCenter = la.inv(self.Km + self.Knm.transpose() * blockDiagMult(blocks, self.Knm))
        self.pvec = -la.multi_dot([
            blockDiagMult(blocks, self.Knm),
            blockDiagMult(blocks, invCenter * self.Knm.transpose(), left=False),
            y
        ])
        self.pvec += blockDiagMult(blocks, y)

        print("Low-rank inverse computation complete!")

        self.wBlocks = []
        self.vBlocks = []

        self.wM = np.matrix(np.zeros((len(self.induced), 1)))

        for i in range(len(self.centers)):
            # Precomputation for mean prediction
            Kmb = np.matrix(np.zeros(((len(induced), self.clusterLengths[i]))))

            for j in range(len(induced)):
                for k in range(self.clusterLengths[i]):
                    Kmb[j, k] = self.covariance(induced[j], X[self.clusterStarts[i] + k])

            wB = self.Kminv * Kmb * self.pvec[self.clusterStarts[i] : self.clusterStarts[i+1]]
            self.wBlocks.append(wB)
            self.wM += wB

        # Variance precomputations
        self.sum = np.matrix(np.zeros((len(self.induced), len(self.induced))))

        # Not-B, Not-B terms
        self.aBlocks = [np.matrix(np.zeros((len(self.induced), len(self.induced)))) for i in range(len(self.centers))]

        # Off-diagonal B, Not-B terms
        self.bBlocks = [np.matrix(np.zeros((self.clusterLengths[i], len(self.induced)))) for i in range(len(self.centers))]

        # B, B terms handled at prediction time

        # TODO: Need to get rid of pitc_inv!!

        # Testing purpose only
        self.pitc_inv = sla.block_diag(*blocks) - la.multi_dot([
            blockDiagMult(blocks, self.Knm),
            blockDiagMult(blocks, la.inv(
                self.Km + self.Knm.transpose() * blockDiagMult(blocks, self.Knm)) * self.Knm.transpose(), left=False)
        ])

        for i in range(len(self.centers)):
            for j in range(len(self.centers)):
                pComponent = -la.multi_dot([
                    blocks[i],
                    self.Knm[self.clusterStarts[i]: self.clusterStarts[i + 1]],
                    invCenter,
                    self.Knm.transpose()[:, self.clusterStarts[j]: self.clusterStarts[j + 1]],
                    blocks[j],
                    self.Knm[self.clusterStarts[j]: self.clusterStarts[j + 1]]
                ])

                aBlock_delta = self.Knm.transpose()[:,
                                   self.clusterStarts[i]: self.clusterStarts[i + 1]] * pComponent

                if i == j:
                    aBlock_delta += la.multi_dot([
                        self.Knm.transpose()[:, self.clusterStarts[i]: self.clusterStarts[i + 1]],
                        blocks[i],
                        self.Knm[self.clusterStarts[j]: self.clusterStarts[j + 1]]
                    ])

                self.aBlocks[i] += aBlock_delta
                self.sum += aBlock_delta

                if i != j:
                    self.aBlocks[j] += aBlock_delta
                    self.bBlocks[i] += \
                        2 * pComponent

        # Conjugate w/ Km^(-1)
        for i in range(len(self.centers)):
            self.aBlocks[i] = self.Kminv * self.aBlocks[i] * self.Kminv
            self.bBlocks[i] = self.bBlocks[i] * self.Kminv

        self.sum = self.Kminv * self.sum * self.Kminv

        self.trained = True
        print("GP Approximation Complete!")

    def train(self, X, y, numInduced, numBlocks):
        # For now, just use the cluster points as the induced inputs...
        print("Training Gaussian Process with PIC approximation...")
        induced = self.cluster(X, numInduced)

        # Place the training inputs into blocks
        blockedX, blockedY = self.getBlocks(X, y, numBlocks)
        blockedY = np.matrix(blockedY).transpose()

        self.computePIC(blockedX, blockedY, induced)

    def predict_pitc(self, X):
        kst = np.zeros((len(X), len(self.Km)))

        for i in range(len(X)):
            for j in range(len(self.Km)):
                kst[i, j] = self.covariance(X[i], self.induced[j])

        return kst * self.pvec_augmented

    def predict(self, X, computeVariance=False):
        predictions = np.zeros(len(X))
        variances = np.zeros(len(X))

        kstM = np.matrix(np.zeros((1, len(self.induced))))

        Qn = self.Kminv * self.Knm.transpose()

        for i in progressbar(range(len(X))):
            # Compute covariances with the inducing inputs
            for j in range(len(self.Km)):
                kstM[0, j] = self.covariance(X[i], self.induced[j])

            # Find the closest block
            cc = self.getClosestBlockIdx(X[i])
            kstB = np.matrix(np.zeros((1, self.clusterLengths[cc])))

            # Compute covariances with the points in the block
            for j in range(self.clusterLengths[cc]):
                kstB[0, j] = self.covariance(X[i], self.X[self.clusterStarts[cc] + j])

            predictions[i] = kstM * (self.wM - self.wBlocks[cc]) + kstB * self.pvec[self.clusterStarts[cc] : self.clusterStarts[cc+1]]

            if computeVariance:
                variances[i] = self.covariance(X[i], X[i]) + self.sigma_n ** 2 \
                    - kstM * (self.sum - self.aBlocks[cc]) * kstM.transpose() \
                    - kstB * self.bBlocks[cc] * kstM.transpose() \
                    - kstB * self.pitc_inv[self.clusterStarts[cc] : self.clusterStarts[cc + 1], self.clusterStarts[cc] : self.clusterStarts[cc + 1]] * kstB.transpose()

        if computeVariance:
            return predictions, variances
        else:
            return predictions

def test_univariate():
    gp = PICGaussian('RBF', [2.0], 0.13)

    X = np.concatenate((np.array([[i * 0.1] for i in range(0, 50)]), np.array([[i * 0.1] for i in range(90, 100)])))
    y = np.array(np.sin(X.transpose()) + np.random.normal(0, scale=0.15, size=len(X)).transpose())[0]

    gp.train(X, y, 9, 3)

    X_pred = np.array([[i * 0.01] for i in range(-50, 1500)])
    predictions, variances = gp.predict(X_pred, computeVariance=True)

    x_array = X_pred.transpose()[0]
    plt.plot(x_array, predictions, 'k-')
    plt.fill_between(x_array, predictions - 2 * np.sqrt(variances), predictions + 2 * np.sqrt(variances))
    plt.scatter(X, y)
    plt.show()

if __name__ == '__main__':
    test_univariate()