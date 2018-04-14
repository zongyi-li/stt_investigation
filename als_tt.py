import numpy as np

class als_tt:
    def __init__(self, rank, gridIndices, fcn, eta, lda):
        '''
        gridIndices must have dimension at least 2.

        lda: The regularization parameter
        eta: The learning rate for SGD
        '''
        self.eta = eta
        self.lda = lda
        self.rank = rank
        self.gridIndices = gridIndices
        self.fcn = fcn


        self.approx = []
        self.dim = len(gridIndices)
        self.approx.append(np.random.rand(1, len(gridIndices[0]), rank))

        # Can fix this to preallocate memory, since we know the tensor dimensions in advance
        self.reversePartials = [np.identity(1)]

        for i in range(self.dim - 2):
            self.approx.append(np.array(np.random.rand(rank, len(gridIndices[i + 1]), rank)))

        self.approx.append(np.random.rand(rank, len(gridIndices[-1]), 1))

    def recurse(self, idx, curpoint):
        if idx == self.dim:
            self.sgd_update(curpoint)
        else:
            # Here, shuffle the points in some random order for the SGD
            indices = np.random.shuffle([i for i in range(len(self.gridIndices[idx]))])

            for i in indices:
                curpoint[idx] = i
                self.recurse(idx + 1, curpoint)

    def get(self, point):
        res = np.identity(1)

        for i in range(self.dim):
            res = res * self.approx[i][:, point[i], :]

        return res[0, 0]

    def compute_error(self):
        return self.err_recurse(0, np.zeros(self.dim))

    def err_recurse(self, idx, point):
        err = 0
        if idx == self.dim:
            print(self.gridIndices)
            gridpt = [self.gridIndices[i][0] for i in range(self.dim)]
            print(gridpt)
            fval = self.fcn(gridpt)
            err = (fval - self.get(point)) ** 2
        else:
            for i in range(len(self.gridIndices[idx])):
                point[idx] = i
                err += self.err_recurse(idx + 1, point)

        return err

    def sgd_update(self, point):
        '''
        Performs a single iteration of the ALS SGD update with respect
        to a single point in the tensor, whose indices are specified in
        the argument 'point'.
        '''
        forward_partial = np.identity(1)
        gridpt = [self.gridIndices[i][point[i]] for i in range(self.dim)]
        fval = self.fcn(gridpt)

        # Precompute matrix products so we don't redo the same calculation over and over
        for i in reversed(range(self.dim)):
            self.reversePartials.append(np.matrix(self.approx[i][:, point[i], :]) * self.reversePartials[-1])

        self.reversePartials.reverse()

        for i in range(self.dim):
            # Compute the approximation
            y_approx = (forward_partial * self.reversePartials[i])[0, 0]

            # Compute the tensor product for use in the gradient.
            outerprod = np.outer(forward_partial, self.reversePartials[i + 1])

            # ALS / SGD update: the term w/ coefficient self.lda is regularization
            self.approx[i][:, point[i], :] -= self.eta * ((y_approx - fval) * np.matrix(outerprod) + (self.lda * np.matrix(self.approx[i][:, point[i], :])))

            # Update the forward partial matrix
            forward_partial = forward_partial * np.matrix(self.approx[i][:, point[i], :])

    def build(self, num_iter):
        for i in range(num_iter):
            self.recurse(0, np.zeros(self.dim))


def polytest(params):
    return params[0] ** 2 + params[1] ** 2 + params[2] ** 2

grid = [[0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7],
        [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7]]

if __name__ == '__main__':
    test = als_tt(3, grid, polytest, 0.2, 0.001)
    # print(test.approx)
    # test.build(1)
    test.sgd_update(np.array([1, 1, 1, 1]))
    print(test.compute_error())
    # Compute the approximation error

