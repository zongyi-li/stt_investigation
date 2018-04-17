import numpy as np
import progressbar

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

        for i in range(self.dim - 2):
            self.approx.append(np.array(np.random.rand(rank, len(gridIndices[i + 1]), rank)))

        self.approx.append(np.random.rand(rank, len(gridIndices[-1]), 1))

        # Scale down the random values
        for mat in self.approx:
           mat -= 0.5

    def recurse(self, idx, curpoint):
        if idx == self.dim:
            self.sgd_update(curpoint.astype(int))
        else:
            # Here, shuffle the points in some random order for the SGD
            indices = [i for i in range(len(self.gridIndices[idx]))]
            np.random.shuffle(indices)

            for i in indices:
                curpoint[idx] = i
                self.recurse(idx + 1, curpoint)

    def get(self, point):
        res = np.identity(1)

        for i in range(self.dim):
            res = np.matmul(res, self.approx[i][:, int(point[i]), :])

        return res[0, 0]

    def compute_error(self):
        return self.err_recurse(0, np.zeros(self.dim)) / np.prod([len(x) for x in self.gridIndices])

    def err_recurse(self, idx, point):
        err = 0
        if idx == self.dim:
            gridpt = [self.gridIndices[i][int(point[i])] for i in range(self.dim)]
            fval = self.fcn(
                gridpt)
            err = (fval - self.get(point)) ** 2
            # print("Point: {}, fval: {}, approx: {}".format(gridpt, fval, self.get(point)))
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

        self.reversePartials = [np.identity(1)]
        # Precompute matrix products so we don't redo the same calculation over and over
        for i in reversed(range(self.dim)):
            self.reversePartials.append(np.matrix(self.approx[i][:, point[i], :]) * self.reversePartials[-1])

        grads = []
        self.reversePartials.reverse()

        # U = self.approx[0][0, point[0], :]
        # V = self.approx[1][:, point[1], 0]

        # gradU = self.eta * (-V.T * (fval - np.dot(U, V)))
        # gradV = self.eta * (-U.T * (fval - np.dot(U, V)))

        # print(self.approx[1])

        for i in range(self.dim):
            # Compute the approximation
            y_approx = np.matmul(forward_partial, self.reversePartials[i])[0, 0]

            # Compute the tensor product for use in the gradient.
            outerprod = np.outer(forward_partial, self.reversePartials[i + 1])
            # ALS / SGD update: the term w/ coefficient self.lda is regularization w/ Froebenius norm
            grads.append(- self.eta * ((fval - y_approx) * np.matrix(outerprod)))

            # Update the forward partial matrix
            forward_partial = np.matmul(forward_partial, np.matrix(self.approx[i][:, point[i], :]))

        for i in range(len(grads)):
            # Debugging purposes only...
            # print("Computed Gradient: ")
            # print(grads[i])
            #if i == 0:
            #    print(gradU)
            #else:
            #    print(gradV)
            self.approx[i][:, point[i], :] -= grads[i]

        #self.approx[0][0, point[0], :] -= gradU
        #self.approx[1][:, point[1], 0] -= gradV


    def build(self, num_iter):
        for i in progressbar.progressbar(range(num_iter)):
            self.recurse(0, np.zeros(self.dim))


def polytest(params):
    return 2 * params[0] + 3 * params[1] ** 3 + 2 * params[2] ** 2

grid = [[1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0],
        [1.0, 2.0, 3.0, 4.0]]

if __name__ == '__main__':
    test = als_tt(3, grid, polytest, 0.0005, 0.0000)

    print(test.compute_error())
    test.build(500)
    print(test.compute_error())

    print("=============")



