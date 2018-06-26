import numpy as np
import progressbar
import cProfile

class tensor_train:
    def __init__(self, rank, gridIndices, fcn, eta, lda):
        '''
        gridIndices must have dimension at least 2.

        rank: The canonical rank of the approximation
        gridIndices: The grid to evaluate the function on
        fcn: The function to call to get the entries of the high-dimensional tensor
        lda: The regularization parameter
        eta: The learning rate for SGD
        '''
        self.eta = eta
        self.lda = lda
        self.rank = rank
        self.gridIndices = gridIndices
        self.fcn = fcn
        self.rmse = None

        self.approx = []
        self.dim = len(gridIndices)
        self.approx.append(np.random.rand(1, len(gridIndices[0]), rank))

        for i in range(self.dim - 2):
            self.approx.append(np.array(np.random.rand(rank, len(gridIndices[i + 1]), rank)))

        self.approx.append(np.random.rand(rank, len(gridIndices[-1]), 1))

        # Scale down the random values
        for mat in self.approx:
           mat -= 0.5


        # RMS Prop / SGD initialization

        # Pre-initialize the array of gradients
        self.grads = []
        self.grads.append(np.zeros((1, self.rank), dtype=float))

        for i in range(self.dim - 2):
            self.grads.append(np.zeros((self.rank, self.rank), dtype=float))

        self.grads.append(np.zeros((self.rank, 1), dtype=float))

        # Initialize the forward partial product
        self.forward_partial = np.zeros((1, self.rank), dtype=float)

        # Initialize a list for the reverse partial products
        self.reversePartials = [np.identity(1)]
        for _ in range(self.dim - 1):
            self.reversePartials.append(np.matrix(np.zeros((self.rank, 1), dtype=float)))
        self.reversePartials.append(np.identity(1))

    def get(self, point):
        '''
        Returns the value of the approximation at the specified grid point.

        :param point: The grid point to evaluate at (given as an integer-tuple)
        :return: The value of the TT-approximation at the specified grid point
        '''
        res = np.identity(1)

        for i in range(self.dim):
            res = np.matmul(res, self.approx[i][:, int(point[i]), :])

        return res[0, 0]

    def recurse(self, idx, curpoint, fcn):
        '''
        Helper for apply-function, recursively stepping through each dimension of the grid
        :param idx:
        :param curpoint:
        :param fcn:
        :return:
        '''
        if idx == self.dim:
            fcn(curpoint.astype(int))
        else:
            # Here, shuffle the points in some random order for the SGD
            indices = [i for i in range(len(self.gridIndices[idx]))]
            np.random.shuffle(indices)

            for i in indices:
                curpoint[idx] = i
                self.recurse(idx + 1, curpoint, fcn)

    def applyFunction(self, fcn):
        '''
        Helper that applies the specified function at every point on the grid.


        :param fcn: The function to apply
        :return void
        '''
        self.recurse(0, np.zeros(self.dim), fcn)

    def add_error(self, point):
        gridpt = [self.gridIndices[i][point[i]] for i in range(self.dim)]
        self.rmse += (self.get(point) - self.fcn(gridpt)) ** 2


    def sgd_update(self, point):
        '''
        Performs a single iteration of the ALS SGD update with respect
        to a single point in the tensor, whose indices are specified in
        the argument 'point'.
        '''
        pass

    def rmsprop_update(self, point):
        '''
        Performs the RMSProp update instead of Vanilla SGD. This solver should converge a lot faster than
        just SGD.
        :param point:
        :return:
        '''
        gridpt = [self.gridIndices[i][point[i]] for i in range(self.dim)]
        fval = self.fcn(gridpt)

        # Precompute matrix products so we don't redo the same calculation over and over
        for i in reversed(range(self.dim)):
            np.matmul(self.approx[i][:, point[i], :], self.reversePartials[i+1], out=self.reversePartials[i])

        gt_sq = 0
        # theta_sq = 0

        for i in range(self.dim):
            # Compute the gradient
            if i == 0:
                np.outer(np.identity(1), self.reversePartials[i + 1], out=self.grads[i])
                y_approx = np.matmul(np.identity(1), self.reversePartials[i])[0, 0]
            else:
                np.outer(self.forward_partial, self.reversePartials[i + 1], out=self.grads[i])
                y_approx = np.matmul(self.forward_partial, self.reversePartials[i])[0, 0]

            self.grads[i] *= -1.0 * (fval - y_approx)
            self.grads[i] -= self.lda * self.approx[i][:, point[i], :]

            # Add to the accumulated gradient value
            gt_sq += np.sum(np.square(self.grads[i]))

            # Update the forward partial matrix

            if i == 0:
                np.matmul(np.identity(1), self.approx[i][:, point[i], :], out=self.forward_partial)
            elif i < self.dim - 1:
                np.matmul(self.forward_partial, self.approx[i][:, point[i], :], out=self.forward_partial)

        # Perform the RMSProp gradient accumulation update
        self.e_g2 = self.gamma * self.e_g2 + (1 - self.gamma) * gt_sq

        # Perform the parameter update
        for i in range(self.dim):
            self.grads[i] *= self.eta / np.sqrt(self.e_g2 + self.epsilon)
            self.approx[i][:, point[i], :] -= self.grads[i]
            # theta_sq += np.sum(np.square(self.grads[i])) Residual leftover from Adadelta


        # Accumulate the updates; this step performed out-of-order with the one before it, see alg. 1
        # self.e_theta2 = self.gamma * self.e_theta2 + (1 - self.gamma) * theta_sq


    def build_sgd(self, tol_fraction):
        print("Constructing Approximation with SGD...\n")
        old_err = self.compute_error()
        new_err = None

        errorValues = []

        print("Initial RMSE: {}".format(old_err))
        print("Minimum Percent Error Decrease to Continue Loop Past 40 Iterations: {}%".format(tol_fraction * 100))

        i = 0
        # Keep 200 as an upper-bound on the number of iterations
        while True and i < 200:
            # Compute the error every 5 iterations, test to break the loop, and store the computed error value
            if i % 5 == 1:
                new_err = self.compute_error()
                delta_fraction = (old_err - new_err) / old_err
                print("Iteration: {}, Error: {}, Error Decrease Percent: {}.%".format(i, new_err, delta_fraction * 100))

                # Delta too low? Break the loop
                if i > 50 and delta_fraction < tol_fraction:
                    break

                old_err = new_err
                errorValues.append([i, new_err])

            self.applyFunction(self.sgd_update)
            i += 1

        print("Final RMSE: {}".format(new_err))
        print("\nApproximation Complete!")

        return np.matrix(errorValues)

    def build_rmsprop(self, tol_fraction):
        print("Constructing Approximation with RMSProp...\n")

        self.epsilon = 1e-8 # Avoid division-by-zero error terms
        self.gamma = 0.95   # RMSProp decay term

        self.e_g2 = 0
        self.e_theta2 = 0

        old_err = self.compute_error()
        new_err = None

        errorValues=[]

        print("Initial RMSE: {}".format(old_err))
        print("Minimum Percent Error Decrease to Continue Loop Past 40 Iterations: {}%".format(tol_fraction * 100))

        i = 0
        # Keep 200 as an upper-bound on the number of iterations
        while True and i < 200:
            # Compute the error every 5 iterations, test to break the loop, and store the computed error value
            if i % 5 == 1:
                new_err = self.compute_error()
                delta_fraction = (old_err - new_err) / old_err
                print("Iteration: {}, Error: {}, Error Decrease Percent: {}.%".format(i, new_err, delta_fraction * 100))

                # Delta too low? Break the loop
                if i > 40 and delta_fraction < tol_fraction:
                    break

                old_err = new_err
                errorValues.append([i, new_err])

            self.applyFunction(self.rmsprop_update)
            i += 1

        print("Final RMSE: {}".format(new_err))
        print("\nApproximation Complete!")

        return np.matrix(errorValues)

    def compute_error(self):
        self.rmse = 0
        self.applyFunction(self.add_error)
        return np.sqrt(self.rmse / np.prod([len(x) for x in self.gridIndices]))


# =================================================================================
# Small-scale testing...

# Define a very simple test function and a small grid
def polytest(params):
    return 2 * params[0] ** 2 + 3 * params[1] * params[2] ** 2 + 5 * params[2] ** 2 * params[3] ** 4

grid = [[1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0],
        [1.0, 2.0, 3.0, 3.5, 4.0, 5.0, 6.0, 7.0, 8.0, 9.0, 10.0, 11.0, 12.0, 13.0, 14.0]]

if __name__ == '__main__':
    test = tensor_train(3, grid, polytest, 0.005, 0.000)
    test.build_rmsprop(0.01)

    print("=============")