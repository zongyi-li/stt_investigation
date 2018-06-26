# Contains testing utilities for tensor-train code

import numpy as np
import numpy.linalg as la

class Grid:
    def __init__(self, gridIndices):
        # Note: this explicit initialization is only temporary...
        self.gridIndices = gridIndices

        self.dim = len(self.gridIndices)

    def applyFunction(self, fcn):
        '''
        Helper that applies the specified function at every point on the grid.


        :param fcn: The function to apply
        :return void
        '''
        self.recurse(0, np.zeros(self.dim), fcn)

    def recurse(self, idx, curpoint, fcn, shuffle=False):
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

            if shuffle:
                np.random.shuffle(indices)

            for i in indices:
                curpoint[idx] = i
                self.recurse(idx + 1, curpoint, fcn)

    def pointArrayHelper(self, idx, curpoint, arr, arrIdx):
        if idx == self.dim:
            arr[arrIdx[0]] = curpoint
            arrIdx[0] += 1
        else:
            # Here, shuffle the points in some random order for the SGD
            indices = [i for i in range(len(self.gridIndices[idx]))]

            for i in indices:
                curpoint[idx] = self.gridIndices[idx][i]
                self.pointArrayHelper(idx + 1, curpoint, arr, arrIdx)

    def getPointArray(self):
        arrIdx = [0]
        gridDims = [len(x) for x in self.gridIndices]
        arr = np.zeros((np.prod(gridDims), len(gridDims)))
        self.pointArrayHelper(0, np.zeros(len(gridDims)), arr, arrIdx)
        return arr


# Define a very simple test function, and a utility to return grids of varying sizes
def polytest4(params):
    return 2 * params[0] ** 2 + 3 * params[1] * params[2] ** 2 + 5 * params[2] ** 2 * params[3] ** 4

def polytest2(params):
    return params[0] ** 2 + params[0] * params[1] ** 2 + params[1] ** 3

def getEquispaceGrid(n_dim, rng, subdivisions):
    '''
    Returns a grid of equally-spaced points in the specified number of dimensions

    n_dim       : The number of dimensions to construct the tensor grid in
    rng         : The maximum dimension coordinate (grid starts at 0)
    subdivisions: Number of subdivisions of the grid to construct
    '''
    return Grid(np.array([np.array(range(subdivisions + 1)) * rng * 1.0 / subdivisions for i in range(n_dim)]))