import numpy as np
import matplotlib.pyplot as plt
import timeit

from tensor_train import tensor_train

def plotError(errValues):
    plt.plot(errValues[:, 0], errValues[:, 1], 'r-')
    plt.xlabel('Epochs')
    plt.ylabel('RMSE')
    plt.show()

def synth_tensor_test_rmsprop(gridSize, nDims, rank, random_range):
    # Re-use the tensor-train class to generate a synthetic high-dimensional tensor
    # whose tensor-train decomposition is filled with random values
    grid = [np.array(range(gridSize))] * nDims
    tt_ground_truth = tensor_train(rank, grid, None, None, None)

    for i in range(nDims):
        tt_ground_truth.approx[i] *= random_range

    print("Attempting to learn ground-truth tensor via approximation...")
    tt_approx = tensor_train(rank, grid, tt_ground_truth.get, 0.005, 0.000)

    # Returns the error after building the approximation
    return tt_approx.build_rmsprop(0.01)

def synth_tensor_test_sgd(gridSize, nDims, rank, random_range):
    # Re-use the tensor-train class to generate a synthetic high-dimensional tensor
    # whose tensor-train decomposition is filled with random values
    grid = [np.array(range(gridSize))] * nDims
    tt_ground_truth = tensor_train(rank, grid, None, None, None)

    for i in range(nDims):
        tt_ground_truth.approx[i] *= random_range

    print("Attempting to learn ground-truth tensor via approximation...")
    tt_approx = tensor_train(rank, grid, tt_ground_truth.get, 0.01, 0.000)

    # Returns the error after building the approximation
    return tt_approx.build_sgd(0.01)


if __name__ == '__main__':
    # Grid size [first parameter], 4-dimensional grid, rank-3 approximation,
    # random values from -5 to 5 (divide the last param by 2)
    # plotError(synth_tensor_test_sgd(10, 4, 2, 6))
    plotError(synth_tensor_test_rmsprop(15, 4, 3, 7))