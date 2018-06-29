# Combines Gaussian Processes with the Tensor Train Decomposition

# Numpy and linear algebra libraries
import numpy as np
import numpy.linalg as la
import matplotlib.pyplot as plt

import csv

import tensorly as tl
import itertools

import TensorToolbox.multilinalg as mla
import TensorToolbox as DT
from TensorToolbox.core import STT
import SpectralToolbox

FILENAME = 'data/usgs_bath_data/capstone_data_sample.csv'
data=[]

# Angle to rotate the river data by
ROT_ANGLE = 0.13 * np.pi

# Rotate to align river with axes
ROT_MATRIX = np.array([
        [np.cos(ROT_ANGLE), -np.sin(ROT_ANGLE),  0],
        [np.sin(ROT_ANGLE),  np.cos(ROT_ANGLE),  0],
        [0,                  0,                  1]
        ], dtype=float)


def import_data():
    global data
    with open(FILENAME) as csvfile:
        filereader = csv.reader(csvfile, delimiter=',')
        header = True
        for row in filereader:
            if header:
                header = False
            else:
                data.append([float(row[1]), float(row[2]), float(row[4])])

    data = np.array(data, dtype=float)
    data = data.transpose()

def preprocess(data):
    '''
    Translate / rotate the data, normalize it so that it has mean 0.
    :param data:
    :return:
    '''
    data[0, :] -= min(data[0, :])
    data[1, :] -= min(data[1, :])

def plotData(data):
    print("Plotting data...")
    plt.plot(data[0, :], data[1, :])
    plt.show()

if __name__ == '__main__':
    import_data()
    data = np.matmul(ROT_MATRIX, data)
    preprocess(data)
    print(data)
    plotData(data)