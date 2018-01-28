#
# This file is part of TensorToolbox.
#
# TensorToolbox is free software: you can redistribute it and/or modify
# it under the terms of the LGNU Lesser General Public License as published by
# the Free Software Foundation, either version 3 of the License, or
# (at your option) any later version.
#
# TensorToolbox is distributed in the hope that it will be useful,
# but WITHOUT ANY WARRANTY; without even the implied warranty of
# MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
# LGNU Lesser General Public License for more details.
#
# You should have received a copy of the LGNU Lesser General Public License
# along with TensorToolbox.  If not, see <http://www.gnu.org/licenses/>.
#
# DTU UQ Library
# Copyright (C) 2014-2015 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Author: Daniele Bigoni
#

import sys

import operator
import time

import random

import numpy as np
import numpy.linalg as npla
import numpy.random as npr

from scipy import stats
from scipy import sparse as scsp
import scipy.io as sio

import TensorToolbox as DT
import TensorToolbox.multilinalg as mla

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from UQToolbox import UncertaintyQuantification as UQ

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

import ExtPython

fsize = (6,4.5)
IS_STORING = True
FIG_FOLDER = "./Fig-tmp"
PKL_FOLDER = "./Data"
MAT_FOLDER = "./Data"
func_list = [0,1,2,3]

SpectralType = "Projection"
names = ["Oscillatory","Product Peak","Corner Peak", "Gaussian", "Continuous", "Discontinuous"]
file_names = ["Oscillatory","ProductPeak","CornerPeak", "Gaussian", "Continuous", "Discontinuous"]
file_name_ext = 'New'

for (FUNC,name),file_name,i_dat in zip(enumerate([names[i] for i in func_list]),
                                       [file_names[i] for i in func_list],
                                       [i+1 for i in func_list]):
    # Open TTcross results
    path = PKL_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "data-Patrick.pkl"
    d_tt = ExtPython.loadVariables(path)
    
    # Open Mat
    d_sml = sio.loadmat( MAT_FOLDER + "/" + "GenzResultsAdaptive" + str(i_dat) + ".mat" )
    
    plt.figure(figsize=fsize)
    plt.loglog(d_tt['feval'][0,:,:],d_tt['L2err'][0,:,:],'.',color='b')
    plt.loglog(np.mean(d_tt['feval'][0,:,:],axis=0), np.mean(d_tt['L2err'][0,:,:],axis=0), 'o-',color='b', label='STT')
    plt.loglog(d_sml['pts'],d_sml['L2err'],'.',color='r')
    plt.loglog(np.mean(d_sml['pts'],axis=0), np.mean(d_sml['L2err'],axis=0), 's-',color='r', label='Smolyak')
    plt.legend(loc='best')
    plt.xlabel('# Func. eval')
    plt.ylabel('L2err')
    plt.title(name)
    plt.show(block=False)
    
    if IS_STORING:
        path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "STTvsSG"
        plt.savefig(path + ".png", format="png")
        plt.savefig(path + ".pdf", format="pdf")
        plt.savefig(path + ".ps", format="ps")
