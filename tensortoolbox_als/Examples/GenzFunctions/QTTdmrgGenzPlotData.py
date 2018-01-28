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
import os

import operator
import time

import cPickle as pkl

import random

import numpy as np
import numpy.linalg as npla
import numpy.random as npr

from scipy import stats
from scipy import sparse as scsp

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

plt.close('all')
fsize = (6,4.5)
DAT_FOLDER = "./Data-qttdmrg"
FIG_FOLDER = "./Fig-tmp-qttdmrg"
IS_STORING = True
IS_PLOTTING = False

def plot_test(FNUM, GenzNormalized=False, SpectralType='Projection'):

    #########################################
    # Genz functions
    #########################################
    if SpectralType == "Projection": 
        sizes = np.arange(1,16)
    elif SpectralType == "PolyInterp":
        sizes = np.arange(1,16)
    elif SpectralType == "LinInterp":
        sizes = 2**np.arange(1,7)

    if FNUM == 0:
        FUNC = 0
        ds = [10,50,100]
        # ds = [10]
    elif FNUM == 1:
        FUNC = 1
        ds = [10,15,20]
    elif FNUM == 2:
        FUNC = 2
        ds = [10]
        # ds = [5,10,15]
    elif FNUM == 3:
        FUNC = 3
        ds = [10,50,100]
    elif FNUM == 4:
        FUNC = 4
        ds = [10,50,100]
    elif FNUM == 5:
        FUNC = 5
        ds = [10,15,20]

    print "Function: " + str(FUNC) + " Norm: " + str(GenzNormalized) + " Dims: " + str(ds) + " Type: " + SpectralType
    
    colmap = plt.get_cmap('jet')
    cols = [colmap(i) for i in np.linspace(0, 1.0, len(ds))]
    xspan = [0,1]

    if not GenzNormalized:
        file_name_ext = "New"
    else: 
        file_name_ext = ""


    names = ["Oscillatory","Product Peak","Corner Peak", "Gaussian", "Continuous", "Discontinuous"]
    file_names = ["Oscillatory","ProductPeak","CornerPeak", "Gaussian", "Continuous", "Discontinuous"]

    # Iter size1D
    path = DAT_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "data.pkl"
    if not os.path.isfile(path):
        return

    ff = file(path,'rb')
    d = pkl.load(ff)
    ff.close()

    L2err = d['L2err']
    feval = d['feval']
    N_EXP = L2err.shape[1]

    plt.figure(figsize=fsize)
    for i_d,d in enumerate(ds):
        for n_exp in range(N_EXP):
            plt.semilogy(sizes,L2err[i_d,n_exp,:],'.',color=cols[i_d])
        plt.semilogy(sizes,np.mean(L2err[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
    plt.xlabel('Order')
    plt.ylabel('L2err')
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.title(names[FUNC])
    plt.legend(loc='best')
    if IS_STORING:
        path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "OrdVsL2err"
        plt.savefig(path + ".png", format="png")
        plt.savefig(path + ".pdf", format="pdf")
        plt.savefig(path + ".eps", format="eps")

    plt.figure(figsize=fsize)
    for i_d,d in enumerate(ds):
        for n_exp in range(N_EXP):
            plt.semilogy(feval[i_d,n_exp,:],L2err[i_d,n_exp,:],'.',color=cols[i_d])
        plt.semilogy(np.mean(feval[i_d,:,:],axis=0),np.mean(L2err[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
    plt.xlabel('# func. eval')
    plt.ylabel('L2err')
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.title(names[FUNC])
    plt.legend(loc='best')

    plt.figure(figsize=fsize)
    for i_d,d in enumerate(ds):
        for n_exp in range(N_EXP):
            plt.loglog(feval[i_d,n_exp,:],L2err[i_d,n_exp,:],'.',color=cols[i_d])
        plt.loglog(np.mean(feval[i_d,:,:],axis=0),np.mean(L2err[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
    plt.xlabel('# func. eval')
    plt.ylabel('L2err')
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.title(names[FUNC])
    plt.legend(loc='best')
    if IS_STORING:
        path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "convergence"
        plt.savefig(path + ".png", format="png")
        plt.savefig(path + ".pdf", format="pdf")
        plt.savefig(path + ".eps", format="eps")

    plt.figure(figsize=fsize)
    for i_d,d in enumerate(ds):
        for n_exp in range(N_EXP):
            plt.semilogy(sizes,feval[i_d,n_exp,:],'.',color=cols[i_d])
        plt.semilogy(sizes,np.mean(feval[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
    plt.xlabel('Order')
    plt.ylabel('# func. eval')
    plt.subplots_adjust(bottom=0.15)
    plt.grid(True)
    plt.title(names[FUNC])
    plt.legend(loc='best')
    if IS_STORING:
        path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "OrdVsFeval"
        plt.savefig(path + ".png", format="png")
        plt.savefig(path + ".pdf", format="pdf")
        plt.savefig(path + ".eps", format="eps")

    if IS_PLOTTING:
        plt.show(block=False)

if __name__ == "__main__":
    FNUM = int(sys.argv[1])
    GenzNorm = (sys.argv[2] == 'True')
    Type = sys.argv[3]
    plot_test(FNUM,GenzNorm,Type)
