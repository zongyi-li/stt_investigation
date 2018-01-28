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
# Copyright (C) 2014-2016 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Author: Daniele Bigoni
#

import logging
import sys

from TensorToolbox.unittests.auxiliary import bcolors, print_ok, print_fail, print_summary

def run(maxprocs, PLOTTING=False, loglev=logging.WARNING):

    logging.basicConfig(level=loglev)

    import numpy as np
    import numpy.linalg as npla
    import itertools
    import time

    import TensorToolbox as DT
    import TensorToolbox.multilinalg as mla

    if PLOTTING:
        from matplotlib import pyplot as plt

    nsucc = 0
    nfail = 0

    #####################################################################################
    # Test matrix-vector product by computing the matrix-vector product
    #####################################################################################
    span = np.array([0.,1.])
    d = 2
    N = 16
    h = 1/float(N-1)
    eps = 1e-10

    # sys.stdout.write("Matrix-vector: Laplace  N=%4d   , d=%3d      [START] \n" % (N,d))
    # sys.stdout.flush()

    # Construct 2D Laplace (with 2nd order finite diff)
    D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    D[0,0:3] = np.array([1./(3.*h**2.),-2./(3.*h**2.),1./(3.*h**2.)])
    D[-1,-3:] = -np.array([1./(3.*h**2.),-2./(3.*h**2.),1./(3.*h**2.)])
    I = np.eye(N)
    FULL_LAP = np.zeros((N**d,N**d))
    for i in range(d):
        tmp = np.array([[1.]])
        for j in range(d):
            if i != j: tmp = np.kron(tmp,I)
            else: tmp = np.kron(tmp,D)
        FULL_LAP += tmp

    # Construction of TT Laplace operator
    CPtmp = []
    # D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    # I = np.eye(N)
    D_flat = D.flatten()
    I_flat = I.flatten()
    for i in range(d):
        CPi = np.empty((d,N**2))
        for alpha in range(d):
            if i != alpha:
                CPi[alpha,:] = I_flat
            else:
                CPi[alpha,:] = D_flat
        CPtmp.append(CPi)

    CP_lap = DT.Candecomp(CPtmp)
    TT_LAP = DT.TTmat(CP_lap,nrows=N,ncols=N)
    TT_LAP.build(eps)
    TT_LAP.rounding(eps)
    CPtmp = None
    CP_lap = None

    # Construct input vector
    X = np.linspace(span[0],span[1],N)
    SIN = np.sin(X)
    I = np.ones((N))
    FULL_SIN = np.zeros((N**d))
    for i in range(d):
        tmp = np.array([1.])
        for j in range(d):
            if i != j: tmp = np.kron(tmp,I)
            else: tmp = np.kron(tmp,SIN)
        FULL_SIN += tmp

    if PLOTTING:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        (XX,YY) = np.meshgrid(X,X)
        fig = plt.figure()
        if d == 2:
            # Plot function
            ax = fig.add_subplot(221,projection='3d')
            ax.plot_surface(XX,YY,FULL_SIN.reshape((N,N)),rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            plt.show(block=False)

    # Construct TT input vector
    CPtmp = []
    for i in range(d):
        CPi = np.empty((d,N))
        for alpha in range(d):
            if i != alpha: CPi[alpha,:] = I
            else: CPi[alpha,:] = SIN
        CPtmp.append(CPi)

    CP_SIN = DT.Candecomp(CPtmp)
    W = [np.ones(N,dtype=float)/float(N) for i in range(d)]
    TT_SIN = DT.WTTvec(CP_SIN,W)
    TT_SIN.build()
    TT_SIN.rounding(eps)

    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(222,projection='3d')
        ax.plot_surface(XX,YY,TT_SIN.to_tensor(),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

    # Apply full laplacian
    FULL_RES = np.dot(FULL_LAP,FULL_SIN)
    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(223,projection='3d')
        ax.plot_surface(XX,YY,FULL_RES.reshape((N,N)),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

    # Apply TT laplacian
    TT_RES = mla.dot(TT_LAP,TT_SIN)
    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(224,projection='3d')
        ax.plot_surface(XX,YY,TT_RES.to_tensor(),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

    # Check results
    if not np.allclose(FULL_RES,TT_RES.to_tensor().flatten()):
        print_fail("2.1 Matrix-vector: Laplace  N=%4d   , d=%3d" % (N,d))
        nfail += 1
    else:
        print_ok("2.1 Matrix-vector: Laplace  N=%4d   , d=%3d" % (N,d))
        nsucc += 1

    #####################################################################################
    # Test matrix-vector product by computing the matrix-vector product of randomly generated input
    #####################################################################################
    span = np.array([0.,1.])
    d = 3
    nrows = [16,20,24]
    ncols = [16,12,14]
    if isinstance(nrows,int): nrows = [nrows for i in range(d)]
    if isinstance(ncols,int): ncols = [ncols for i in range(d)]
    eps = 1e-10

    # sys.stdout.write("Matrix-vector: Random\n  nrows=[%s],\n  ncols=[%s],  d=%3d      [START] \n" % (','.join(map(str,nrows)),','.join(map(str,ncols)),d))
    # sys.stdout.flush()

    # Construction of TT random matrix
    TT_RAND = DT.randmat(d,nrows,ncols)

    # Construct FULL random tensor
    FULL_RAND = TT_RAND.to_tensor()
    import itertools
    rowcol = list(itertools.chain(*[[ri,ci] for (ri,ci) in zip(nrows,ncols)]))
    FULL_RAND = np.reshape(FULL_RAND,rowcol)
    idxswap = list(range(0,2*d,2))
    idxswap.extend(range(1,2*d,2))
    FULL_RAND = np.transpose(FULL_RAND,axes=idxswap)
    FULL_RAND = np.reshape(FULL_RAND,(np.prod(nrows),np.prod(ncols)))

    # Construct TT random vector
    TT_VEC = DT.randvec(d,ncols)

    # Construct FULL random vector
    FULL_VEC = TT_VEC.to_tensor().flatten()

    # Apply TT
    TT_RES = mla.dot(TT_RAND,TT_VEC)

    # Apply FULL
    FULL_RES = np.dot(FULL_RAND,FULL_VEC)

    # Check results
    if not np.allclose(FULL_RES,TT_RES.to_tensor().flatten()):
        print_fail("2.2 Matrix-vector: Random  N=%4d   , d=%3d" % (N,d),'')
        nfail += 1
    else:
        print_ok("2.2 Matrix-vector: Random  N=%4d   , d=%3d" % (N,d))
        nsucc += 1

    print_summary("TT Matrix-Vector", nsucc, nfail)
    
    return (nsucc,nfail)

if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestTT.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs,PLOTTING=True, loglev=logging.INFO)
