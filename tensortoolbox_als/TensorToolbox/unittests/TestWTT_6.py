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

    ####################################################################################
    # Test Stabilized Bi-Conjugate Gradient method on simple multidim laplace equation
    ####################################################################################

    span = np.array([0.,1.])
    d = 2
    N = 16
    h = 1/float(N-1)
    eps_cg = 1e-8
    eps_round = 1e-10

    # sys.stdout.write("Bi-Conjugate Gradient Stab.: Laplace  N=%4d   , d=%3d      [START] \n" % (N,d))
    # sys.stdout.flush()

    # Construct d-D Laplace (with 2nd order finite diff)
    D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    D[0,0:2] = np.array([1.,0.])
    D[-1,-2:] = np.array([0.,1.])
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
    TT_LAP.build()
    TT_LAP.rounding(eps_round)
    CPtmp = None
    CP_lap = None

    # Construct Right hand-side (b=1, Dirichlet BC = 0)
    X = np.linspace(span[0],span[1],N)
    b1D = np.ones(N)
    b1D[0] = 0.
    b1D[-1] = 0.
    # Construct the d-D right handside
    tmp = np.array([1.])
    for j in range(d):
        tmp = np.kron(tmp,b1D)
    FULL_b = tmp
    # Construct the TT right handside
    CPtmp = []
    for i in range(d):
        CPi = np.empty((1,N))
        CPi[0,:] = b1D
        CPtmp.append(CPi)
    CP_b = DT.Candecomp(CPtmp)
    W = [np.ones(N,dtype=float)/float(N) for i in range(d)]
    TT_b = DT.WTTvec(CP_b,W)
    TT_b.build()
    TT_b.rounding(eps_round)


    # Solve full system using npla.solve
    FULL_RES = npla.solve(FULL_LAP,FULL_b)

    if PLOTTING:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        (XX,YY) = np.meshgrid(X,X)
        fig = plt.figure(figsize=(18,7))
        plt.suptitle("BiCG-stab")
        if d == 2:
            # Plot function
            ax = fig.add_subplot(131,projection='3d')
            ax.plot_surface(XX,YY,FULL_RES.reshape((N,N)),rstride=1, cstride=1, cmap=cm.coolwarm,
                            linewidth=0, antialiased=False)
            plt.show(block=False)

    # Solve TT cg
    x0 = DT.zerosvec(d,N)
    (TT_RES,conv,info) = mla.bicgstab(TT_LAP,TT_b,x0=x0,ext_info=True)
    print(info)
    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(132,projection='3d')
        ax.plot_surface(XX,YY,TT_RES.to_tensor(),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

    # Error
    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(133,projection='3d')
        ax.plot_surface(XX,YY,np.abs(TT_RES.to_tensor()-FULL_RES.reshape((N,N))),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

    err2 = npla.norm(TT_RES.to_tensor().flatten()-FULL_RES,2)
    if err2 < 1e-2:
        print_ok("6.1 Weighted BI-CGSTAB: Laplace  N=%4d   , d=%3d  , 2-err=%f" % (N,d,err2))
        nsucc += 1
    else:
        print_fail("6.1 Weighted BI-CGSTAB: Laplace  N=%4d   , d=%3d  , 2-err=%f" % (N,d,err2))
        nfail += 1

    print_summary("WTT BI-CGSTAB", nsucc, nfail)
    
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
