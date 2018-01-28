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

    ###################################################################
    # Test timings and comp. rate for compression of Laplace-like op.
    ###################################################################
    eps = 0.001
    Ns = 2**np.arange(4,7,1,dtype=int)
    ds = 2**np.arange(4,6,dtype=int)
    timing = np.zeros((len(Ns),len(ds)))
    comp_rate = np.zeros((len(Ns),len(ds)))
    for i_N, N in enumerate(Ns):
        D = np.diag(-np.ones((N-1)),-1) + np.diag(-np.ones((N-1)),1) + np.diag(2*np.ones((N)),0)
        I = np.eye(N)
        D_flat = D.flatten()
        I_flat = I.flatten()
        for i_d, d in enumerate(ds):
            sys.stdout.write('N=%d   , d=%d      [STARTED]\r' % (N,d))
            sys.stdout.flush()
            # Canonical form of n-dimentional Laplace operator
            CPtmp = [] # U[i][alpha,k] = U_i(alpha,k)
            for i in range(d):
                CPi = np.empty((d,N**2))
                for alpha in range(d):
                    if i != alpha:
                        CPi[alpha,:] = I_flat
                    else:
                        CPi[alpha,:] = D_flat
                CPtmp.append(CPi)
            CP = DT.Candecomp(CPtmp)

            # Canonical to TT
            sys.stdout.write("\033[K")
            sys.stdout.write('N=%4d   , d=%3d      [CP->TT]\r' % (N,d))
            sys.stdout.flush()
            TT = DT.TTmat(CP,nrows=N,ncols=N)
            TT.build()
            TT_pre = TT.copy()
            pre_norm = mla.norm(TT_pre,'fro')

            # Rounding TT
            sys.stdout.write("\033[K")
            sys.stdout.write('N=%4d   , d=%3d      [TT-round]\r' % (N,d))
            sys.stdout.flush()
            st = time.clock()
            TT.rounding(eps)
            end = time.clock()

            if np.max(TT.ranks()) != 2:
                print_fail("\033[K" + "1.1 Compression Timing N=%4d   , d=%3d      [RANK ERROR]   Time: %f" % (N,d,end-st))
                nfail += 1
            elif mla.norm(TT_pre - TT,'fro') > eps * pre_norm:
                print_fail("\033[K" + "1.1 Compression Timing N=%4d   , d=%3d      [NORM ERROR]   Time: %f" % (N,d,end-st))
                nfail += 1
            else:
                print_ok("\033[K" + "1.1 Compression Timing N=%4d   , d=%3d      [ENDED]   Time: %f" % (N,d,end-st))
                nsucc += 1

            comp_rate[i_N,i_d] = float(TT.size())/N**(2.*d)
            timing[i_N,i_d] = end-st

    # Compute scalings with respect to N and d
    if PLOTTING:
        d_sc = np.polyfit(np.log2(ds),np.log2(timing[-1,:]),1)[0]
        N_sc = np.polyfit(np.log2(Ns),np.log2(timing[:,-1]),1)[0]
        sys.stdout.write("Scaling: N^%f, d^%f\n" % (N_sc,d_sc))
        sys.stdout.flush()

        plt.figure(figsize=(14,7))
        plt.subplot(1,2,1)
        plt.loglog(Ns,comp_rate[:,-1],'o-',basex=2, basey=2)
        plt.grid()
        plt.xlabel('N')
        plt.ylabel('Comp. Rate TT/FULL')
        plt.subplot(1,2,2)
        plt.loglog(Ns,timing[:,-1],'o-',basex=2, basey=2)
        plt.grid()
        plt.xlabel('N')
        plt.ylabel('Round Time (s)')
        plt.show(block=False)

        plt.figure(figsize=(14,7))
        plt.subplot(1,2,1)
        plt.loglog(ds,comp_rate[-1,:],'o-',basex=2, basey=2)
        plt.grid()
        plt.xlabel('d')
        plt.ylabel('Comp. Rate TT/FULL')
        plt.subplot(1,2,2)
        plt.loglog(ds,timing[-1,:],'o-',basex=2, basey=2)
        plt.grid()
        plt.xlabel('d')
        plt.ylabel('Round Time (s)')
        plt.show(block=False)

        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        (NN,dd) = np.meshgrid(np.log2(Ns),np.log2(ds))
        T = timing.copy().T
        T[T==0.] = np.min(T[np.nonzero(T)])
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.plot_surface(NN,dd,np.log2(T),rstride=1, cstride=1, cmap=cm.coolwarm,
                linewidth=0, antialiased=False)
        plt.show(block=False)


    print_summary("TT Compression", nsucc, nfail)

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
