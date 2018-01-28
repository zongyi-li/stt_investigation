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
import operator
import time

import numpy as np
import numpy.linalg as npla
import numpy.random as npr
import itertools

from scipy import stats

import TensorToolbox as DT
import TensorToolbox.multilinalg as mla

from TensorToolbox.unittests.auxiliary import bcolors, print_ok, print_fail, print_summary

def run(maxprocs,PLOTTING=False, loglev=logging.WARNING):

    logging.basicConfig(level=loglev)

    if PLOTTING:
        import matplotlib.pyplot as plt
        from matplotlib import cm
        from mpl_toolkits.mplot3d import Axes3D

    nsucc = 0
    nfail = 0

    ####
    # Test Tensor Wrapper
    ####
    def f(x,params=None): 
        if x.ndim == 1:
            return np.sum(x)
        if x.ndim == 2:
            return np.sum(x,axis=1)

    dims = [11,21,31]
    X = [np.linspace(1.,10.,dims[0]), np.linspace(1,20.,dims[1]), np.linspace(1,30.,dims[2])]
    XX = np.array(list(itertools.product(*X)))
    F = f( XX ).reshape(dims)

    tw = DT.TensorWrapper(f,X,None)

    if F[5,10,15] == tw[5,10,15] and \
            np.all(F[1,2,:] == tw[1,2,:]) and \
            np.all(F[3:5,2:3,20:24] == tw[3:5,2:3,20:24]):
        print_ok("TTdmrgcross: Tensor Wrapper")
        nsucc += 1
    else:
        print_fail("TTdmrgcross: TensorWrapper")
        nsucc += 1

    ####
    # 1./(x+y+1) Low Rank Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5
    eps = 1e-10

    d = 2
    size = [100] * d

    # Build up the 2d tensor wrapper
    def f(X,params): return 1./(X[0]+X[1]+1.)
    X = [np.linspace(0,2*np.pi,size[0]), np.linspace(0,2*np.pi,size[1])]
    TW = DT.TensorWrapper(f,X,None)

    # Compute low rank approx
    TTapprox = DT.TTvec(TW)
    TTapprox.build(method='ttdmrgcross',eps=eps,mv_eps=maxvoleps)
    fill = TW.get_fill_level()
    crossRanks = TTapprox.ranks()
    PassedRanks = all( map( operator.eq, crossRanks[1:-1], TTapprox.ranks()[1:-1] ) )

    A = TW.copy()[tuple( [slice(None,None,None) for i in range(len(size))] ) ]
    FroErr = mla.norm( TTapprox.to_tensor() - A, 'fro')
    kappa = np.max(A)/np.min(A) # This is slightly off with respect to the analysis
    r = np.max(TTapprox.ranks())
    epsTarget = (2.*r + kappa * r + 1.)**(np.log2(d)) * (r+1.) * eps
    if FroErr < epsTarget:
        print_ok('TTdmrgcross: 1./(x+y+1) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1
    else:
        print_fail('TTdmrgcross: 1./(x+y+1) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1

    if PLOTTING:
        # Get filled idxs
        fill_idxs = np.array(TW.get_fill_idxs())
        plt.figure()
        plt.plot(fill_idxs[:,0],fill_idxs[:,1],'o')

        plt.show(block=False)

    ####
    # sin(sum(x)) TTcross Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5
    eps = 1e-10
    
    d = 3
    size = [20] * d

    # Build up the tensor wrapper
    def f(X,params): return np.sin( np.sum(X) )
    X = [np.linspace(0,2*np.pi,size[0])] * d
    TW = DT.TensorWrapper(f,X)

    TTapprox = DT.TTvec(TW)
    TTapprox.build(method='ttdmrgcross',eps=eps,mv_eps=maxvoleps)
    fill = TW.get_fill_level()
    crossRanks = TTapprox.ranks()
    PassedRanks = all( map( operator.eq, crossRanks[1:-1], TTapprox.ranks()[1:-1] ) )

    A = TW.copy()[tuple( [slice(None,None,None) for i in range(len(size))] ) ]
    FroErr = mla.norm( TTapprox.to_tensor() - TW.copy()[tuple( [slice(None,None,None) for i in range(len(size))] ) ], 'fro')
    kappa = np.max(A)/np.min(A) # This is slightly off with respect to the analysis
    r = np.max(TTapprox.ranks())
    epsTarget = (2.*r + kappa * r + 1.)**(np.log2(d)) * (r+1.) * eps
    if FroErr < epsTarget:
        print_ok('TTdmrgcross: sin(sum(x)) - d=3 - Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1
    else:
        print_fail('TTdmrgcross: sin(sum(x)) - d=3 - Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1

    if PLOTTING and d == 3:
        # Get filled idxs
        fill_idxs = np.array(TW.get_fill_idxs())
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(fill_idxs[:,0],fill_idxs[:,1],fill_idxs[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Get last used idxs
        last_idxs = TTapprox.get_ttdmrg_eval_idxs()
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(last_idxs[:,0],last_idxs[:,1],last_idxs[:,2],c='r')

        plt.show(block=False)
    
    
    ####
    # 1/(sum(x)+1) TTdmrgcross Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5
    eps = 1e-10

    d = 5
    size = [10] * d

    # Build up the 2d tensor wrapper
    def f(X,params): return 1./(np.sum(X) + 1.)
    X = [np.linspace(0,1,size[i]) for i in range(len(size))]
    TW = DT.TensorWrapper(f,X)

    # Compute low rank approx
    TTapprox = DT.TTvec(TW)
    TTapprox.build(method='ttdmrgcross',eps=eps,mv_eps=maxvoleps)
    fill = TW.get_fill_level()
    crossRanks = TTapprox.ranks()
    PassedRanks = all( map( operator.eq, crossRanks[1:-1], TTapprox.ranks()[1:-1] ) )

    A = TW.copy()[tuple( [slice(None,None,None) for i in range(len(size))] ) ]
    FroErr = mla.norm( TTapprox.to_tensor() - A, 'fro')
    MaxErr = np.max( TTapprox.to_tensor() - A )
    kappa = np.max(A)/np.min(A) # This is slightly off with respect to the analysis
    r = np.max(TTapprox.ranks())
    epsTarget = (2.*r + kappa * r + 1.)**(np.log2(d)) * (r+1.) * eps
    if FroErr < epsTarget:
        print_ok('TTdmrgcross: 1/(sum(x)+1), d=%d, Low Rank Approx (FroErr=%e, MaxErr=%e, Fill=%.2f%%)' % (d,FroErr,MaxErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1
    else:
        print_fail('TTdmrgcross: 1/(sum(x)+1), d=%d, Low Rank Approx (FroErr=%e, FroErr=%e, Fill=%.2f%%)' % (d,FroErr,MaxErr,100.*np.float(fill)/np.float(TW.get_size())))
        nfail += 1

    print_summary("TTdmrgcross", nsucc, nfail)
    
    return (nsucc,nfail)


if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestTTdmrgcross.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs,PLOTTING=True,loglev=logging.INFO)
