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

    tw = DT.TensorWrapper(f,X,None,dtype=float)

    if F[5,10,15] == tw[5,10,15] and \
            np.all(F[1,2,:] == tw[1,2,:]) and \
            np.all(F[3:5,2:3,20:24] == tw[3:5,2:3,20:24]):
        print_ok("TTcross: Tensor Wrapper")
        nsucc += 1
    else:
        print_fail("TTcross: TensorWrapper")
        nfail += 1

    ####
    # Test Maxvol
    ####
    maxvoleps = 1e-2
    pass_maxvol = True
    N = 100

    i = 0
    while pass_maxvol == True and i < N:
        i += 1
        A = npr.random(600).reshape((100,6))
        (I,AsqInv,it) = DT.maxvol(A,delta=maxvoleps)
        if np.max(np.abs(np.dot(A,AsqInv))) > 1. + maxvoleps:
            pass_maxvol = False

    if pass_maxvol == True:
        print_ok('TTcross: Maxvol')
        nsucc += 1
    else:
        print_fail('TTcross: Maxvol at it=%d' % i)
        nsucc += 1


    ####
    # Test Low Rank Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5
    pass_lowrankapprox = True
    N = 10

    i = 0
    logging.info( "(rows,cols,rank) FroA, FroErr, FroErr/FroA, maxAAinv, maxAinvA" )
    while pass_lowrankapprox == True and i < N:
        i += 1
        size = npr.random_integers(10,100,2)
        r = npr.random_integers(max(1,np.min(size)-10),np.min(size))
        A = npr.random(np.prod(size)).reshape(size)
        (I,J,AsqInv,it) = DT.lowrankapprox(A,r,delta=delta,maxvoleps=maxvoleps)

        AAinv = np.max(np.abs( np.dot(A[:,J],AsqInv) ) )
        AinvA = np.max(np.abs( np.dot(AsqInv, A[I,:])  ) )
        FroErr = npla.norm( np.dot(A[:,J],np.dot(AsqInv, A[I,:])) - A , 'fro')
        FroA = npla.norm(A,'fro')
        logging.info( "(%d,%d,%d) %f, %f, %f %f %f" % (size[0],size[1],r,FroA, FroErr, FroErr/FroA, AAinv, AinvA) )
        if AAinv > 1. + maxvoleps:
            pass_maxvol = False

    if pass_maxvol == True:
        print_ok('TTcross: Random Low Rank Approx')
        nsucc += 1
    else:
        print_fail('TTcross: Random Low Rank Approx at it=%d' % i)        
        nsucc += 1


    ####
    # Sin*Cos Low Rank Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5

    size = (100,100)
    r = 1

    # Build up the 2d tensor wrapper
    def f(X,params): return np.sin(X[0])*np.cos(X[1])
    X = [np.linspace(0,2*np.pi,size[0]), np.linspace(0,2*np.pi,size[1])]
    TW = DT.TensorWrapper(f,X,None,dtype=float)

    # Compute low rank approx
    (I,J,AsqInv,it) = DT.lowrankapprox(TW,r,delta=delta,maxvoleps=maxvoleps)
    fill = TW.get_fill_level()

    Fapprox = np.dot(TW[:,J].reshape((TW.shape[0],len(J))),np.dot(AsqInv, TW[I,:].reshape((len(I),TW.shape[1])) ) )
    FroErr = npla.norm(Fapprox-TW[:,:], 'fro')
    if FroErr < 1e-12:
        print_ok('TTcross: sin(x)*cos(y) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1
    else:
        print_fail('TTcross: sin(x)*cos(y) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1

    if PLOTTING:
        plt.figure(figsize=(12,7))
        plt.subplot(1,2,1)
        plt.imshow(TW[:,:])
        plt.subplot(1,2,2)
        plt.imshow(Fapprox)

    ####
    # Sin(x+y) Low Rank Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5

    size = (100,100)
    r = 2

    # Build up the 2d tensor wrapper
    def f(X,params): return np.sin(X[0])*np.cos(X[1])
    def f(X,params): return np.sin(X[0]+X[1])
    X = [np.linspace(0,2*np.pi,size[0]), np.linspace(0,2*np.pi,size[1])]
    TW = DT.TensorWrapper(f,X,None,dtype=float)

    # Compute low rank approx
    (I,J,AsqInv,it) = DT.lowrankapprox(TW,r,delta=delta,maxvoleps=maxvoleps)
    fill = TW.get_fill_level()

    Fapprox = np.dot(TW[:,J],np.dot(AsqInv, TW[I,:]))
    FroErr = npla.norm(Fapprox-TW[:,:], 'fro')
    if FroErr < 1e-12:
        print_ok('TTcross: sin(x+y) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1
    else:
        print_fail('TTcross: sin(x+y) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1

    if PLOTTING:
        plt.figure(figsize=(12,7))
        plt.subplot(1,2,1)
        plt.imshow(TW[:,:])
        plt.subplot(1,2,2)
        plt.imshow(Fapprox)

    ####
    # Sin(x)*cos(y)*Sin(z) TTcross Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5
    eps = 1e-10

    size = (10,10,10)

    # Build up the 2d tensor wrapper
    def f(X,params): return np.sin(X[0])*np.cos(X[1])*np.sin(X[2])
    X = [np.linspace(0,2*np.pi,size[0]), np.linspace(0,2*np.pi,size[1]), np.linspace(0,2*np.pi,size[2])]
    TW = DT.TensorWrapper(f,X,dtype=float)

    # Compute low rank approx
    TTapprox = DT.TTvec(TW)
    TTapprox.build( method='ttcross',eps=eps,mv_eps=maxvoleps,delta=delta)
    fill = TW.get_fill_level()
    crossRanks = TTapprox.ranks()
    PassedRanks = all( map( operator.gt, crossRanks[1:-1], TTapprox.rounding(eps=delta).ranks()[1:-1] ) )

    FroErr = mla.norm( TTapprox.to_tensor() - TW.copy()[tuple( [slice(None,None,None) for i in range(len(size))] ) ], 'fro')
    if FroErr < eps:
        print_ok('TTcross: sin(x)*cos(y)*sin(z) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))
        nsucc += 1
    else:
        print_fail('TTcross: sin(x)*cos(y)*sin(z) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size())))        
        nsucc += 1

    if PLOTTING:
        # Get filled idxs
        fill_idxs = np.array(TW.get_fill_idxs())
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(fill_idxs[:,0],fill_idxs[:,1],fill_idxs[:,2])
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        # Get last used idxs
        Is = TTapprox.Is
        Js = TTapprox.Js
        ndim = len(X)
        dims = [len(Xi) for Xi in X]
        idxs = []
        for k in range(len(Is)-1,-1,-1):
            for i in range(len(Is[k])):
                for j in range(len(Js[k])):
                    for kk in range(dims[k]):
                        idxs.append( Is[k][i] + (kk,) + Js[k][j] )

        last_idxs = np.array(idxs)
        fig = plt.figure()
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(last_idxs[:,0],last_idxs[:,1],last_idxs[:,2],c='r')

        plt.show(block=False)

    print_summary("TTcross", nsucc, nfail)
    
    return (nsucc,nfail)

if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestTTcross.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs,PLOTTING=True,loglev=logging.INFO)
