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

import numpy as np
import numpy.linalg as npla
import numpy.random as npr

from scipy import stats

import TensorToolbox as DT
import TensorToolbox.multilinalg as mla

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from UQToolbox import RandomSampling as RS

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

npr.seed(1)

##########################################################
# TEST 1
#  Try single high polynomial mode (worst case in SG)
#
# TEST 2
#  Best case SG
#
# TEST 3
#  Try high-rank on 1st-last parameter (worst case TT)
#  Use function with non-zero coeffs in n-dimensional simplexes
#  

TESTS = [3]
IS_PLOTTING = True
TITLE = True
fsize = (6,4.5)

if 1 in TESTS:
    ##########################################################
    # TEST 1
    import random

    maxvoleps = 1e-5
    delta = 1e-5

    xspan = [-1.,1.]
    d = 2
    size1D = 20
    maxord = 20
    
    P = S1D.Poly1D(S1D.JACOBI,[0.,0.])
    (x,w) = P.Quadrature(size1D)
    ords = [random.randrange(6,maxord) for i in range(d)]
    def f(X,params): return np.prod( [ P.GradEvaluate(np.array([X[i]]),params['ords'][i],0,norm=False) for i in range(len(X)) ] )

    X = [x for i in range(d)]
    W = [w for i in range(d)]
    params = { 'ords': ords }
    TW = DT.TensorWrapper( f, X, params )
    
    TTapprox = DT.TTvec(TW,method='ttcross')
    
    # Compute Fourier coefficients TT
    Vs = [P.GradVandermonde1D(X[i],maxord,0,norm=False)] * d
    TT_four = TTapprox.project(Vs,W)
    
    if IS_PLOTTING and d == 2:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(121, projection='3d')
        [XX,YY] = np.meshgrid(X[0],X[1])
        ax.plot_surface(XX,YY,TW.copy()[tuple( [slice(None,None,None) for i in range(d)] ) ],rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(XX,YY,TTapprox.to_tensor(),rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Plot 2D Fourier Coeff from TensorToolbox.product
        V2D = np.kron(Vs[0],Vs[1])
        WW = np.kron(W[0],W[1])
        fhat = np.dot(V2D.T,WW*TW.copy()[:,:].flatten()).reshape([size1D+1 for s in range(d)])

        # Plot 2D Fourier Coeff
        TT_fourier_abs = np.maximum(np.abs(TT_four.to_tensor()),1e-20*np.ones(TT_four.shape()))
        fhat_abs = np.maximum(np.abs(fhat),1e-20*np.ones(fhat.shape))
        VMAX = max( np.max(np.log10(TT_fourier_abs)), np.max(np.log10(fhat_abs)))
        VMIN = min( np.min(np.log10(TT_fourier_abs)), np.min(np.log10(fhat_abs)))
        fig = plt.figure(figsize=(15,12))
        plt.subplot(2,2,1)
        plt.imshow(np.log10(TT_fourier_abs), interpolation='none', vmin = VMIN, vmax = VMAX)
        plt.colorbar()
        plt.subplot(2,2,2)
        plt.imshow(np.log10(fhat_abs), interpolation='none', vmin = VMIN, vmax = VMAX)
        plt.colorbar()
        plt.subplot(2,2,3)
        plt.imshow( np.log10(np.abs(TT_fourier_abs - fhat_abs)), interpolation = 'none')
        plt.colorbar()


if 2 in TESTS:
    ##########################################################
    # TEST 2

    maxvoleps = 1e-5
    delta = 1e-5

    xspan = [-1.,1.]
    d = 2
    size1D = 20
    orders = [(15,0,0),(7,1,1),(3,2,3),(1,4,7),(0,8,15)]
    
    idxs = []
    for (oo,ii,jj) in orders:
        for i in range(ii,jj+1):
            for j in range(oo+1):
                idxs.append((i,j))

    idxArr = np.array(idxs)
    plt.figure()
    plt.plot(idxArr[:,0],idxArr[:,1],'.')
    plt.show(block=False)

    P = S1D.Poly1D(S1D.JACOBI,[0.,0.])
    (x,w) = P.Quadrature(size1D)
    def f(X,params):
        out = 0.
        for idx in params['idxs']:
            out += P.GradEvaluate(np.array([X[0]]),idx[0],0,norm=True) * P.GradEvaluate(np.array([X[1]]),idx[1],0,norm=True)
        return out

    X = [x for i in range(d)]
    W = [w for i in range(d)]
    params = { 'idxs': idxs }
    TW = DT.TensorWrapper( f, X, params )
    
    TTapprox = DT.TTvec(TW,method='ttcross')

    # Compute Fourier coefficients TT
    Vs = [P.GradVandermonde1D(X[i],size1D,0,norm=True)] * d
    TT_four = TTapprox.project(Vs,W)
    
    if IS_PLOTTING and d == 2:
        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(121, projection='3d')
        [XX,YY] = np.meshgrid(X[0],X[1])
        ax.plot_surface(XX,YY,TW.copy()[tuple( [slice(None,None,None) for i in range(d)] ) ],rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(XX,YY,TTapprox.to_tensor(),rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Plot 2D Fourier Coeff from TensorToolbox.product
        V2D = np.kron(Vs[0],Vs[1])
        WW = np.kron(W[0],W[1])
        fhat = np.dot(V2D.T,WW*TW.copy()[:,:].flatten()).reshape([size1D+1 for s in range(d)])

        # Plot 2D Fourier Coeff
        TT_fourier_abs = np.maximum(np.abs(TT_four.to_tensor()),1e-20*np.ones(TT_four.shape()))
        fhat_abs = np.maximum(np.abs(fhat),1e-20*np.ones(fhat.shape))
        VMAX = max( np.max(np.log10(TT_fourier_abs)), np.max(np.log10(fhat_abs)))
        VMIN = min( np.min(np.log10(TT_fourier_abs)), np.min(np.log10(fhat_abs)))
        
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(TT_fourier_abs), interpolation='none', vmin = VMIN, vmax = VMAX)
        plt.colorbar()
        plt.title('TT-Fourier')

        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(fhat_abs), interpolation='none', vmin = VMIN, vmax = VMAX)
        plt.colorbar()
        plt.title('Original-Fourier')

        fig = plt.figure(figsize=fsize)
        plt.imshow( np.log10(TT_fourier_abs - fhat_abs), interpolation = 'none')
        plt.colorbar()
        plt.title('TT-Fourier error')

if 3 in TESTS:
    ##########################################################
    # TEST 3

    import itertools
    
    COEFF = 4

    maxvoleps = 1e-10
    delta = 1e-10
    eps = 1e-13

    xspan = [-1.,1.]
    d = 4
    size1D = 20
    nzdim = [2,3]
    vol = (xspan[1]-xspan[0])**d
    
    # # Using simplex
    # maxord = 3
    # idxs = SND.MultiIndex(len(nzdim),maxord)

    # Using all idxs 
    ls_tmp = [ range(size1D+1) for i in range(len(nzdim)) ]
    idxs = list(itertools.product( *ls_tmp ))

    if COEFF == 0:
        coeffs = [1.] * len(idxs)
    elif COEFF == 1:
        coeffs = [ np.exp(-np.sum(idx)) for idx in idxs ]
    elif COEFF == 2:
        coeffs = [ np.exp(-np.prod( np.array(idx)+1. )) for idx in idxs ]
    elif COEFF == 3:
        coeffs = [ 1.1**-np.sum( np.array(idx) ** 3. ) for idx in idxs ]
    elif COEFF == 4:
        corr_mat = -0.9 * np.ones((len(nzdim),len(nzdim)))
        np.fill_diagonal(corr_mat,1.)
        coeffs = np.array([ np.exp(- np.dot( np.array(idx), np.dot(corr_mat,np.array(idx))) ) for idx in idxs ])

    names = ['Constant','Exp. f1','Exp f2','Exp f3','Exp f4']

    # ############################################
    # # Using Sparse-Grid admissible set
    # d = 2
    # size1D = 20
    # nzdim = [0,1]
    # orders = [(15,0,0),(7,1,1),(3,2,3),(1,4,7),(0,8,15)]
    # idxs = []
    # for (oo,ii,jj) in orders:
    #     for i in range(ii,jj+1):
    #         for j in range(oo+1):
    #             idxs.append((i,j))

    # End SG adm. set
    #############################################

    P = S1D.Poly1D(S1D.JACOBI,[0.,0.])
    (x,w) = P.Quadrature(size1D)
    VV = [ P.GradVandermonde1D(x,size1D,0,norm=True)] * d
    # Gammas = [ [P.Gamma(i) for i in range(size1D+1)] ] * d
    Gammas = [ np.diag(np.dot(VV[i],VV[i].T)) for i in range(d) ]

    def f(X,params):
        # if isinstance(X,np.ndarray):
        #     out = np.zeros(X.shape[0])
        # else:

        V = np.array([1.])
        for i in range(len(nzdim)):
            (idx,) = np.where(x == X[nzdim[i]])
            V = np.kron( V, params['VV'][i][idx,:] )

        normFact = np.sum( [np.dot( w, params['VV'][i][:,0]) for i in range(len(nzdim)) ] )
        out = np.dot( params['coeffs'], V.flatten() ) / normFact

        # out = 0.
        # for idxs,coeff in zip(params['idxs'],params['coeffs']):
        #     tmp = 1.
        #     for i,ii in enumerate(idxs):
        #         idx = np.where(x == X[nzdim[i]])
        #         tmp *= params['VV'][i][idx,ii] # * params['Gammas'][i][ii]
        #     # for i,ii in enumerate(idx): tmp *= P.GradEvaluate(np.array([ X[nzdim[i]] ]), ii, 0, norm=True)

        #     out += coeff * tmp / ( len(nzdim) * np.dot( w, VV[0][:,0]))

        return out 

    # Used to compute MC
    def fMC(X,params): 
        if isinstance(X,np.ndarray):
            out = np.zeros(X.shape[0])
        else:
            out = 0.
        for idxs,coeff in zip(params['idxs'],params['coeffs']):
            tmp = 1.
            for i,ii in enumerate(idxs):
                tmp *= params['VV'][nzdim[i]][:,ii] # * params['Gammas'][i][ii]
            # for i,ii in enumerate(idx): tmp *= P.GradEvaluate(np.array([ X[nzdim[i]] ]), ii, 0, norm=True)

            out += coeff * tmp / ( len(nzdim) * np.dot( w, VV[0][:,0]))
        return out

    X = [x for i in range(d)]
    W = [w for i in range(d)]
    params = { 'idxs': list(idxs),
               'coeffs': coeffs,
               'VV': VV,
               'Gammas': Gammas}
    TW = DT.TensorWrapper( f, X, params )
    
    TTapprox = DT.TTvec(TW,eps=eps,method='ttcross',delta=delta,mv_eps=maxvoleps)

    print "TTcross: grid size %d" % TW.get_size()
    print "TTcross: function evaluations %d" % TW.get_fill_level()

    # Compute Fourier coefficients TT
    Vs = [P.GradVandermonde1D(X[i],size1D,0,norm=True)] * d
    TT_four = TTapprox.project(Vs,W)

    if size1D**len(nzdim) < 1e6:
        four_tens = np.zeros( tuple( [size1D+1 for i in range(len(nzdim))] ) )
        ls_tmp = [ range(size1D+1) for i in range(len(nzdim)) ]
        idx_tmp = itertools.product( *ls_tmp )
        for ii  in idx_tmp:
            ii_tt = [0]*d
            for jj, tti in enumerate(nzdim): ii_tt[tti] = ii[jj]
            ii_tt = tuple(ii_tt)
            four_tens[ii] = TT_four[ii_tt]

    print "TTcross: ranks: %s" % str( TTapprox.ranks() )
    print "TTcross: Frobenius norm TT_four:  %e" % mla.norm(TT_four,'fro')
    print "TTcross: Frobenius norm sub_tens: %e" % npla.norm(four_tens.flatten())

    # Check approximation error using MC
    MCestVarLimit = 1e-1
    MCestMinIter = 100
    MCestMaxIter = 1e4
    MCstep = 10000
    var = 1.
    dist = stats.uniform()
    DIST = RS.MultiDimDistribution([dist] * d)
    intf = []
    values = []
    while (len(values) < MCestMinIter or var > mean**2. * MCestVarLimit) and len(values) < MCestMaxIter:
        # Monte Carlo
        xx = np.asarray( DIST.rvs(MCstep) )

        VsI = None
        VsI = [P.GradVandermonde1D(xx[:,i]*2.-1.,size1D,0,norm=True) for i in range(d)]
        TTval = TT_four.interpolate(VsI)

        TTvals = [ TTval[tuple([i]*d)] for i in range(MCstep) ]

        paramsMC = { 'idxs': list(idxs),
                     'coeffs': coeffs,
                     'VV': VsI,
                     'Gammas': Gammas}
        fval = fMC(xx,paramsMC)

        intf.extend( list(fval**2.) )
        values.extend( [(fval[i]-TTvals[i])**2. for i in range(MCstep)])
        mean = vol * np.mean(values)
        var = vol**2. * np.var(values) / len(values)

        sys.stdout.write("L2err estim. iter: %d Var: %e VarLim: %e \r" % (len(values) , var, mean**2. * MCestVarLimit))
        sys.stdout.flush()

    sys.stdout.write("\n")
    sys.stdout.flush()
    L2err = np.sqrt(mean/np.mean(intf))
    print "TTcross: L2 err TTapprox: %e" % L2err

    # Error of the TT Fourier coeff., with respect to their exact value
    idx_slice = []
    for i in range(d):
        if i in nzdim:
            idx_slice.append( slice(None,None,None) )
        else:
            idx_slice.append( 0 )
    idx_slice = tuple(idx_slice)
    four_approx = TT_four[ idx_slice ]
    print "TTcross: L2 norm error TT_four: %e" % npla.norm( four_approx.flatten() - np.asarray(coeffs) ,2)
    print "TTcross: Inf norm error TT_four: %e" % npla.norm( four_approx.flatten() - np.asarray(coeffs) , np.inf)

    if size1D**d < 1e4:
        # Construct full tensor
        full_tens = TW.copy()[ tuple([slice(None,None,None) for i in range(d)]) ]

        print "TTcross: Frobenius err.: %e" % (npla.norm( (TTapprox.to_tensor()-full_tens).flatten()) / npla.norm(full_tens.flatten()))
        
        TTapproxSVD = DT.TTvec(full_tens)

        # Compute Fourier coefficients TT
        Vs = [P.GradVandermonde1D(X[i],size1D,0,norm=True)] * d
        TT_fourSVD = TTapproxSVD.project(Vs,W)

        four_tensSVD = np.zeros( tuple( [size1D+1 for i in range(len(nzdim))] ) )
        import itertools
        ls_tmp = [ range(size1D+1) for i in range(len(nzdim)) ]
        idx_tmp = itertools.product( *ls_tmp )
        for ii  in idx_tmp:
            ii_tt = [0]*d
            for jj, tti in enumerate(nzdim): ii_tt[tti] = ii[jj]
            ii_tt = tuple(ii_tt)
            four_tensSVD[ii] = TT_fourSVD[ii_tt]
        
        print "TT-SVD: ranks: %s" % str( TTapproxSVD.ranks() )
        print "TT-SVD: Frobenius norm TT_four:  %e" % mla.norm(TT_fourSVD,'fro')
        print "TT-SVD: Frobenius norm sub_tens: %e" % npla.norm(four_tensSVD.flatten())

    VMAX = 0.
    VMIN = np.inf
    if size1D**d < 1e4 and IS_PLOTTING and len(nzdim) == 2:
        # Plot 2D Fourier Coeff
        TTsvd_fourier_abs = np.maximum(np.abs(four_tensSVD),1e-20*np.ones(four_tens.shape))
        VMAX = np.max(np.log10(TTsvd_fourier_abs))
        VMIN = np.min(np.log10(TTsvd_fourier_abs))

    if IS_PLOTTING and len(nzdim) == 2:
        # Plot 2D Fourier Coeff
        TTcross_fourier_abs = np.maximum(np.abs(four_tens),1e-20*np.ones(four_tens.shape))
        VMAX = max(VMAX, np.max(np.log10(TTcross_fourier_abs)))
        VMIN = min(VMIN, np.min(np.log10(TTcross_fourier_abs)))

    if IS_PLOTTING and d == 2:
        # Plot 2D Fourier Coeff from TensorToolbox.product
        V2D = np.kron(Vs[0],Vs[1])
        WW = np.kron(W[0],W[1])
        fhat = np.dot(V2D.T,WW*TW.copy()[:,:].flatten().astype('float64')).reshape([size1D+1 for s in range(d)])
        fhat_abs = np.maximum(np.abs(fhat),1e-20*np.ones(fhat.shape))
        VMAX = max(VMAX, np.max(np.log10(fhat_abs)))
        VMIN = min(VMIN, np.min(np.log10(fhat_abs)))

    if size1D**d < 1e4 and IS_PLOTTING and len(nzdim) == 2:
        # Plot 2D Fourier Coeff
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(TTsvd_fourier_abs), interpolation='none', vmin = VMIN, vmax = VMAX, origin='lower')
        plt.colorbar()
        if TITLE: plt.title('TT-SVD Fourier - %s' % names[COEFF])

    if IS_PLOTTING and len(nzdim) == 2:
        # Plot 2D Fourier Coeff
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(TTcross_fourier_abs), interpolation='none', vmin = VMIN, vmax = VMAX, origin='lower')
        plt.colorbar()
        if TITLE: plt.title('TT-cross Fourier - %s' % names[COEFF])

    if IS_PLOTTING and d == 2:
        # Plot 2D Fourier Coeff from TensorToolbox.product
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(fhat_abs), interpolation='none', vmin = VMIN, vmax = VMAX, origin='lower')
        plt.colorbar()
        if TITLE: plt.title('Full Fourier - %s' % names[COEFF])

    if size1D**d < 1e4 and IS_PLOTTING and len(nzdim) == 2:
        # TT-svd - TT-cross
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(np.abs(TTsvd_fourier_abs - TTcross_fourier_abs)), interpolation='none', origin='lower')
        plt.colorbar()
        if TITLE: plt.title('(TT-svd - TT-cross) Fourier - %s' % names[COEFF])

    if IS_PLOTTING and d == 2:    
        # Full - TT-cross
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(np.abs(fhat_abs - TTcross_fourier_abs)), interpolation='none', origin='lower')
        plt.colorbar()
        if TITLE: plt.title('(Full - TT-cross) Fourier - %s' % names[COEFF])
        
plt.show(block=False)
