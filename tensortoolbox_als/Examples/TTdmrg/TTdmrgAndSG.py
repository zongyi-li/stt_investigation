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

import TensorToolbox as TT
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

TESTS = [0]
IS_PLOTTING = True
TITLE = False
fsize = (6,4.5)

if 0 in TESTS:
    ####
    # exp(- |X-X0|^2/2*l^2) Low Rank Approximation
    ####
    maxvoleps = 1e-5
    delta = 1e-5
    eps = 1e-10

    d = 2
    size = tuple([32]*d) # 1024 points

    # Build up the 2d tensor wrapper
    X0 = np.array([0.2]*d)
    l = 0.05
    params = {'X0': X0, 'l': l}
    def f(X,params): return np.exp( - np.sum( (X-params['X0'])**2. ) / (2*params['l']**2.) )
    X = [np.linspace(0,1.,size[0])]*d
    TW = TT.TensorWrapper(f,X,params)

    # Compute low rank approx
    TTapprox = TT.QTTvec(TW)
    TTapprox.build(method='ttdmrg',eps=eps,mv_eps=maxvoleps,kickrank=0)
    fill = TW.get_fill_level()
    crossRanks = TTapprox.ranks()
    PassedRanks = all( map( operator.eq, crossRanks[1:-1], TTapprox.ranks()[1:-1] ) )

    A = TW.copy()[tuple( [slice(None,None,None) for i in range(len(size))] ) ]
    FroErr = mla.norm( TTapprox.to_tensor() - A, 'fro')
    kappa = np.max(A)/np.min(A) # This is slightly off with respect to the analysis
    r = np.max(TTapprox.ranks())
    epsTarget = (2.*r + kappa * r + 1.)**(np.log2(d)) * (r+1.) * eps
    if FroErr < epsTarget:
        print '[SUCCESS] QTTdmrg: 1./(x+y+1) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size()))
    else:
        print '[FAIL] QTTdmrg: 1./(x+y+1) Low Rank Approx (FroErr=%e, Fill=%.2f%%)' % (FroErr,100.*np.float(fill)/np.float(TW.get_size()))

    if IS_PLOTTING and d==2:
        # Get filled idxs
        fill_idxs = TW.get_fill_idxs()
        last_idxs = TTapprox.get_ttdmrg_eval_idxs()
        last_idxs = [ tuple(last_idxs[i,:]) for i in range(last_idxs.shape[0]) ]

        fill_idxs = set(fill_idxs) - set(last_idxs)

        fill_idxs = np.asarray(list(fill_idxs))
        last_idxs = np.asarray(list(last_idxs))

        XX,YY = np.meshgrid(X[0],X[1])
        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        # ax.plot_wireframe( XX, YY, A.astype(float) )
        ax.plot_surface(XX, YY, A.astype(float), rstride=1, cstride=1, alpha=0.3)
        # ax.contour( XX, YY, A.astype(float), zdir='z', offset=0. )

        # ax.scatter(XX.flatten(),YY.flatten(), -2. * np.ones(XX.size),c='w',alpha=1)
        
        ax.scatter(X[0][fill_idxs[:,0]],X[1][fill_idxs[:,1]],-np.ones(fill_idxs.shape[0]),c='w',alpha=1)
        ax.scatter(X[0][last_idxs[:,0]],X[1][last_idxs[:,1]],-np.ones(last_idxs.shape[0]),c='k',alpha=1)
        
        ax.set_xlim(0.,1.)
        ax.set_ylim(0.,1.)
        
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('f(x,y)')

        # plt.figure(figsize=(6,5))
        # plt.imshow(A.astype(float),origin='lower',extent=(0.,1.,0.,1.))
        # plt.plot(X[0][fill_idxs[:,0]],X[1][fill_idxs[:,1]],'wo')
        # plt.plot(X[0][last_idxs[:,0]],X[1][last_idxs[:,1]],'ko')
        # plt.xlabel('x')
        # plt.ylabel('y')
        plt.show(block=False)
    if IS_PLOTTING and d==3:
        fill_idxs = TW.get_fill_idxs()
        last_idxs = TTapprox.get_ttdmrg_eval_idxs()
        last_idxs = [ tuple(last_idxs[i,:]) for i in range(last_idxs.shape[0]) ]

        fill_idxs = set(fill_idxs) - set(last_idxs)

        fill_idxs = np.asarray(list(fill_idxs))
        last_idxs = np.asarray(list(last_idxs))

        fig = plt.figure(figsize=(6,5))
        ax = fig.add_subplot(111, projection='3d')
        ax.scatter(X[0][fill_idxs[:,0]],X[1][fill_idxs[:,1]],X[2][fill_idxs[:,2]],c='w',alpha=1)
        ax.set_xlabel('x')
        ax.set_ylabel('y')
        ax.set_zlabel('z')

        ax.scatter(X[0][last_idxs[:,0]],X[1][last_idxs[:,1]],X[2][last_idxs[:,2]],c='k',alpha=1)

        plt.show(block=False)

if 1 in TESTS:
    ##########################################################
    # TEST 1
    import random

    maxvoleps = 1e-5
    delta = 1e-5

    xspan = [-1.,1.]
    d = 2
    size1D = 31
    maxord = 31
    
    P = S1D.Poly1D(S1D.JACOBI,[0.,0.])
    (x,w) = P.Quadrature(size1D)
    ords = [random.randrange(6,maxord) for i in range(d)]
    def f(X,params): return np.prod( [ P.GradEvaluate(np.array([X[i]]),params['ords'][i],0,norm=False) for i in range(len(X)) ] )

    X = [ (S1D.JACOBI,S1D.GAUSS,(0.,0.),[-1.,1.]) for i in xrange(d)]
    orders = [size1D] * d
    params = { 'ords': ords }
    surr_type = TT.PROJECTION

    STTapprox = TT.SQTT( f, X, params, range_dim=0, method='ttdmrg', eps=1e-10, orders=orders, surrogateONOFF=True, surrogate_type=surr_type, kickrank=1 )
    STTapprox.build()
    print "Fill level: %d/%d" % (STTapprox.TW.get_fill_level(), STTapprox.TW.get_size())
    
    # Compute Fourier coefficients TT
    TT_four = STTapprox.TTfour[0]
    
    if IS_PLOTTING and d == 2:
        X = [x] * d
        W = [w] * d
        Vs = [P.GradVandermonde1D(X[i],size1D,0,norm=True)] * d

        fig = plt.figure(figsize=(12,7))
        ax = fig.add_subplot(121, projection='3d')
        [XX,YY] = np.meshgrid(X[0],X[1])
        ax.plot_surface(XX,YY,STTapprox.TW.copy()[tuple( [slice(None,None,None) for i in range(d)] ) ],rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        ax = fig.add_subplot(122, projection='3d')
        ax.plot_surface(XX,YY,STTapprox.TTapprox[0].to_tensor(),rstride=1, cstride=1, cmap=cm.coolwarm, linewidth=0, antialiased=False)
        
        # Plot 2D Fourier Coeff from TensorToolbox.product
        V2D = np.kron(Vs[0],Vs[1])
        WW = np.kron(W[0],W[1])
        fhat = np.dot(V2D.T,WW*STTapprox.TW.copy()[:,:].flatten()).reshape([size1D+1 for s in range(d)])

        # Plot 2D Fourier Coeff
        TT_fourier_abs = np.maximum(np.abs(TT_four.to_tensor()),1e-20*np.ones(TT_four.shape()))
        fhat_abs = np.maximum(np.abs(fhat),1e-20*np.ones(fhat.shape)).astype(float)
        VMAX = max( np.max(np.log10(TT_fourier_abs)), np.max(np.log10(fhat_abs)))
        VMIN = min( np.min(np.log10(TT_fourier_abs)), np.min(np.log10(fhat_abs)))
        fig = plt.figure()
        plt.imshow(np.log10(TT_fourier_abs), interpolation='none', origin='lower', vmin = VMIN, vmax = VMAX, cmap=cm.gray_r)
        plt.colorbar()

        plt.figure()
        plt.imshow(np.log10(fhat_abs), interpolation='none', origin='lower', vmin = VMIN, vmax = VMAX)
        plt.colorbar()

        plt.figure()
        plt.imshow( np.log10(np.abs(TT_fourier_abs - fhat_abs)), interpolation = 'none', origin='lower')
        plt.colorbar()

        # Get filled idxs
        fill_idxs = np.array(STTapprox.TW.get_fill_idxs())
        last_idxs = STTapprox.generic_approx[0].get_ttdmrg_eval_idxs()
        plt.figure()
        plt.plot(fill_idxs[:,0],fill_idxs[:,1],'wo')
        plt.plot(last_idxs[:,0],last_idxs[:,1],'ro')


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
    TW = TT.TensorWrapper( f, X, params )
    
    TTapprox = TT.TTvec(TW,method='ttdmrg')

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
    d = 2
    size1D = 15
    nzdim = [0,1]
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
            (idx,) = np.where(np.abs(x-X[nzdim[i]]) <= 10.*np.spacing(1) )
            V = np.kron( V, params['VV'][i][idx,:] )

        # normFact = np.sum( [np.dot( w, params['VV'][i][:,0]) for i in range(len(nzdim)) ] )
        out = np.dot( params['coeffs'], V.flatten() ) # / normFact

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

            out += coeff * tmp # / ( len(nzdim) * np.dot( w, VV[0][:,0]))
        return out

    params = { 'idxs': list(idxs),
               'coeffs': coeffs,
               'VV': VV,
               'Gammas': Gammas}
    orders = [size1D] * d
    surr_type = TT.PROJECTION
    X = []
    for i in xrange(d):
        X.append( (S1D.JACOBI, S1D.GAUSS, (0.,0.), [-1.,1.]) )
    
    STTapprox = TT.SQTT( f, X, params, range_dim=0, method='ttdmrg', eps=1e-10, orders=orders, surrogateONOFF=True, surrogate_type=surr_type, kickrank=5)
    STTapprox.build()

    print "TTdmrg: grid size %d" % STTapprox.TW.get_size()
    print "TTdmrg: function evaluations %d" % STTapprox.TW.get_fill_level()

    if size1D**len(nzdim) < 1e6:
        four_tens = np.zeros( tuple( [size1D+1 for i in range(len(nzdim))] ) )
        ls_tmp = [ range(size1D+1) for i in range(len(nzdim)) ]
        idx_tmp = itertools.product( *ls_tmp )
        for ii  in idx_tmp:
            ii_tt = [0]*d
            for jj, tti in enumerate(nzdim): ii_tt[tti] = ii[jj]
            ii_tt = tuple(ii_tt)
            four_tens[ii] = STTapprox.TTfour[0][ii_tt]

    print "TTdmrg: ranks: %s" % str( STTapprox.TTapprox[0].ranks() )
    print "TTdmrg: Frobenius norm TT_four:  %e" % mla.norm(STTapprox.TTfour[0],'fro')
    print "TTdmrg: Frobenius norm sub_tens: %e" % npla.norm(four_tens.flatten())

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
        xx = (2. * np.asarray( DIST.rvs(MCstep) )) - 1.

        TTvals = STTapprox(xx)

        VsI = None
        VsI = [P.GradVandermonde1D(xx[:,i],size1D,0,norm=True) for i in range(d)]
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
    print "TTdmrg: L2 err TTapprox: %e" % L2err

    # Error of the TT Fourier coeff., with respect to their exact value
    idx_slice = []
    for i in range(d):
        if i in nzdim:
            idx_slice.append( slice(None,None,None) )
        else:
            idx_slice.append( 0 )
    idx_slice = tuple(idx_slice)
    four_approx = STTapprox.TTfour[0][ idx_slice ]
    print "TTdmrg: L2 norm error TT_four: %e" % npla.norm( four_approx.flatten() - np.asarray(coeffs) ,2)
    print "TTdmrg: Inf norm error TT_four: %e" % npla.norm( four_approx.flatten() - np.asarray(coeffs) , np.inf)

    if size1D**d < 1e4:
        # Construct full tensor
        full_tens = STTapprox.TW.copy()[ tuple([slice(None,None,None) for i in range(d)]) ]

        print "TTdmrg: Frobenius err.: %e" % (npla.norm( (STTapprox.TTapprox[0].to_tensor()-full_tens).flatten()) / npla.norm(full_tens.flatten()))
        
        TTapproxSVD = TT.TTvec(full_tens)
        TTapproxSVD.build()

        # Compute Fourier coefficients TT
        Xs = [x] * d
        Ws = [w] * d
        Vs = [P.GradVandermonde1D(Xs[i],size1D,0,norm=True) for i in range(d)]
        TT_fourSVD = TTapproxSVD.project(Vs,Ws)

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
        TTdmrg_fourier_abs = np.maximum(np.abs(four_tens),1e-20*np.ones(four_tens.shape))
        VMAX = max(VMAX, np.max(np.log10(TTdmrg_fourier_abs)))
        VMIN = min(VMIN, np.min(np.log10(TTdmrg_fourier_abs)))

    if IS_PLOTTING and d == 2:
        # Plot 2D Fourier Coeff from TensorToolbox.product
        V2D = np.kron(Vs[0],Vs[1])
        WW = np.kron(Ws[0],Ws[1])
        fhat = np.dot(V2D.T,WW*STTapprox.TW.copy()[:,:].flatten().astype('float64')).reshape([size1D+1 for s in range(d)])
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
        plt.imshow(np.log10(TTdmrg_fourier_abs), interpolation='none', vmin = VMIN, vmax = VMAX, origin='lower',cmap=cm.gray_r)
        plt.colorbar()
        if TITLE: plt.title('TT-dmrg Fourier - %s' % names[COEFF])

    if IS_PLOTTING and d == 2:
        # Plot 2D Fourier Coeff from TensorToolbox.product
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(fhat_abs), interpolation='none', vmin = VMIN, vmax = VMAX, origin='lower')
        plt.colorbar()
        if TITLE: plt.title('Full Fourier - %s' % names[COEFF])

    if size1D**d < 1e4 and IS_PLOTTING and len(nzdim) == 2:
        # TT-svd - TT-dmrg
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(np.abs(TTsvd_fourier_abs - TTdmrg_fourier_abs)), interpolation='none', origin='lower')
        plt.colorbar()
        if TITLE: plt.title('(TT-svd - TT-dmrg) Fourier - %s' % names[COEFF])

    if IS_PLOTTING and d == 2:    
        # Full - TT-dmrg
        fig = plt.figure(figsize=fsize)
        plt.imshow(np.log10(np.abs(fhat_abs - TTdmrg_fourier_abs)), interpolation='none', origin='lower')
        plt.colorbar()
        if TITLE: plt.title('(Full - TT-dmrg) Fourier - %s' % names[COEFF])
        
plt.show(block=False)
