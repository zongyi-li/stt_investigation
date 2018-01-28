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
import numpy as np
import numpy.linalg as npla
import scipy.sparse as sp
import scipy.sparse.linalg as spla
import itertools
import time

import TensorToolbox as DT
import TensorToolbox.multilinalg as mla
from TensorToolbox.unittests.auxiliary import print_ok, print_fail, print_summary

def run(maxprocs, PLOTTING=False, loglev=logging.WARNING):

    import numpy.linalg as npla

    logging.basicConfig(level=loglev)

    if PLOTTING:
        from matplotlib import pyplot as plt

    nsucc = 0
    nfail = 0

    ################ TEST 1 ###########################
    # Test folding/unfolding index function
    sys.stdout.write("Test folding/unfolding index function\r")
    sys.stdout.flush()

    dlist = (4,2,8)
    base = 2
    dfold = [base for i in range(int(np.log(np.prod(dlist))/np.log(base)))]
    A = np.arange(64).reshape(dlist)
    Aflat = A.flatten()
    Arsh = A.reshape(dfold)
    test = True
    err = []
    for i in range(dlist[0]):
        for j in range(dlist[1]):
            for k in range(dlist[2]):
                idxs = (i,j,k)
                val = (A[idxs] == Aflat[DT.idxunfold(dlist,idxs)] and A[idxs] == Arsh[DT.idxfold(dfold,DT.idxunfold(dlist,idxs))])
                if not val:
                    err.append(idxs)
                    test = False

    if test:
        print_ok("Test folding/unfolding index function")
        nsucc += 1
    else:
        print_fail("Test folding/unfolding index function")
        nfail += 1

    ################ TEST 2.1 ###########################
    # Test exponential N-dimensional vector (2^6 points)
    sys.stdout.write("Test exponential N-dimensional vector (2^6 points)\r")
    sys.stdout.flush()

    z = 2.
    q = 2
    L = 6
    N = q**L
    X = z**np.arange(N)

    TT = DT.QTTvec(X)
    TT.build()

    if TT.ranks() == [1 for i in range(L+1)]:
        print_ok("Test exponential N-dimensional vector (2^6 points)")
        nsucc += 1
    else:
        print_fail("Test exponential N-dimensional vector (2^6 points)")
        nfail += 1

    ################ TEST 2.1b ###########################
    # Test exponential N-dimensional vector (28 points)
    sys.stdout.write("Test exponential N-dimensional vector (28 points)\r")
    sys.stdout.flush()

    z = 2.
    N = 28
    X = z**np.arange(N)
    eps = 1e-6

    TT = DT.QTTvec(X)
    TT.build(eps)

    L2err = npla.norm(TT.to_tensor() - X)
    if L2err <= eps:
        print_ok("Test exponential N-dimensional vector (28 points)")
        nsucc += 1
    else:
        print_fail("Test exponential N-dimensional vector (28 points): L2err=%e" % L2err)
        nfail += 1

    ################ TEST 2.2 ###########################
    # Test sum of exponential N-dimensional vector
    sys.stdout.write("Test sum of exponential N-dimensional vector\r")
    sys.stdout.flush()

    import numpy.random as npr
    R = 3
    z = npr.rand(R)
    c = npr.rand(R)
    q = 2
    L = 8
    N = q**L
    X = np.dot(c, np.tile(z,(N,1)).T ** np.tile(np.arange(N),(R,1)))

    TT = DT.QTTvec(X)
    TT.build()

    if np.max(TT.ranks()) <= R:
        print_ok("Test sum of exponential N-dimensional vector")
        nsucc += 1
    else:
        print_fail("Test sum of exponential N-dimensional vector")
        nfail += 1

    ################ TEST 2.3 ###########################
    # Test sum of trigonometric N-dimensional vector
    sys.stdout.write("Test sum of trigonometric N-dimensional vector\r")
    sys.stdout.flush()

    import numpy.random as npr
    R = 3
    a = npr.rand(R)
    c = npr.rand(R)
    q = 2
    L = 8
    N = q**L
    X = np.dot(c, np.sin(np.tile(z,(N,1)).T * np.tile(np.arange(N),(R,1))) )

    TT = DT.QTTvec(X)
    TT.build()

    if np.max(TT.ranks()) <= 2*R:
        print_ok("Test sum of trigonometric N-dimensional vector")
        nsucc += 1
    else:
        print_fail("Test sum of trigonometric N-dimensional vector")
        nfail += 1

    ################ TEST 2.4 ###########################
    # Test sum of exponential-trigonometric N-dimensional vector
    sys.stdout.write("Test sum of exponential-trigonometric N-dimensional vector\r")
    sys.stdout.flush()

    import numpy.random as npr
    R = 3
    a = npr.rand(R)
    z = npr.rand(R)
    c = npr.rand(R)
    q = 2
    L = 8
    N = q**L
    X1 = np.tile(z,(N,1)).T ** np.tile(np.arange(N),(R,1))
    X2 = np.sin(np.tile(z,(N,1)).T * np.tile(np.arange(N),(R,1)))
    X = np.dot(c, X1*X2 )

    TT = DT.QTTvec(X)
    TT.build()

    if np.max(TT.ranks()) <= 2*R:
        print_ok("Test sum of exponential-trigonometric N-dimensional vector")
        nsucc += 1
    else:
        print_fail("Test sum of exponential-trigonometric N-dimensional vector")
        nfail += 1

    ################ TEST 2.4 ###########################
    # Test sum of exponential-trigonometric N-dimensional vector
    sys.stdout.write("Test Chebyshev polynomial vector\r")
    sys.stdout.flush()

    from SpectralToolbox import Spectral1D as S1D
    P = S1D.Poly1D(S1D.JACOBI,[-0.5,-0.5])
    q = 2
    L = 8
    N = q**L
    (x,w) = P.GaussQuadrature(N-1)
    X = P.GradEvaluate(x,N-1,0).flatten()

    TT = DT.QTTvec(X)
    TT.build()

    if np.max(TT.ranks()) <= 2:
        print_ok("Test Chebyshev polynomial vector")
        nsucc += 1
    else:
        print_fail("Test Chebyshev polynomial vector")
        nfail += 1

    ################ TEST 2.5 ###########################
    # Test N-dimensional vector
    sys.stdout.write("Test generic polynomial equidistant vector\r")
    sys.stdout.flush()

    from SpectralToolbox import Spectral1D as S1D
    import numpy.random as npr
    R = 100
    c = npr.rand(R+1) - 0.5
    q = 2
    L = 16
    N = q**L
    x = np.linspace(-1,1,N)

    X = np.dot(c, np.tile(x,(R+1,1))**np.tile(np.arange(R+1),(N,1)).T)

    TT = DT.QTTvec(X)
    TT.build(eps=1e-6)

    if np.max(TT.ranks()) <= R+1:
        print_ok("Test generic polynomial (ord=%d) equidistant vector" % R)
        nsucc += 1
    else:
        print_fail("Test generic polynomial (ord=%d) equidistant vector" % R)
        nfail += 1

    ################ TEST 2.6 ###########################
    # Test N-dimensional vector
    sys.stdout.write("Test 1/(1+25x^2) Cheb vector\r")
    sys.stdout.flush()

    TT_eps = 1e-6
    from SpectralToolbox import Spectral1D as S1D
    P = S1D.Poly1D(S1D.JACOBI,[-0.5,-0.5])
    q = 2
    L = 16
    N = q**L
    (x,w) = P.GaussQuadrature(N-1)
    X = 1./(1.+25.*x**2.)

    TT = DT.QTTvec(X)
    TT.build(eps=1e-6)

    import numpy.linalg as npla
    V = P.GradVandermonde1D(x,60,0)
    (xhat,res,rnk,s) = npla.lstsq(V,X) # Polynomial approximation is better

    print_ok("Test 1/(1+25x^2) Cheb vector: Max-rank = %d, Size = %d, Poly-int res = %e" % (np.max(TT.ranks()),TT.size(),res))
    nsucc += 1

    # ################ TEST 2.7 ###########################
    # # Test discontinuos function N-dimensional vector
    # sys.stdout.write("Test discontinuous vector\r")
    # sys.stdout.flush()

    # TT_eps = 1e-6
    # from SpectralToolbox import Spectral1D as S1D
    # P = S1D.Poly1D(S1D.JACOBI,[-0.5,-0.5])
    # q = 2
    # L = 16
    # N = q**L
    # (x,w) = P.GaussQuadrature(N-1)
    # X = (x<-0.1).astype(float) - (x>0.1).astype(float)

    # TT = DT.QTTvec(X,q,eps=1e-6)

    # import numpy.linalg as npla
    # V = P.GradVandermonde1D(x,TT.size(),0)
    # (xhat,res,rnk,s) = npla.lstsq(V,X) # Polynomial approximation is better

    # print_ok("Test discontinuous vector: Max-rank = %d, Size = %d, Eps = %e, Poly-int res = %e" % (np.max(TT.ranks()),TT.size(),TT_eps,res))

    ################# TEST 3.1 ##########################
    # Test d-dimensional Laplace operator
    # Scaling of storage for d-dimensional Laplace operator:
    # 1) Full tensor product: N^(2d)
    # 2) Sparse tensor product: ~ (3N)^d
    # 3) QTT format: 1D -> max-rank = 3: ~ 3*4*3*log2(N)
    #                dD -> max-rank = 4: ~ d*4*4*4*log2(N)

    d = 4
    span = np.array([0.,1.])
    q = 2
    L = 5
    N = q**L
    h = 1/float(N-1)
    TT_round = 1e-13

    D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    #D[0,0:2] = np.array([1.,0.])
    #D[-1,-2:] = np.array([0.,1.])

    D_tensor= DT.matkron_to_mattensor(D,nrows=N,ncols=N)
    TT_D = DT.QTTmat(D_tensor,base=q,nrows=N,ncols=N)
    TT_D.build(eps=TT_round)

    I = np.eye(N)
    I_tensor= DT.matkron_to_mattensor(I,nrows=N,ncols=N)
    TT_I = DT.QTTmat(I_tensor,base=q,nrows=N,ncols=N)
    TT_I.build(eps=TT_round)

    tt_list = []
    for i in range(d):
        if i == 0: tmp = TT_D.copy()
        else: tmp = TT_I.copy()
        for j in range(1,d):
            if i == j: tmp.kron(TT_D)
            else: tmp.kron(TT_I)
        tt_list.append(tmp)

    TT_Dxy = np.sum(tt_list).rounding(TT_round)

    if d == 2 and N <= 8:
        sys.stdout.write("Test 2-dimensional laplace from kron of 1D QTTmat\r")
        sys.stdout.flush()

        Dd = np.zeros((N**d,N**d))
        for i in range(d):
            tmp = np.array([1])
            for j in range(d):
                if i != j :
                    tmp = np.kron(tmp,I)
                else:
                    tmp = np.kron(tmp,D)
            Dd += tmp

        # Check equality with Dd
        nrows = [N for i in range(d)]
        ncols = [N for i in range(d)]
        err = []
        test = True
        for i in range(N**d):
            for j in range(N**d):
                sys.stdout.write("i = %d, j = %d \r" % (i,j))
                sys.stdout.flush()
                if np.abs(Dd[i,j] - TT_Dxy[DT.idxfold(nrows,i),DT.idxfold(ncols,j)]) > TT_round:
                    err.append((i,j))
                    test = False

        if test:
            print_ok("Test 2-dimensional laplace from kron of 1D QTTmat")
            nsucc += 1
        else:
            print_fail("Test 2-dimensional laplace from kron of 1D QTTmat")
            nfail += 1

    ################# TEST 3.2 ######################################
    # Test 2-dimensional Laplace operator from full tensor product

    if d == 2 and N <= 8:
        sys.stdout.write("Test 2-dimensional laplace from full kron product\r")
        sys.stdout.flush()

        Dd = np.zeros((N**d,N**d))
        for i in range(d):
            tmp = np.array([1])
            for j in range(d):
                if i != j :
                    tmp = np.kron(tmp,I)
                else:
                    tmp = np.kron(tmp,D)
            Dd += tmp

        Dd_tensor = DT.matkron_to_mattensor(Dd,nrows=[N for i in range(d)],ncols=[N for i in range(d)])
        TT_Dxykron = DT.QTTmat(Dd_tensor, base=q,nrows=[N for i in range(d)],ncols=[N for i in range(d)])
        TT_Dxykron.build()

        # Check equality with Dd
        nrows = [N for i in range(d)]
        ncols = [N for i in range(d)]
        err = []
        test = True
        for i in range(N**d):
            for j in range(N**d):
                sys.stdout.write("i = %d, j = %d \r" % (i,j))
                sys.stdout.flush()
                if np.abs(Dd[i,j] - TT_Dxykron[DT.idxfold(nrows,i),DT.idxfold(ncols,j)]) > TT_round:
                    err.append((i,j))
                    test = False

        if test:
            print_ok("Test 2-dimensional laplace from full kron product")
            nsucc += 1
        else:
            print_fail("Test 2-dimensional laplace from full kron product")
            nfail += 1

    ################# TEST 4.0 #########################################
    # Solve the d-dimensional Dirichlet-Poisson equation using full matrices
    # Use Conjugate-Gradient method

    d = 3
    span = np.array([0.,1.])
    q = 2
    L = 4
    N = q**L
    h = 1/float(N-1)
    X = np.linspace(span[0],span[1],N)
    eps_cg = 1e-13

    sys.stdout.write("%d-dim Dirichlet-Poisson problem FULL with CG\r" % d)
    sys.stdout.flush()


    try:
        # Construct d-D Laplace (with 2nd order finite diff)
        D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
        D[0,0:2] = np.array([1.,0.])
        D[-1,-2:] = np.array([0.,1.])
        D_sp = sp.coo_matrix(D)
        I_sp = sp.identity(N)
        I = np.eye(N)
        FULL_LAP = sp.coo_matrix((N**d,N**d))
        for i in range(d):
            tmp = sp.identity((1))
            for j in range(d):
                if i != j: tmp = sp.kron(tmp,I_sp)
                else: tmp = sp.kron(tmp,D_sp)
            FULL_LAP = FULL_LAP + tmp
    except MemoryError:
        print("FULL CG: Memory Error")
        dofull = False

    # Construct Right hand-side (b=1, Dirichlet BC = 0)
    b1D = np.ones(N)
    b1D[0] = 0.
    b1D[-1] = 0.
    tmp = np.array([1.])
    for j in range(d):
        tmp = np.kron(tmp,b1D)
    FULL_b = tmp

    # Solve full system using npla.solve
    (FULL_RES,FULL_CONV) = spla.cg(FULL_LAP,FULL_b,tol=eps_cg)

    if PLOTTING and d == 2:
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm
        X = np.linspace(span[0],span[1],N)
        (XX,YY) = np.meshgrid(X,X)
        fig = plt.figure(figsize=(14,10))

    ################# TEST 4.1 #########################################
    # Solve the 2-dimensional Dirichlet-Poisson equation using QTTmat and QTTvec
    # Use Conjugate-Gradient method

    sys.stdout.write("%d-dim Dirichlet-Poisson problem QTTmat,QTTvec with CG\r" % d)
    sys.stdout.flush()

    TT_round = 1e-8
    eps_cg = 1e-3

    # Laplace operator
    D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    D[0,0:2] = np.array([1.,0.])
    D[-1,-2:] = np.array([0.,1.])

    D_tensor = DT.matkron_to_mattensor(D,nrows=N,ncols=N)
    TT_D = DT.QTTmat(D_tensor,base=q,nrows=N,ncols=N)
    TT_D.build(eps=TT_round)

    I = np.eye(N)
    I_tensor= DT.matkron_to_mattensor(I,nrows=N,ncols=N)
    TT_I = DT.QTTmat(I_tensor,base=q,nrows=N,ncols=N)
    TT_I.build(eps=TT_round)

    tt_list = []
    for i in range(d):
        if i == 0: tmp = TT_D.copy()
        else: tmp = TT_I.copy()
        for j in range(1,d):
            if i == j: tmp.kron(TT_D)
            else: tmp.kron(TT_I)
        tt_list.append(tmp)

    TT_Dxy = np.sum(tt_list).rounding(TT_round)

    # Right hand side
    b1D = np.ones(N)
    b1D[0] = 0.
    b1D[-1] = 0.

    B = np.array([1.])
    for j in range(d):
        B = np.kron(B,b1D)
    B = np.reshape(B,[N for i in range(d)])

    TT_B = DT.QTTvec(B)
    TT_B.build(TT_round)

    # Solve QTT cg
    x0 = DT.QTTzerosvec(d=d,N=N,base=q)

    cg_start = time.clock()
    (TT_RES,TT_conv,TT_info) = mla.cg(TT_Dxy,TT_B,x0=x0,eps=eps_cg,ext_info=True,eps_round=TT_round)
    cg_stop = time.clock()

    L2err = mla.norm(TT_RES.to_tensor().reshape([N for i in range(d)])-FULL_RES.reshape([N for i in range(d)]), 'fro')
    if L2err  < eps_cg:
        print_ok("%d-dim Dirichlet-Poisson problem QTTmat,QTTvec with CG      [PASSED] Time: %.10f\n" % (d, cg_stop-cg_start))
        nsucc += 1
    else:
        print_fail("%d-dim Dirichlet-Poisson problem QTTmat,QTTvec with CG      [FAILED] L2err: %.e\n" % (d,L2err))
        nfail += 1

    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(321,projection='3d')
        ax.plot_surface(XX,YY,TT_RES.to_tensor().reshape((N,N)),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax = fig.add_subplot(322,projection='3d')
        ax.plot_surface(XX,YY,np.abs(TT_RES.to_tensor().reshape((N,N))-FULL_RES.reshape((N,N))),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)


    ################# TEST 4.2 #########################################
    # Solve the 2-dimensional Dirichlet-Poisson equation using QTTmat and np.ndarray
    # Use Conjugate-Gradient method

    sys.stdout.write("%d-dim Dirichlet-Poisson problem QTTmat,ndarray with CG\r" % d)
    sys.stdout.flush()

    TT_round = 1e-8
    eps_cg = 1e-3

    # Laplace operator
    D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    D[0,0:2] = np.array([1.,0.])
    D[-1,-2:] = np.array([0.,1.])

    D_tensor = DT.matkron_to_mattensor(D,nrows=N,ncols=N)
    TT_D = DT.QTTmat(D_tensor, base=q,nrows=N,ncols=N)
    TT_D.build(eps=TT_round)

    I = np.eye(N)
    I_tensor = DT.matkron_to_mattensor(I,nrows=N,ncols=N)
    TT_I = DT.QTTmat(I_tensor,base=q,nrows=N,ncols=N)
    TT_I.build(eps=TT_round)

    tt_list = []
    for i in range(d):
        if i == 0: tmp = TT_D.copy()
        else: tmp = TT_I.copy()
        for j in range(1,d):
            if i == j: tmp.kron(TT_D)
            else: tmp.kron(TT_I)
        tt_list.append(tmp)

    TT_Dxy = np.sum(tt_list).rounding(TT_round)

    # Right hand side
    b1D = np.ones(N)
    b1D[0] = 0.
    b1D[-1] = 0.

    B = np.array([1.])
    for j in range(d):
        B = np.kron(B,b1D)
    B = np.reshape(B,[N for i in range(d)])
    B = np.reshape(B,[q for i in range(d*L)])

    # Solve QTT cg
    x0 = np.zeros([q for i in range(d*L)])

    cg_start = time.clock()
    (ARR_RES,TT_conv,TT_info1) = mla.cg(TT_Dxy,B,x0=x0,eps=eps_cg,ext_info=True,eps_round=TT_round)
    cg_stop = time.clock()

    L2err = mla.norm(ARR_RES.reshape([N for i in range(d)])-FULL_RES.reshape([N for i in range(d)]), 'fro')
    if L2err  < eps_cg:
        print_ok("%d-dim Dirichlet-Poisson problem QTTmat,ndarray with CG     [PASSED] Time: %.10f" % (d, cg_stop-cg_start))
        nsucc += 1
    else:
        print_fail("%d-dim Dirichlet-Poisson problem QTTmat,ndarray with CG     [FAILED] L2err: %.e" % (d,L2err))
        nfail += 1

    if PLOTTING and d == 2:
        # Plot function
        ax = fig.add_subplot(323,projection='3d')
        ax.plot_surface(XX,YY,ARR_RES.reshape((N,N)),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        ax = fig.add_subplot(324,projection='3d')
        ax.plot_surface(XX,YY,np.abs(ARR_RES.reshape((N,N))-FULL_RES.reshape((N,N))),rstride=1, cstride=1, cmap=cm.coolwarm,
                        linewidth=0, antialiased=False)
        plt.show(block=False)

    # ################# TEST 4.3 #########################################
    # # Solve the 2-dimensional Dirichlet-Poisson equation using QTTmat and np.ndarray
    # # Use Preconditioned Conjugate-Gradient method

    # sys.stdout.write("%d-dim Dirichlet-Poisson problem QTTmat,ndarray with Prec-CG\r" % d)
    # sys.stdout.flush()

    # TT_round = 1e-8
    # eps_cg = 1e-3

    # # Laplace operator
    # D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
    # D[0,0:2] = np.array([1.,0.])
    # D[-1,-2:] = np.array([0.,1.])

    # D_tensor = DT.matkron_to_mattensor(D,nrows=N,ncols=N)
    # TT_D = DT.QTTmat(D_tensor, base=q,nrows=N,ncols=N,eps=TT_round)

    # I = np.eye(N)
    # I_tensor = DT.matkron_to_mattensor(I,nrows=N,ncols=N)
    # TT_I = DT.QTTmat(I_tensor,base=q,nrows=N,ncols=N,eps=TT_round)

    # tt_list = []
    # for i in range(d):
    #     if i == 0: tmp = TT_D.copy()
    #     else: tmp = TT_I.copy()
    #     for j in range(1,d):
    #         if i == j: tmp.kron(TT_D)
    #         else: tmp.kron(TT_I)
    #     tt_list.append(tmp)

    # TT_Dxy = np.sum(tt_list).rounding(TT_round)

    # # Construct Preconditioner using Newton-iterations
    # TT_II = TT_I.copy()
    # for j in range(1,d): TT_II.kron(TT_I)
    # alpha = 1e-6
    # TT_Pround = 1e-4
    # TT_P = alpha*TT_II
    # eps = mla.norm(TT_II-mla.dot(TT_Dxy,TT_P),'fro')/mla.norm(TT_II,'fro')
    # i = 0
    # while eps > 5.*1e-1:
    #     i += 1
    #     TT_P  = (2. * TT_P - mla.dot(TT_P,mla.dot(TT_Dxy,TT_P).rounding(TT_Pround)).rounding(TT_Pround)).rounding(TT_Pround)
    #     eps = mla.norm(TT_II-mla.dot(TT_Dxy,TT_P),'fro')/mla.norm(TT_II,'fro')
    #     sys.stdout.write("\033[K")
    #     sys.stdout.write("Prec: err=%e, iter=%d\r" % (eps,i))
    #     sys.stdout.flush()

    # # Right hand side
    # b1D = np.ones(N)
    # b1D[0] = 0.
    # b1D[-1] = 0.

    # B = np.array([1.])
    # for j in range(d):
    #     B = np.kron(B,b1D)
    # B = np.reshape(B,[N for i in range(d)])
    # B = np.reshape(B,[q for i in range(d*L)])

    # # Solve QTT cg
    # x0 = np.zeros([q for i in range(d*L)])

    # # Precondition
    # TT_DP = mla.dot(TT_P,TT_Dxy).rounding(TT_round)
    # BP = mla.dot(TT_P,B)

    # cg_start = time.clock()
    # (ARR_RES,TT_conv,TT_info) = mla.cg(TT_DP,BP,x0=x0,eps=eps_cg,ext_info=True,eps_round=TT_round)
    # cg_stop = time.clock()

    # L2err = mla.norm(ARR_RES.reshape([N for i in range(d)])-FULL_RES.reshape([N for i in range(d)]), 'fro')
    # if L2err  < eps_cg:
    #     print_ok("%d-dim Dirichlet-Poisson problem QTTmat,ndarray with Prec-CG [PASSED] Time: %.10f" % (d, cg_stop-cg_start))
    # else:
    #     print_fail("%d-dim Dirichlet-Poisson problem QTTmat,ndarray with Prec-CG [FAILED] L2err: %.e" % (d,L2err))

    # if PLOTTING and d == 2:
    #     # Plot function
    #     ax = fig.add_subplot(325,projection='3d')
    #     ax.plot_surface(XX,YY,ARR_RES.reshape((N,N)),rstride=1, cstride=1, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)
    #     ax = fig.add_subplot(326,projection='3d')
    #     ax.plot_surface(XX,YY,np.abs(ARR_RES.reshape((N,N))-FULL_RES.reshape((N,N))),rstride=1, cstride=1, cmap=cm.coolwarm,
    #                     linewidth=0, antialiased=False)
    #     plt.show(block=False)


    print_summary("QTT", nsucc, nfail)
    
    return (nsucc,nfail)

if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestQTT.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs, PLOTTING=True, loglev=logging.INFO)

