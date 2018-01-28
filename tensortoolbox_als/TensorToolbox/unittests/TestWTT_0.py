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
    # Test Weighted TT. Weights are uniform.

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

    N = 16
    d = 3
    nrows = [N for i in range(d)]
    ncols = [N for i in range(d)]
    D = np.diag(-np.ones((N-1)),-1) + np.diag(-np.ones((N-1)),1) + np.diag(2*np.ones((N)),0)
    I = np.eye(N)
    Dd = np.zeros((N**d,N**d))

    for i in range(d):
        tmp = np.array([1])
        for j in range(d):
            if i != j:
                tmp = np.kron(tmp,I)
            else:
                tmp = np.kron(tmp,D)
        Dd += tmp

    if PLOTTING:
        plt.figure()
        plt.spy(Dd)
        plt.show(block=False)

    idxs = [range(N) for i in range(d)]
    MI = list(itertools.product(*idxs)) # Multi indices

    # Canonical form of n-dimentional Laplace operator
    D_flat = D.flatten()
    I_flat = I.flatten()

    # CP = np.empty((d,d,N**2),dtype=np.float64) 
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

    # Let's compare Dd[i,j] with its Canonical counterpart
    T_idx = (10,9) # Index in the tensor product repr.
    idxs = np.vstack( (np.asarray(MI[T_idx[0]]), np.asarray(MI[T_idx[1]])) ) # row 1 contains row multi-idx, row 2 contains col multi-idx for Tensor
    # Now if we take the columns of idxs we get the multi-indices for the CP.
    # Since in CP we flattened the array, compute the corresponding indices for CP.
    CP_idxs = idxs[0,:]*N + idxs[1,:]

    TT = DT.TTmat(CP,nrows=N,ncols=N)
    TT.build()

    if np.abs(Dd[T_idx[0],T_idx[1]] - CP[CP_idxs]) < 100.*np.spacing(1) and np.abs(CP[CP_idxs] - TT[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]) < 100.*np.spacing(1):
        print_ok("0.1 Weighted Tensor Test: Entry comparison (pre-rounding) FULL, CP, TT")
        nsucc += 1
    else:
        print_fail("0.1 Weighted Tensor Test: Entry comparison FULL, CP, TT")
        nfail += 1
        # print("  T      CP     TT")
        # print("%.5f  %.5f  %.5f" % (Dd[T_idx[0],T_idx[1]],CP[CP_idxs],TT[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]))

    # print("Space Tensor: %d" % np.prod(Dd.shape))
    # print("Space CP: %d" % CP.size())
    # print("Space TT: %d" % TT.size())

    ########################################
    # Multi-Linear Algebra
    ########################################

    # Sum by scalar
    CPa = DT.Candecomp([5.*np.ones((1,5)),np.ones((1,6)),np.ones((1,7))])
    W = [ np.ones(5)/5., np.ones(6)/6., np.ones(7)/7.]
    TTa = DT.WTTvec(CPa,W)
    TTa.build(1e-13)
    TTb = TTa + 3.
    if np.abs(TTb[3,3,3] - 8.) < 1e-12:
        print_ok("0.2 Weighted Tensor Test: TT sum by scalar")
        nsucc += 1
    else:
        print_fail("0.2 Weighted Tensor Test: TT sum by scalar", "TT[idx] + b = %e, Expected = %e" % (TTb[3,3,3],8.))
        nfail += 1

    # Diff by scalar
    CPa = DT.Candecomp([5.*np.ones((1,5)),np.ones((1,6)),np.ones((1,7))])
    W = [ np.ones(5)/5., np.ones(6)/6., np.ones(7)/7.]
    TTa = DT.WTTvec(CPa,W)
    TTa.build(1e-13)
    TTb = TTa - 3.
    if np.abs(TTb[3,3,3] - 2.) < 1e-12:
        print_ok("0.2 Weighted Tensor Test: TT diff by scalar")
        nsucc += 1
    else:
        print_fail("0.2 Weighted Tensor Test: TT diff by scalar", "TT[idx] + b = %e, Expected = %e" % (TTb[3,3,3],2.))
        nfail += 1

    # Mul by scalar
    CPa = DT.Candecomp([5.*np.ones((1,5)),np.ones((1,6)),np.ones((1,7))])
    W = [ np.ones(5)/5., np.ones(6)/6., np.ones(7)/7.]
    TTa = DT.WTTvec(CPa,W)
    TTa.build(1e-13)
    TTb = TTa * 3.
    if np.abs(TTb[3,3,3] - 15.) < 1e-12:
        print_ok("0.2 Weighted Tensor Test: TT mul by scalar")
        nsucc += 1
    else:
        print_fail("0.2 Weighted Tensor Test: TT mul by scalar", "TT[idx] + b = %e, Expected = %e" % (TTb[3,3,3],15.))
        nfail += 1

    # Div by scalar
    CPa = DT.Candecomp([15.*np.ones((1,5)),np.ones((1,6)),np.ones((1,7))])
    W = [ np.ones(5)/5., np.ones(6)/6., np.ones(7)/7.]
    TTa = DT.WTTvec(CPa,W)
    TTa.build(1e-13)
    TTb = TTa / 3.
    if np.abs(TTb[3,3,3] - 5.) < 1e-12:
        print_ok("0.2 Weighted Tensor Test: TT div by scalar")
        nsucc += 1
    else:
        print_fail("0.2 Weighted Tensor Test: TT div by scalar", "TT[idx] + b = %e, Expected = %e" % (TTb[3,3,3],5.))
        nfail += 1

    # Sum
    C = TT + TT
    if  C[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])] - 2. * Dd[T_idx[0],T_idx[1]] <= 2e2 * np.spacing(1):
        print_ok("0.2 Weighted Tensor Test: TT sum")
        nsucc += 1
    else:
        print_fail("0.2 Weighted Tensor Test: TT sum", "TT[idx] + TT[idx] = %.5f" % C[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])])
        nfail += 1

    C = TT * TT
    if  C[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])] - Dd[T_idx[0],T_idx[1]]**2. <= 10.*np.spacing(1):
        print_ok("0.3 Weighted Tensor Test: TT mul")
        nsucc += 1
    else:
        print_fail("0.3 Weighted Tensor Test: TT mul", "TT[idx] * TT[idx] = %.5f" % C[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])])
        nfail += 1

    # C *= (C+TT)
    # if  C[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])] == Dd[T_idx[0],T_idx[1]]**2. * (Dd[T_idx[0],T_idx[1]]**2.+Dd[T_idx[0],T_idx[1]]):
    #     print_ok("0.4 Weighted Tensor Test: TT operations")
    # else:
    #     print_fail("0.4 Weighted Tensor Test: TT operations", "(TT*TT)*(TT*TT+TT) = %.5f" % C[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])])

    if np.abs(npla.norm(Dd,ord='fro')-mla.norm(TT,ord='fro')) < TT.size() * 100.*np.spacing(1):
        print_ok("0.5 Weighted Tensor Test: Frobenius norm (pre-rounding) FULL, TT")
        nsucc += 1
    else:
        print_fail("0.5 Weighted Tensor Test: Frobenius norm (pre-rounding) FULL, TT",
                   "                  T          TT\n"\
                       "Frobenius norm  %.5f         %.5f" % (npla.norm(Dd,ord='fro'), DT.norm(TT,ord='fro')))
        nfail += 1

    #######################################
    # Check TT-SVD
    #######################################

    # Contruct tensor form of Dd
    Dd_flat = np.zeros((N**(2*d)))
    for i in range(d):
        tmp = np.array([1])
        for j in range(d):
            if i != j:
                tmp = np.kron(tmp,I_flat)
            else:
                tmp = np.kron(tmp,D_flat)
        Dd_flat += tmp

    Dd_tensor= Dd_flat.reshape([N**2 for j in range(d)])

    TT_tensor = TT.to_tensor()

    # From Dd_tensor obtain a TT representation with accuracy eps
    eps = 0.001
    TT_svd = DT.TTmat(Dd_tensor,nrows=N,ncols=N)
    TT_svd.build(eps=eps)

    if np.abs(Dd[T_idx[0],T_idx[1]] - TT_svd[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]) < d * 100.*np.spacing(1):
        print_ok("0.6 Weighted Tensor Test: Entry comparison FULL, TT-svd")
        nsucc += 1
    else:
        print_fail("0.6 Weighted Tensor Test: Entry comparison FULL, TT-svd","  T - TT-svd = %e" % np.abs(Dd[T_idx[0],T_idx[1]] - TT_svd[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]))
        nfail += 1

    Dd_norm = npla.norm(Dd,ord='fro')
    TT_svd_norm = mla.norm(TT_svd,ord='fro')
    if np.abs(Dd_norm - TT_svd_norm) < eps * Dd_norm:
        print_ok("0.6 Weighted Tensor Test: Frobenius norm FULL, TT-svd")
        nsucc += 1
    else:
        print_fail("0.6 Weighted Tensor Test: Frobenius norm FULL, TT-svd",
                   "                  T          TT_svd\n"\
                       "Frobenius norm  %.5f         %.5f" % (npla.norm(Dd,ord='fro'), mla.norm(TT_svd,ord='fro')))
        nfail += 1

    #######################################
    # Check TT-SVD with kron prod
    #######################################

    # Contruct tensor form of Dd
    Dd = np.zeros((N**d,N**d))
    for i in range(d):
        tmp = np.array([1])
        for j in range(d):
            if i != j:
                tmp = np.kron(tmp,I)
            else:
                tmp = np.kron(tmp,D)
        Dd += tmp

    Dd_tensor = DT.matkron_to_mattensor(Dd,[N for i in range(d)],[N for i in range(d)])

    TT_tensor = TT.to_tensor()

    # From Dd_tensor obtain a TT representation with accuracy eps
    eps = 0.001
    TT_svd = DT.TTmat(Dd_tensor,nrows=N,ncols=N)
    TT_svd.build(eps=eps)

    if np.abs(Dd[T_idx[0],T_idx[1]] - TT_svd[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]) < d * 100.*np.spacing(1):
        print_ok("0.7 Weighted Tensor Test: Entry comparison FULL, TT-svd-kron")
        nsucc += 1
    else:
        print_fail("0.7 Weighted Tensor Test: Entry comparison FULL, TT-svd-kron",
                   "  T - TT-svd = %e" % np.abs(Dd[T_idx[0],T_idx[1]]-TT_svd[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]))
        nfail += 1

    Dd_norm = npla.norm(Dd,ord='fro')
    TT_svd_norm = mla.norm(TT_svd,ord='fro')
    if np.abs(Dd_norm - TT_svd_norm) < eps * Dd_norm:
        print_ok("0.7 Weighted Tensor Test: Frobenius norm FULL, TT-svd-kron")
        nsucc += 1
    else:
        print_fail("0.7 Weighted Tensor Test: Frobenius norm FULL, TT-svd-kron",
                   "                  T          TT_svd\n"\
                       "Frobenius norm  %.5f         %.5f" % (npla.norm(Dd,ord='fro'), mla.norm(TT_svd,ord='fro')))
        nfail += 1

    #######################################
    # Check TT-rounding
    #######################################
    TT_round = TT.copy()
    eps = 0.001
    TT_round.rounding(eps)

    if np.abs(TT_round[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])] - TT_svd[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]) < d * 100.*np.spacing(1):
        print_ok("0.8 Weighted Tensor Test: Entry comparison (post-rounding) TT-svd, TT-round")
        nsucc += 1
    else:
        print_fail("0.8 Weighted Tensor Test: Entry comparison  (post-rounding) TT-svd, TT-round",
               "  T-svd - TT-round = %e" % np.abs(TT_svd[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])] - TT_round[DT.idxfold(nrows,T_idx[0]),DT.idxfold(ncols,T_idx[1])]))
        nfail += 1

    Dd_norm = npla.norm(Dd,ord='fro')
    TT_svd_norm = mla.norm(TT_svd,ord='fro')
    TT_round_norm = mla.norm(TT_round,ord='fro')
    if np.abs(Dd_norm - TT_svd_norm) < eps * Dd_norm and np.abs(TT_svd_norm - TT_round_norm) < eps * Dd_norm:
        print_ok("0.8 Weighted Tensor Test: Frobenius norm (post-rounding) FULL, TT-svd, TT-round")
        nsucc += 1
    else:
        print_fail("0.8 Weighted Tensor Test: Frobenius norm (post-rounding) FULL, TT-svd, TT-round",
                   "                  T          TT_svd         TT_rounding\n"\
                       "Frobenius norm  %.5f         %.5f           %.5f" % (Dd_norm, TT_svd_norm, TT_round_norm))
        nfail += 1

    print_summary("WTT Algebra", nsucc, nfail)
    
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
