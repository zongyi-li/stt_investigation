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
    # Test matrix 2-norm on random matrices
    #####################################################################################
    span = np.array([0.,1.])
    d = 3
    nrows = 16
    ncols = 16
    if isinstance(nrows,int): nrows = [nrows for i in range(d)]
    if isinstance(ncols,int): ncols = [ncols for i in range(d)]
    eps = 1e-6
    round_eps = 1e-12

    # sys.stdout.write("Matrix 2-norm: Random\n  nrows=[%s],\n  ncols=[%s],  d=%3d      [START] \n" % (','.join(map(str,nrows)),','.join(map(str,ncols)),d))
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

    # Check results
    tt_norm = mla.norm(TT_RAND,2,round_eps=round_eps,eps=eps)
    full_norm = npla.norm(FULL_RAND,2)
    if np.abs(tt_norm-full_norm)/npla.norm(FULL_RAND,'fro') <= 0.02:
        print_ok("3.1 Matrix 2-norm: Random  nrows=%s, ncols=%s , d=%3d  , TT-norm = %.5f , FULL-norm = %.5f" % (str(nrows),str(ncols),d,tt_norm,full_norm))
        nsucc += 1
    else:
        print_fail("3.1 Matrix 2-norm: Random  nrows=%s, ncols=%s, d=%3d  , TT-norm = %.5f , FULL-norm = %.5f" % (str(nrows),str(ncols),d,tt_norm,full_norm),'')
        nfail += 1


# opt = 'a'
# while (opt != 'c' and opt != 's' and opt != 'q'):
#     print("Matrix-vector product test with Schrodinger operator:")
#     print("\t [c]: continue")
#     print("\t [s]: skip")
#     print("\t [q]: exit")
#     opt = sys.stdin.read(1)
#     if (opt ==  'q'):
#         exit(0)

# if opt == 'c':
#     #####################################################################################
#     # Test matrix-vector product by computing the smallest eigenvalue of the operator in
#     # "Tensor-Train decomposition" I.V.Oseledets
#     # "Algorithms in high dimensions" Beylkin and Mohlenkamp
#     #####################################################################################
#     span = np.array([0.,1.])
#     d = 2
#     N = 16
#     h = 1/float(N-1)
#     cv = 100.
#     cw = 5.
#     eps =1e-10

#     # Construction of TT Laplace operator
#     CPtmp = []
#     D = -1./h**2. * ( np.diag(np.ones((N-1)),-1) + np.diag(np.ones((N-1)),1) + np.diag(-2.*np.ones((N)),0) )
#     I = np.eye(N)
#     D_flat = D.flatten()
#     I_flat = I.flatten()
#     for i in range(d):
#         CPi = np.empty((d,N**2))
#         for alpha in range(d):
#             if i != alpha:
#                 CPi[alpha,:] = I_flat
#             else:
#                 CPi[alpha,:] = D_flat
#         CPtmp.append(CPi)

#     CP_lap = DT.Candecomp(CPtmp)
#     TT_lap = DT.TTmat(CP_lap,nrows=N,ncols=N)
#     TT_lap.rounding(eps)
#     CPtmp = None
#     CP_lap = None

#     # Construction of TT Potential operator
#     CPtmp = []
#     X = np.linspace(span[0],span[1],N)
#     # B = np.diag(np.cos(2.*np.pi*X),0)
#     B = np.diag(np.cos(X),0)
#     I = np.eye(N)
#     B_flat = B.flatten()
#     I_flat = I.flatten()
#     for i in range(d):
#         CPi = np.empty((d,N**2))
#         for alpha in range(d):
#             if i != alpha:
#                 CPi[alpha,:] = I_flat
#             else:
#                 CPi[alpha,:] = B_flat
#         CPtmp.append(CPi)

#     CP_pot = DT.Candecomp(CPtmp)
#     TT_pot = DT.TTmat(CP_pot,nrows=N,ncols=N)
#     TT_pot.rounding(eps)
#     CPtmp = None
#     CP_pot = None

#     # Construction of TT electron-electron interaction
#     CPtmp_cos = []
#     CPtmp_sin = []
#     X = np.linspace(span[0],span[1],N)
#     # Bcos = np.diag(np.cos(2.*np.pi*X),0)
#     # Bsin = np.diag(np.sin(2.*np.pi*X),0)
#     Bcos = np.diag(np.cos(X),0)
#     Bsin = np.diag(np.sin(X),0)
#     I = np.eye(N)
#     # D_flat = D.flatten()
#     Bcos_flat = Bcos.flatten()
#     Bsin_flat = Bsin.flatten()
#     I_flat = I.flatten()

#     for i in range(d):
#         CPi_cos = np.zeros((d*(d-1)/2,N**2))
#         CPi_sin = np.zeros((d*(d-1)/2,N**2))
#         k=0
#         for alpha in range(d):
#             for beta in range(alpha+1,d):
#                 if alpha == i or beta == i :
#                     CPi_cos[k,:] = Bcos_flat
#                     CPi_sin[k,:] = Bsin_flat
#                 else:
#                     CPi_cos[k,:] = I_flat
#                     CPi_sin[k,:] = I_flat
#                 k += 1
#         CPtmp_cos.append(CPi_cos)
#         CPtmp_sin.append(CPi_sin)

#     CP_int_cos = DT.Candecomp(CPtmp_cos)
#     CP_int_sin = DT.Candecomp(CPtmp_sin)
#     TT_int_cos = DT.TTmat(CP_int_cos,nrows=N,ncols=N)
#     TT_int_sin = DT.TTmat(CP_int_sin,nrows=N,ncols=N)
#     TT_int_cos.rounding(eps)
#     TT_int_sin.rounding(eps)
#     TT_int = (TT_int_cos + TT_int_sin).rounding(eps)
#     CPtmp_cos = None
#     CPtmp_sin = None
#     CP_int_cos = None
#     CP_int_sin = None

#     # # Construction of TT Scholes-tensor
#     # CPtmp = []
#     # X = np.linspace(span[0],span[1],N)
#     # D = 1./(2*h) * (np.diag(np.ones(N-1),1) - np.diag(np.ones(N-1),-1))
#     # D[0,0] = -1./h
#     # D[0,1] = 1./h
#     # D[-1,-1] = 1./h
#     # D[-1,-2] = -1./h
#     # I = np.eye(N)
#     # D_flat = D.flatten()
#     # I_flat = I.flatten()
#     # for i in range(d):
#     #     CPi = np.zeros((d*(d-1)/2,N**2))
#     #     k = 0
#     #     for alpha in range(d):
#     #         for beta in range(alpha+1,d):
#     #             if alpha == i:
#     #                 CPi[k,:] = D_flat
#     #             elif beta == i:
#     #                 CPi[k,:] = D_flat
#     #             else:
#     #                 CPi[k,:] = I_flat
#     #             k += 1
#     #     CPtmp.append(CPi)

#     # CP_sch = DT.Candecomp(CPtmp)
#     # TT_sch = DT.TTmat(CP_sch,nrows=N,ncols=N)
#     # TT_sch.rounding(eps)

#     H = (TT_lap + TT_pot + TT_int).rounding(eps)
#     Cd = mla.norm(H,2)

#     # Identity tensor
#     TT_id = DT.eye(d,N)

#     Hhat = (Cd * TT_id - H).rounding(eps)


    print_summary("TT Norms", nsucc, nfail)
    
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
