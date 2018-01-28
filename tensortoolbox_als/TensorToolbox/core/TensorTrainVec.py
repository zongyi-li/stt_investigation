#!/usr/bin/env python
# -*- coding: utf-8 -*-

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

__all__ = ['TTvec','randvec','zerosvec']

import logging
import sys
import warnings
import itertools
try:
    import itertools.izip as zip
except ImportError:
    pass
import copy
import operator
import random
from functools import reduce

import numpy as np
import numpy.random as npr
import numpy.linalg as npla

from scipy import linalg as scla
from scipy import sparse as scsp
from scipy.sparse import linalg as spla

from TensorToolbox.core import ConvergenceError, TTcrossLoopError,\
    TensorWrapper, maxvol, idxfold, idxunfold, storable_object,\
    Candecomp, reort, isint, isfloat
from TensorToolbox import multilinalg as mla

class TTvec(storable_object):
    """ Constructor of multidimensional tensor in Tensor Train format :cite:`Oseledets2011`
        
    :param Candecomp,ndarray,TT,TensorWrapper A: Available input formats are Candecomp, full tensor in numpy.ndarray, Tensor Train structure (list of cores), or a Tensor Wrapper.
    :param string store_location: Store computed values during construction on the specified file path. The stored values are ttcross_Jinit and the values used in the TensorWrapper. This permits a restart from already computed values. If empty string nothing is done. (method=='ttcross')
    :param string store_object: Object to be stored (default are the tensor wrapper and ttcross_Jinit)
    :param int store_freq: storage frequency. ``store_freq==1`` stores intermediate values at every iteration. The program stores data every ``store_freq`` internal iterations. If ``store_object`` is a SpectralTensorTrain, then ``store_freq`` determines the number of seconds every which to store values.
    :param int multidim_point: If the object A returns a multidimensional array, then this can be used to define which point to apply ttcross to.
    
    .. document private functions
    .. automethod:: __getitem__
    
    """

    logger = logging.getLogger(__name__)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    def __init__(self,A, 
                 store_location="",store_object=None,store_freq=1, store_overwrite=False, 
                 multidim_point=None):
        super(TTvec,self).__init__(store_location, store_freq,
                                   store_overwrite, store_object)

        ##############################
        # List of attributes
        self.A = None                    # Multidimensional data structure
        self.TT = None
        self.init = False
        self.method = 'svd'
        self.eps = None
        self.max_ranks = None
        self.multidim_point = None

        # Values only stored for cross approximations (ttcross, ttdmrg)
        self.rs = None
        self.Js = None
        self.Is = None
        self.Js_last = None
        self.Jinit = None
        self.ltor_fiber_lists = None
        self.rtol_fiber_lists = None

        # Values only stored for ttdmrg for restarting purposes
        self.dmrg_not_converged = None
        self.dmrg_sweep_it = None
        self.dmrg_sweep_direction = None
        self.dmrg_Pinv = None

        self.serialize_list.extend( [
            'TT', 'init', 'method', 'eps', 'max_ranks', 'multidim_point',
            'rs', 'Js', 'Is', 'Js_last', 'Jinit', 'ltor_fiber_lists',
            'rtol_fiber_lists', 'dmrg_not_converged', 'dmrg_sweep_it',
            'dmrg_sweep_direction', 'dmrg_Pinv'] )
        self.subserialize_list.extend( ['A'] )   # Not serialized thanks to the STT interface
        # End list of attributes
        ###############################

        # Initialize the tensor with the input tensor in TT ([][]numpy.ndarray),
        # tensor(numpy.ndarray) or CANDECOMP form
        self.A = A
        self.multidim_point = multidim_point
        
    def build( self, eps=1e-10, method='svd', rs=None, fix_rank=False, Jinit=None, delta=1e-4, maxit=100, mv_eps=1e-6, mv_maxit=100, max_ranks=None, kickrank=None):
        """ Common interface for the construction of the approximation.

        :param float eps: [default == 1e-10] For method=='svd': precision with which to approximate the input tensor. For method=='ttcross': TT-rounding tolerance for rank-check.
        :param string method: 'svd' use singular value decomposition to construct the TT representation :cite:`Oseledets2011`, 'ttcross' use low rank skeleton approximation to construct the TT representation :cite:`Oseledets2010`, 'ttdmrg' uses Tensor Train Renormalization Cross to construct the TT representation :cite:`Savostyanov2011,Savostyanov2013`, 'ttdmrgcross' uses 'ttdmrg' with 'ttcross' approximation of supercores
        :param list rs: list of integer ranks of different cores. If ``None`` then the incremental TTcross approach will be used. (method=='ttcross')
        :param bool fix_rank: determines whether the rank is allowed to be increased (method=='ttcross')
        :param list Jinit: list of list of integers containing the r starting columns in the lowrankapprox routine for each core. If ``None`` then pick them randomly. (method=='ttcross')
        :param float delta: accuracy parameter in the TT-cross routine (method=='ttcross'). It is the relative error in Frobenious norm between two successive iterations.
        :param int maxit: maximum number of iterations in the lowrankapprox routine (method=='ttcross')
        :param float mv_eps: accuracy parameter for each usage of the maxvol algorithm (method=='ttcross')
        :param int mv_maxit: maximum number of iterations in the maxvol routine (method=='ttcross')
        :param bool fix_rank: Whether the rank is allowed to increase
        :param list max_ranks: Maximum ranks to be used to limit the trunaction rank due to ``eps``. The first and last elements of the list must be ``1``, e.g. ``[1,...,1]``. Default: ``None``.
        :param int kickrank: rank overshooting for 'ttdmrg'
        """

        if isinstance(self.A,Candecomp):
            self.eps = eps
            self.max_ranks = max_ranks
            self.TT = self.A.to_TT()
            self.init = True
            self.rounding(self.eps,self.max_ranks)
        elif isinstance(self.A,np.ndarray) or isinstance(self.A,TensorWrapper):
            self.eps = eps
            self.max_ranks = max_ranks
            if method == 'svd' or method == None: 
                self.svd(self.eps, self.max_ranks)
            elif method == 'ttcross': 
                self.ttcross(self.eps,rs,Jinit,delta,maxit,mv_eps,mv_maxit,fix_rank)
            elif method == 'ttdmrg':
                self.ttdmrg( self.eps, Jinit, maxit, mv_eps, mv_maxit, kickrank=kickrank)
            elif method == 'ttdmrgcross':
                self.ttdmrgcross( self.eps, Jinit, maxit, mv_eps, mv_maxit, kickrank=kickrank)
            else: raise AttributeError("Method name not recognized. Use 'svd' or 'ttcross'")
        elif isinstance(self.A,list):
            self.TT = self.A
            # check consistency just using the ranks function
            self.ranks()
            self.init = True
        else:
            raise NameError("TensorToolbox.TTvec.__init__: Input type not allowed")
        
        return self
    
    def __getstate__(self):
        return super(TTvec,self).__getstate__()
    
    def __setstate__(self,state):
        super(TTvec,self).__setstate__( state )
        
    def __getitem__(self,idxs):
        if not self.init: raise NameError("TensorToolbox.TTvec.__getitem__: TT not initialized correctly")
        # TT format: TT[i][idx] = core (matrix or row/col vector)
        if isint(idxs):
            return self.TT[0][:,idxs,:]
        else:
            out = np.array([1.])
            for i in range(0,len(self.TT)):
                out = np.tensordot(out, self.TT[i][:,idxs[i],:], ( (out.ndim-1,), (0,) ) )
            out = np.tensordot(out,np.array([1.]), ( (out.ndim-1,), (0,) ) )
            return out

    def ranks(self):
        ranks = [1]
        for TTi in self.TT: ranks.append(TTi.shape[2])
        return ranks

    def size(self):
        # TT format: TT[i][idx] = core (matrix or row/col vector)
        return reduce( operator.add, [np.prod(TTi.shape) for TTi in self.TT] ) 

    def ndim(self):
        return len(self.TT)
    
    def shape(self):
        """
        Returns the shape of the tensor represented
        """
        return tuple( [ TTi.shape[1] for TTi in self.TT ] )
    
    def to_tensor(self):
        if not self.init: raise NameError("TensorToolbox.TTvec.to_tensor: TT not initialized correctly")
        T = np.array([1.])
        for TTi in self.TT:
            T_ax = T.ndim - 1
            T = np.tensordot(T,TTi,((T_ax,),(0,)))
        T_ax = T.ndim - 1
        T = np.tensordot(T,np.array([1.]),((T_ax,),(0,)))
        return T

    def get_data_F_norm(self):
        """ Used to get the Frobeniuos norm of the underlying data.
        This needs to be redefined in QTTvec in order to get the Frobeniuous norm of the real tensor.
        
        .. note: To get the Frobenious norm of the TT approximation, use the :py:method:`multilinalg.norm`
        """
        if isinstance(self.A, TensorWrapper):
            nrm = npla.norm(self.A[ tuple([ slice(None,None,None) ] * self.A.get_ndim()) ].flatten(),2)
        else:
            nrm = npla.norm(self.A.flatten(),2)
        return nrm

    def get_ttdmrg_real_subtensor(self,C,idx):
        """ Used to get the real subtensor of the underlying data.
        This needs to be redefined in QTTvec in order to get the subtensor of the real tensor.
        """
        return C

    def get_ttcross_eval_idxs(self):
        idxs = []
        dims = self.shape()
        for k in range(len(self.Is)-1,-1,-1):
            for i in range(len(self.Is[k])):
                for j in range(len(self.Js[k])):
                    for kk in range(dims[k]):
                        idxs.append( self.Is[k][i] + (kk,) + self.Js[k][j] )

        return np.array(idxs)

    def get_ttdmrg_eval_idxs(self):
        idxs = []
        dims = self.shape()
        d = len(dims)
        for k in range(1,d-1):
            for ii in self.Is[k-1]:
                for jj in self.Js[k]:
                    for kk in range(dims[k-1]):
                        for ll in range(dims[k]):
                            idxs.append( ii + (kk,ll) + jj )

        return np.array(idxs)

    def copy(self):
        return copy.deepcopy(self)

    ###########################################
    # Multi-linear Algebra
    ###########################################

    def __add__(A,B):
        C = A.copy()
        C += B
        return C
        
    def __iadd__(A,B):
        """
        In place addition
        """
        if not ( (isinstance(A, TTvec) and A.init) and \
                     ((isinstance(B,TTvec) and B.init) or isfloat(B)) ): 
            raise NameError("TensorToolbox.TTvec.add: TT not initialized correctly")

        if isinstance(A,TTvec) and isfloat(B):
            lscp = [ np.ones((1,sh)) for i,sh in enumerate(A.shape()) ]
            lscp[0] *= B
            CP = Candecomp( lscp )
            B = TTvec(CP)
            B.build(A.eps)
            if not B.init:
                raise NameError("TensorToolbox.TTvec.add: TT not initialized correctly")
            
        for i in range(len(A.TT)):
            if i == 0:
                A.TT[i] = np.concatenate((A.TT[i],B.TT[i]),axis=2)
            elif i == len(A.TT)-1:
                A.TT[i] = np.concatenate((A.TT[i],B.TT[i]),axis=0)
            else:
                tmpi = np.empty((A.TT[i].shape[0]+B.TT[i].shape[0], A.TT[i].shape[1], A.TT[i].shape[2]+B.TT[i].shape[2]),dtype=np.float64)
                for j in range(A.TT[i].shape[1]): tmpi[:,j,:] = scla.block_diag(A.TT[i][:,j,:],B.TT[i][:,j,:])
                A.TT[i] = tmpi

        return A

    def __radd__(A,B):
        B += A
        return B
    
    def __neg__(A):
        B = -1. * A
        return B

    def __sub__(A,B):
        return A + (-B)
    
    def __isub__(A,B):
        """
        In place subtraction
        """
        A += -B
        return A

    def __rsub__(A,B):
        B += -A
        return B

    def __truediv__(A,B): # Python3
        if not isfloat(B):
            raise AttributeError("TensorToolbox.TTvec.div: Division implemented only for floats")
        C = A.copy()
        C *= 1./B
        return C

    def __div__(A,B):
        if not isfloat(B):
            raise AttributeError("TensorToolbox.TTvec.div: Division implemented only for floats")
        C = A.copy()
        C *= 1./B
        return C

    def __idiv__(A,B):
        if not isfloat(B):
            raise AttributeError("TensorToolbox.TTvec.div: Division implemented only for floats")
        A *= 1./B
        return A
        
    def __mul__(A,B):
        """
        * If A,B are TTvec -> Hadamard product of two TT tensors
        * If A TTvec and B scalar -> multiplication by scalar
        """
        C = A.copy()
        C *= B
        return C
    
    def __rmul__(A,B):
        """
        * If A TTvec and B scalar -> multiplication by scalar
        """
        return (A * B)

    def __imul__(A,B):
        """
        * If A,B are TTvec -> In place Hadamard product
        * If A TTvec and B scalar -> In place multiplication by scalar
        """
        if isinstance(A,TTvec) and isinstance(B,TTvec):
            # Hadamard product
            if not A.init or not B.init:
                raise NameError("TensorToolbox.TTvec.mul: TT not initialized correctly")
            
            if A.shape() != B.shape():
                raise NameError("""TensorToolbox.TTvec.mul: A and B have different shapes\n
                               \t A.shape(): %s
                               \t B.shape(): %s""" % (A.shape(),B.shape()))
            
            for i in range(len(A.TT)):
                tmpi = np.empty((A.TT[i].shape[0]*B.TT[i].shape[0], A.TT[i].shape[1], A.TT[i].shape[2]*B.TT[i].shape[2]), dtype=np.float64)
                for j in range(A.TT[i].shape[1]): tmpi[:,j,:] = np.kron(A.TT[i][:,j,:],B.TT[i][:,j,:])
                A.TT[i] = tmpi
            return A

        elif isfloat(B) and isinstance(A,TTvec):
            if not A.init:
                raise NameError("TensorToolbox.TTvec.mul: TT not initialized correctly")

            A.TT[0] *= B
            return A
    
    def __pow__(A,n):
        """
        Power by an integer
        """
        if not A.init:
            raise NameError("TensorToolbox.TTvec.pow: TT not initialized correctly")
        
        if isint(n):
            B = A.copy()
            for i in range(n-1): B *= A
            return B
        else:
            raise NameError("TensorToolbox.TTvec.pow: n must be an integer")

    def dot(self,B):
        from TensorToolbox import TTmat
        
        if not (isinstance(B,TTvec) and not isinstance(B,TTmat)):
            raise AttributeError("TensorToolbox.TTvec.dot: wrong input type")
        else:
            if not self.init or not B.init: raise NameError("TensorToolbox.TTvec.dot: TT not initialized correctly")
        
            # TT vector-vector dot product
            # Check consistency
            if self.shape() != B.shape(): raise NameError("TensorToolbox.TTvec.dot: A.shape != B.shape")
        
            Y = []
            for i,(Ai,Bi) in enumerate(zip(self.TT,B.TT)):
                Ai_rsh = np.reshape(Ai,(Ai.shape[0],1,Ai.shape[1],Ai.shape[2]))
                Ai_rsh = np.transpose(Ai_rsh,axes=(0,3,1,2))
                Ai_rsh = np.reshape(Ai_rsh,(Ai.shape[0]*Ai.shape[2],Ai.shape[1]))
                
                Bi_rsh = np.transpose(Bi,axes=(1,0,2))
                Bi_rsh = np.reshape(Bi_rsh,(Bi.shape[1],Bi.shape[0]*Bi.shape[2]))

                Yi_rsh = np.dot(Ai_rsh,Bi_rsh)
                Yi_rsh = np.reshape(Yi_rsh,(Ai.shape[0],Ai.shape[2],1,Bi.shape[0],Bi.shape[2]))
                Yi_rsh = np.transpose(Yi_rsh,axes=(0,3,2,1,4))
                Yi = np.reshape(Yi_rsh,(Ai.shape[0]*Bi.shape[0],1,Ai.shape[2]*Bi.shape[2]))
            
                if i == 0:
                    out = Yi[:,0,:]
                else:
                    out = np.dot(out, Yi[:,0,:])
        
            return out[0,0]
    
    #########################################
    # Construction
    #########################################
    def svd(self, eps, max_ranks=None):

        self.method = 'svd'
        self.eps = eps
        self.max_ranks = max_ranks
        
        """ TT-SVD """
        d = self.A.ndim
        n = self.A.shape
        delta = (eps/max(np.sqrt(d-1),1e-300)) * self.get_data_F_norm() # Truncation parameter
        
        if self.max_ranks == None:
            self.max_ranks = [ min( reduce(operator.mul,n[:i+1]), reduce(operator.mul,n[i+1:]) ) for i in range(d-1) ]
            self.max_ranks.insert(0,1)
            self.max_ranks.append(1)

        if isinstance(self.A, TensorWrapper):
            C = self.A[ tuple([ slice(None,None,None) ] * self.A.get_ndim()) ]
        else:
            C = self.A.copy()  # Temporary tensor
        G = []
        if d == 1:
            G.append( C.reshape(1,n[0],1) )
            self.TT = G
            self.init = True
            return
        r = np.empty(d,dtype=int)
        r[0] = 1
        for k in range(d-1):
            C = C.reshape(( r[k]*n[k], int(round(C.size/(r[k]*n[k]))) ))
            # Compute SVD
            (U,S,V) = npla.svd(C,full_matrices=False)

            # Compute the delta-rank of C
            CF = npla.norm(S,2)
            rk = 1
            while npla.norm(S[rk:],2) > delta and rk < self.max_ranks[k+1]: rk += 1

            if rk == self.max_ranks[k+1] and npla.norm(S[rk:],2) > delta:
                self.logger.warning("MaxRank truncation. Cores %i-%i. Required accuracy delta=%e, accuracy met: %e" % (k,k+1,delta,npla.norm(S[rk:],2)) )

            r[k+1] = rk

            # Generate new core
            G.append( U[:,:r[k+1]].reshape((r[k],n[k],r[k+1])) )
            C = scsp.spdiags([S[:rk]],[0],rk,rk).dot(V[:rk,:])
        G.append(C.reshape(r[k+1],n[k+1],1))
        self.TT = G
        self.init = True

    def ttdmrgcross(self, eps, Jinit, maxit, mv_eps, mv_maxit, kickrank=None, store_init=True, loop_detection=False):
        """ Construct a TT representation of A using TT-Density matrix renormalization group
        
        :param float eps: Frobenious tolerance of the approximation
        :param list Jinit: listo (A.ndim-1) of lists of init indices
        :param int maxit: maximum number of iterations of the ttdmrg
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol
        :param int kickrank: rank overshooting
        :param bool cross: if True it uses ttcross for the supercores. If False it uses plain SVD.
        :param bool store_init: whether to store the initial indices used (restarting from the same indices will lead to the same construction).
        :param bool loop_detection: whether to check for loops. (Never occurred that we needed it)

        .. note: eps is divided by 2 for convergence reasons.
        """
        self.ttdmrg(eps, Jinit, maxit, mv_eps, mv_maxit, 
                    kickrank=kickrank, cross=True, 
                    store_init=store_init)

    def ttdmrg(self, eps, Jinit, maxit, mv_eps, mv_maxit, kickrank=None, cross=False, store_init=True, loop_detection=False):
        """ Construct a TT representation of A using TT-Density matrix renormalization group
        
        :param float eps: Frobenious tolerance of the approximation
        :param list Jinit: listo (A.ndim-1) of lists of init indices
        :param int maxit: maximum number of iterations of the ttdmrg
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol
        :param int kickrank: rank overshooting
        :param bool cross: if True it uses ttcross for the supercores. If False it uses plain SVD.
        :param bool store_init: whether to store the initial indices used (restarting from the same indices will lead to the same construction).
        :param bool loop_detection: whether to check for loops. (Never occurred that we needed it)

        .. note: eps is divided by 2 for convergence reasons.
        """

        if cross:
            self.method = 'ttdmrgcross'
        else:
            self.method = 'ttdmrg'
        
        self.eps = eps

        d = self.A.ndim
        n = self.A.shape

        if self.TT == None or None in self.TT: # Skip if crude approx is done
            ###################################################################
            # Compute initial crude approximation of A through two sweeps 
            # (left-right and right-left of a rank-1 ttcross (set maxit=1)
            # Rank-2 can be used alternatively...
            r = 2
            MAXRANK = [ min( reduce(operator.mul,n[:i+1]), reduce(operator.mul,n[i+1:]) ) for i in range(d-1) ]
            MAXRANK.insert(0,1)
            MAXRANK.append(1)
            rs = [ min( r, MR ) for MR in MAXRANK ]

            self.Jinit = self.__check_create_Jinit(rs,Jinit)

            self.TT = [None] * d

            Js = self.Jinit

            ######################################
            # left-to-right step
            Is = [[()]]
            for k in range(d-1):
                (IT, flist, Q, QsqInv) = self.__left_right_ttcross_step(1, k, \
                                                                        rs, Is, Js, \
                                                                        mv_eps, mv_maxit)
                Is.append(IT)
            # end left-to-right step
            ###############################################

            ###############################################
            # right-to-left step
            Js = [None] * d
            Js[-1] = [()]
            for k in range(d,1,-1):
                (JT, flist, Q, QsqInv) = self.__right_left_ttcross_step(1, k, \
                                                                        rs, Is, Js, \
                                                                        mv_eps, mv_maxit)
                Js[k-2] = JT

                # Compute core
                self.TT[k-1] = np.dot(Q,QsqInv).T.reshape( (rs[k-1], n[k-1], rs[k]) )

            # Add the last core
            idx = (slice(None,None,None),) + tuple(zip(*Js[0]))
            if self.multidim_point == None:
                C = self.A[ idx ]
            else:
                C = np.asarray(self.A[ idx ].tolist())\
                    [(slice(None,None,None),slice(None,None,None)) + self.multidim_point]

            C = C.reshape(n[0], 1, rs[1])
            C = C.transpose( (1,0,2) )

            self.TT[0] = C

            # end right-to-left step
            ################################################

            # self.rounding(self.eps)
            # rs = self.ranks()

            # end compute initial crude approximation
            ###################################################################

            ###################################################################
            # Warmup: QR + right-left maxvol
            self.dmrg_Pinv = [None] * (d+1)
            self.dmrg_Pinv[0] = np.eye(1)
            self.dmrg_Pinv[d] = np.eye(1)
            r1 = np.eye(1)
            self.Js = [None] * d
            self.Js[-1] = [()]
            for k in range(d,1,-1):
                self.TT[k-1] = np.tensordot( self.TT[k-1], r1, ((2,),(0,)) )
                C = np.reshape(self.TT[k-1], ( rs[k-1], n[k-1] * rs[k] ) ).T

                [Q,R] = scla.qr(C,mode='economic')
                (J,QsqInv,it) = maxvol(Q,mv_eps,mv_maxit)

                # Retrive indices in folded tensor
                JC = [ idxfold( [n[k-1],rs[k]], idx ) for idx in J ] # First retrive idx in folded C
                JT = [  (jc[0],) + self.Js[k-1][jc[1]] for jc in JC ] # Then reconstruct the idx in the tensor
                self.Js[k-2] = JT

                r1 = np.dot(Q[J,:], R).T
                Q = np.dot(Q, QsqInv).T

                self.TT[k-1] = np.reshape(Q, ( rs[k-1], n[k-1], rs[k] ) )

                Q = np.reshape(Q, ( rs[k-1] * n[k-1], rs[k] ) )
                Q = np.dot(Q, self.dmrg_Pinv[k])
                Q = np.reshape(Q, ( rs[k-1], n[k-1] * rs[k] ) ).T
                [_,R] = scla.qr(Q,mode='economic')
                R /= max(npla.norm(R,'fro'),1e-300)
                self.dmrg_Pinv[k-1] = R

            self.TT[0] = np.tensordot( self.TT[0], r1, ((2,),(0,)) )

        it = 0
        stored_not_converged = self.dmrg_not_converged
        self.dmrg_not_converged = True

        if loop_detection:
            if self.ltor_fiber_lists == None: self.ltor_fiber_lists = []
            if self.rtol_fiber_lists == None: self.rtol_fiber_lists = []

        while it < maxit and self.dmrg_not_converged:
            it += 1
            if it == 1 and stored_not_converged != None:
                self.dmrg_not_converged = stored_not_converged
            else:
                self.dmrg_not_converged = False

            if isinstance(self.A,TensorWrapper) and self.A.twtype == 'array':
                try:
                    totsize = float(self.A.get_global_size())
                    self.logger.info("Ranks: %s - Iter: %d - Fill: %e%%" % (str(self.ranks()),it , float(self.A.get_fill_level())/totsize * 100.) )
                except OverflowError:
                    self.logger.info("Ranks: %s - Iter: %d - Fill: %d" % (str(self.ranks()),it , self.A.get_fill_level()) )
            else:
                self.logger.info("Ranks: %s - Iter: %d" % (str(self.ranks()),it) )
            
            ################################
            # Left - right sweep 
            # The following condition is used for exact restarting purposes
            if it == 1 and self.dmrg_sweep_direction == 'lr':
                drange = range(self.dmrg_sweep_it,d-1)
            elif it > 1 or (it == 1 and self.dmrg_sweep_direction == None):
                drange = range(d-1)
                self.Is = [[()]]
                self.dmrg_sweep_direction = 'lr'

            if loop_detection:
                ltor_fiber_list = [] # This might not be consistent on restart (but who cares??)

            if self.dmrg_sweep_direction == 'lr':
                for self.dmrg_sweep_it in drange:
                    ( flist, \
                          self.dmrg_Pinv, \
                          self.dmrg_not_converged ) = self.__left_right_ttdmrg_step(it, \
                                                                    self.dmrg_sweep_it, \
                                                                    self.dmrg_not_converged, \
                                                                    self.dmrg_Pinv, \
                                                                    kickrank, \
                                                                    mv_eps, \
                                                                    mv_maxit, \
                                                                    cross)
                    if loop_detection:
                        ltor_fiber_list.extend(flist)
            
            # End left-right sweep
            ################################
            
            ################################
            # Right - left sweep
            # The following condition is used for exact restarting purposes
            if it == 1 and self.dmrg_sweep_direction == 'rl':
                drange = range(self.dmrg_sweep_it,0,-1)
            else:
                drange = range(d-1,0,-1)
                self.Js = [None] * d
                self.Js[-1] = [()]
                self.dmrg_sweep_direction = 'rl'

            if loop_detection:
                rtol_fiber_list = []
            if self.dmrg_sweep_direction == 'rl':
                for self.dmrg_sweep_it in drange:
                    (flist, \
                         self.dmrg_Pinv, \
                         self.dmrg_not_converged) = self.__right_left_ttdmrg_step(it, \
                                                                    self.dmrg_sweep_it, \
                                                                    self.dmrg_not_converged, \
                                                                    self.dmrg_Pinv, \
                                                                    kickrank, \
                                                                    mv_eps, \
                                                                    mv_maxit, \
                                                                    cross)
                    if loop_detection:
                        rtol_fiber_list.extend(flist)
            
            # End right-left sweep
            ###############################
            
            if loop_detection:
                loop_detected = False
                i = 0
                while (not loop_detected) and i < len(self.rtol_fiber_lists)-1:
                    loop_detected = all(map( operator.eq, self.rtol_fiber_lists[i], rtol_fiber_list )) \
                        and all(map( operator.eq, self.ltor_fiber_lists[i], ltor_fiber_list ))
                    i += 1

                self.ltor_fiber_lists.append(ltor_fiber_list)
                self.rtol_fiber_lists.append(rtol_fiber_list)

                if loop_detected:
                    self.logger.warning( "Loop detected!" )
        
        if it >= maxit:
            raise ConvergenceError('Maximum number of iterations reached.')
        
        # Final storage
        self.store()

        self.rounding(self.eps)
        self.init = True
    
    def __left_right_ttdmrg_step(self, it, k, not_converged, 
                                 Pinv, kickrank, mv_eps, mv_maxit, 
                                 cross):
        """ Compute one step of left-right sweep of ttdmrg.

        :param int it: the actual ttcross iteration
        :param int k: the actual sweep iteration
        :param bool not_converged: convergence flag
        :param ndarray Pinv: inverse of auxiliary matrix
        :param int kickrank: rank overshooting
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol
        :param bool cross: if True it uses ttcross for the supercores. If False it uses plain SVD.

        :returns: a tuple ``(flist,P,Pinv,not_converged)`` with: the list ``flist`` containing the used fibers, the lists ``P`` and ``Pinv`` containing auxiliary matrices updated, and the boolean flag ``not_converged``.
        :rtype: tuple
        """
        
        d = self.A.ndim
        n = self.A.shape
        
        flist = []
   
        if isinstance(self.A,TensorWrapper) and self.A.twtype == 'array':
            try:
                totsize = float(self.A.get_global_size())
                self.logger.debug("Ranks: %s - Iter: %d - LR: %d - Converged: %s - Fill: %e%%" % \
                                 ( str(self.ranks()), it, k, str(not not_converged), \
                                   float(self.A.get_fill_level())/totsize * 100.) )
            except OverflowError:
                self.logger.debug("Ranks: %s - Iter: %d - LR: %d - Converged: %s - Fill: %d" % \
                                 ( str(self.ranks),it,k, str(not not_converged), \
                                   self.A.get_fill_level()) )
        else:
            self.logger.debug("Ranks: %s - Iter: %d - LR: %d " % (str(self.ranks()),it,k))

        #################################
        # Compute subtensor and supercore
        
        # Store used slices
        for ii,jj in itertools.product(self.Is[k],self.Js[k+1]):
            fiber = ii + (slice(None,None,None), slice(None,None,None)) + jj
            flist.append(fiber)

        # Create index list to get the values at once
        if k == 0:
            idx = (slice(None,None,None), slice(None,None,None)) + tuple( zip(*self.Js[k+1]) )
        else:
            it = itertools.product(self.Is[k],self.Js[k+1])
            idx = [ [] for i in range(d) ]
            for (lidx,ridx) in it:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+2+j].append(jj)
            idx[k] = slice(None,None,None)
            idx[k+1] = slice(None,None,None)
            idx = tuple(idx)

        if cross:
            # For each fiber construct a TTvec with ttcross on the filtered TensorWrapper
            C = []
            for ii,jj in itertools.product(self.Is[k],self.Js[k+1]):
                if isinstance(self.A, TensorWrapper):
                    fix_idxs = list( ii + jj )
                    fix_dims = list(range(len(ii))) + list(range(len(ii)+2,d))
                    self.A.fix_indices(fix_idxs,fix_dims)
                    TTapp = TTvec(self.A, 
                                  store_object=self.store_object,
                                  multidim_point=self.multidim_point)
                    TTapp.build(self.eps, method='ttcross')
                    self.A.release_indices()
                elif isinstance(self.A, np.ndarray):
                    A = self.A[ii + (slice(None,None,None), slice(None,None,None)) + jj]
                    TTapp = TTvec(A,
                                  store_object=self.store_object,
                                  multidim_point=self.multidim_point)
                    TTapp.build(self.eps, method='ttcross')
                # Construct TTvec
                C.append(TTapp.to_tensor())
            # Reshape values
            C = np.asarray(C)
            C = np.reshape(C, ( len(self.Is[k]), len(self.Js[k+1]), n[k], n[k+1] ) )
            C = C.transpose( (0,2,3,1) )
        else:
            # Extract values
            C = np.asarray(self.A[ idx ].tolist())
            mdim_point = () if self.multidim_point == None else self.multidim_point
            dim_point = len(mdim_point)
            dim_shp = tuple( C.shape[C.ndim-dim_point:] )
            # Reshape values
            if k == 0:              # In this case the output shape is slightly different
                C = np.reshape(C, ( n[k], n[k+1], len(self.Is[k]), len(self.Js[k+1]) ) + dim_shp )
            else:
                C = np.reshape(C, ( len(self.Is[k]), len(self.Js[k+1]), n[k], n[k+1] ) + dim_shp )
            if self.multidim_point != None:
                slices = tuple([ slice(None,None,None) ] * (C.ndim-dim_point))
                C = C[ slices + self.multidim_point ]
            if k == 0:
                C = C.transpose( (2,0,1,3) )
            else:
                C = C.transpose( (0,2,3,1) )
        
        # Obtain the corresponding QR representation (to compare C with the actual supercore)  
        C = np.tensordot( Pinv[k], C, ( (1,),(0,) ) )
        C = np.tensordot( C, Pinv[k+2], ( (3,),(0,) ) )
        
        # End compute subtensor and supercore
        #####################################

        #####################################
        # Check internal convergence
        B = np.tensordot( self.TT[k], self.TT[k+1], ((2,),(0,)) )
        B = np.tensordot( Pinv[k], B, ( (1,),(0,) ) )
        B = np.tensordot( B, Pinv[k+2], ( (3,),(0,) ) )

        # if npla.norm( (C-B).flatten(), 2 ) > self.eps * npla.norm(C.flatten(),2):
        #     not_converged = True

        # Here the Frobenious norm is taken over only the entries belonging to the real tensor,
        # not to any other extended tensor (e.g. QTT extended). This applies only for the right hand
        # side, because on the left handside the difference will be 0 for the extended part of the
        # tensor (by Linear Dependency of the extensions).
        C_real = self.get_ttdmrg_real_subtensor(C,idx)
        Diff_real = self.get_ttdmrg_real_subtensor(C-B,idx)
        if npla.norm( Diff_real.flatten(), 2 ) > self.eps * npla.norm(C_real.flatten(),2):
            not_converged = True

        ###########################################
        # Decimation step by eps-truncated SVD of C
        # Reshape
        C = C.reshape( len(self.Is[k]) * n[k], n[k+1] * len(self.Js[k+1]) )
        (U,S,V) = npla.svd( C, full_matrices=False )
        
        # Compute the eps-rank of C
        CF = npla.norm(S,2)
        rk = 1

        # Truncation options
        # a) Conservative: increase the number of retained singular values as the rank increases
        maxrank = np.max(self.ranks())
        while npla.norm(S[rk:],2) > self.eps/np.sqrt(d-1.)/maxrank * CF: rk += 1 
        # b) Conservative: increase the number of retained singular values as the rank increases
        # while npla.norm(S[rk:],2) > self.eps/np.sqrt(d-1.)/np.sqrt(rk) * CF: rk += 1 
        # c) Improved convergence
        # while npla.norm(S[rk:],2) > self.eps/(d-1.) * CF: rk += 1 
        # d) Original Oseledets
        # while npla.norm(S[rk:],2) > self.eps/np.sqrt(d-1.) * CF: rk += 1

        U = U[:,:rk]
        S = S[:rk]
        V = V[:rk,:]

        V = S[:,np.newaxis] * V
        # Kick rank (Oseledets)
        ur = npr.randn( U.shape[0], kickrank if kickrank != None else int(np.ceil(np.sqrt(maxrank))) )
        U = reort( U,ur )
        radd = U.shape[1] - rk
        if radd > 0:
            V = np.vstack( (V, np.zeros((radd, V.shape[1]))) )
        rk += radd
        
        U = np.reshape( U, (len(self.Is[k]), n[k] * rk) )
        U = npla.solve(Pinv[k], U)
        U = np.reshape( U, (len(self.Is[k]) * n[k], rk) )
        V = np.reshape( V, (rk * n[k+1], len(self.Js[k+1])) )
        V = npla.solve(Pinv[k+2].T,V.T).T
        V = np.reshape( V, (rk, n[k+1] * len(self.Js[k+1])) )
        # End decimation step by eps-truncated SVD of C
        ###############################################

        ###############################################
        # Recompute left maxvol and Pk
        [U,R] = scla.qr(U, mode='economic')
        (I,QsqInv,it) = maxvol(U, mv_eps, mv_maxit)
        r1 = U[I,:]
        U = np.dot(U, QsqInv)
        self.TT[k] = np.reshape( U, (len(self.Is[k]), n[k], rk) )
        r1 = np.dot(r1, R)
        V = np.dot(r1, V)
        self.TT[k+1] = np.reshape( V, (rk, n[k+1], len(self.Js[k+1])) )

        # Recompute Pk
        U1 = self.TT[k].copy()
        U1 = np.tensordot( Pinv[k], U1, ((1,),(0,)) )
        U1 = np.reshape( U1, (len(self.Is[k]) * n[k], rk) )
        [_,R] = scla.qr(U1,mode='economic')
        R /= max(npla.norm(R,'fro'),1e-300)
        Pinv[k+1] = R
        
        # Retrive indices in folded tensor
        IC = [ idxfold( [ len(self.Is[k]),n[k] ], idx ) for idx in I ] # First retrive idx in folded C
        IT = [ self.Is[k][ic[0]] + (ic[1],) for ic in IC ] # Then reconstruct the idx in the tensor
        self.Is.append(IT)
        # End recompute left maxvol and Pk        
        ###############################################
        
        return (flist, Pinv, not_converged)

    def __right_left_ttdmrg_step(self, it, k, not_converged, Pinv, kickrank, mv_eps, mv_maxit, 
                                 cross):
        """ Compute one step of left-right sweep of ttdmrg.

        :param int it: the actual ttcross iteration
        :param int k: the actual sweep iteration
        :param bool not_converged: convergence flag
        :param ndarray Pinv: inverse of auxiliary matrix
        :param int kickrank: rank overshooting
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol
        :param bool cross: if True it uses ttcross for the supercores. If False it uses plain SVD.

        :returns: a tuple ``(flist,P,Pinv,not_converged)`` with: the list ``flist`` containing the used fibers, the lists ``P`` and ``Pinv`` containing auxiliary matrices updated, and the boolean flag ``not_converged``.
        :rtype: tuple
        """
        
        d = self.A.ndim
        n = self.A.shape
        
        flist = []

        if isinstance(self.A,TensorWrapper) and self.A.twtype == 'array':
            try:
                totsize = float(self.A.get_global_size())
                self.logger.debug("Ranks: %s - Iter: %d - RL: %d - Converged: %s - Fill: %e%%" % \
                                 (str(self.ranks()),it,k, str(not not_converged), \
                                  float(self.A.get_fill_level())/totsize * 100.) )
            except OverflowError:
                self.logger.debug("Ranks: %s - Iter: %d - RL: %d - Converged: %s - Fill: %d" % \
                                 (str(self.ranks()),it,k, str(not not_converged), \
                                  self.A.get_fill_level()) )
        else:
            self.logger.debug("Ranks: %s - Iter: %d - RL: %d" % (str(self.ranks()),it,k))

        
        #################################
        # Compute subtensor and supercore
        
        # Store used slices
        for ii in self.Is[k-1]:
            for jj in self.Js[k]:
                fiber = ii + (slice(None,None,None), slice(None,None,None)) + jj
                flist.append(fiber)

        # Create index list to get the values at once
        if k == d-1:
            idx = tuple( zip(*self.Is[k-1]) ) + (slice(None,None,None), slice(None,None,None))
        else:
            it = itertools.product(self.Is[k-1],self.Js[k])
            idx = [ [] for i in range(d) ]
            for (lidx,ridx) in it:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+2+j].append(jj)
            idx[k-1] = slice(None,None,None)
            idx[k] = slice(None,None,None)
            idx = tuple(idx)

        if cross:
            # For each fiber construct a TTvec with ttcross on the filtered TensorWrapper
            C = []
            for ii,jj in itertools.product(self.Is[k-1],self.Js[k]):
                if isinstance(self.A, TensorWrapper):
                    fix_idxs = list( ii + jj )
                    fix_dims = list(range(len(ii))) + list(range(len(ii)+2,d))
                    self.A.fix_indices(fix_idxs,fix_dims)
                    TTapp = TTvec(self.A, 
                                  store_object=self.store_object,
                                  multidim_point=self.multidim_point)
                    TTapp.build(self.eps, method='ttcross')
                    self.A.release_indices()
                elif isinstance(self.A, np.ndarray):
                    A = self.A[ii + (slice(None,None,None), slice(None,None,None)) + jj]
                    TTapp = TTvec(A, 
                                  store_object=self.store_object,
                                  multidim_point=self.multidim_point)
                    TTapp.build(self.eps, method='ttcross')
                # Construct TTvec
                C.append(TTapp.to_tensor())
            # Reshape values
            C = np.asarray(C)
            C = np.reshape(C, ( len(self.Is[k-1]), len(self.Js[k]), n[k-1], n[k] ) )
            C = C.transpose( (0,2,3,1) )
        else:
            # Extract values        
            C = np.asarray(self.A[ idx ].tolist())
            mdim_point = () if self.multidim_point == None else self.multidim_point
            dim_point = len(mdim_point)
            dim_shp = tuple( C.shape[C.ndim-dim_point:] )
            # Reshape values
            if k == 1:              # In this case the output shape is slightly different
                C = np.reshape( C, (n[k-1], n[k], len(self.Is[k-1]), len(self.Js[k])) + dim_shp )
            else:
                C = np.reshape( C, (len(self.Is[k-1]), len(self.Js[k]), n[k-1], n[k]) + dim_shp )
            if self.multidim_point != None:
                slices = tuple([ slice(None,None,None) ] * (C.ndim-dim_point))
                C = C[ slices + self.multidim_point ]
            if k == 1:
                C = C.transpose( (2,0,1,3) )
            else:
                C = C.transpose( (0,2,3,1) )

        # Obtain the corresponding QR representation (to compare C with the actual supercore)
        C = np.tensordot( Pinv[k-1], C, ( (1,),(0,) ) )
        C = np.tensordot( C, Pinv[k+1], ( (3,),(0,) ) )
        
        # End compute subtensor and supercore
        #####################################

        #####################################
        # Check internal convergence
        B = np.tensordot( self.TT[k-1], self.TT[k], ((2,),(0,)) )
        B = np.tensordot( Pinv[k-1], B, ( (1,),(0,) ) )
        B = np.tensordot( B, Pinv[k+1], ( (3,),(0,) ) )

        # if npla.norm( (C-B).flatten(), 2 ) > self.eps * npla.norm(C.flatten(),2):
        #     not_converged = True        

        # Here the Frobenious norm is taken over only the entries belonging to the real tensor,
        # not to any other extended tensor (e.g. QTT extended). This applies only for the right hand
        # side, because on the left handside the difference will be 0 for the extended part of the
        # tensor (by Linear Dependency of the extensions).
        C_real = self.get_ttdmrg_real_subtensor(C,idx)
        Diff_real = self.get_ttdmrg_real_subtensor(C-B,idx)
        if npla.norm( Diff_real.flatten(), 2 ) > self.eps * npla.norm(C_real.flatten(),2):
            not_converged = True
        
        ###########################################
        # Decimation step by eps-truncated SVD of C
        # Reshape
        C = C.reshape( len(self.Is[k-1]) * n[k-1], n[k] * len(self.Js[k]) )
        (U,S,V) = npla.svd( C, full_matrices=False )
        
        # Compute the eps-rank of C
        CF = npla.norm(S,2)
        rk = 1

        # Truncation options
        # a) Conservative: increase the number of retained singular values as the rank increases
        maxrank = np.max(self.ranks())
        while npla.norm(S[rk:],2) > self.eps/np.sqrt(d-1.)/maxrank * CF: rk += 1 
        # b) Conservative: increase the number of retained singular values as the rank increases
        # while npla.norm(S[rk:],2) > self.eps/np.sqrt(d-1.)/np.sqrt(rk) * CF: rk += 1 
        # c) Improved convergence
        # while npla.norm(S[rk:],2) > self.eps/(d-1.) * CF: rk += 1 
        # d) Original Oseledets
        # while npla.norm(S[rk:],2) > self.eps/np.sqrt(d-1.)/10. * CF: rk += 1

        U = U[:,:rk]
        S = S[:rk]
        V = V[:rk,:]

        U = U * S[np.newaxis,:]
        # Kick rank (Oseledets)
        vr = npr.randn( kickrank if kickrank != None else int(np.ceil(np.sqrt(maxrank))), V.shape[1] )
        V = reort( V.T, vr.T )
        V = V.T
        radd = V.shape[0] - rk
        if radd > 0:
            U = np.hstack( (U, np.zeros((U.shape[0],radd))) )
        rk += radd

        U = np.reshape( U, (len(self.Is[k-1]), n[k-1] * rk) )
        U = npla.solve(Pinv[k-1], U)
        U = np.reshape( U, (len(self.Is[k-1]) * n[k-1], rk) )
        V = np.reshape( V, (rk * n[k], len(self.Js[k])) )
        V = npla.solve(Pinv[k+1].T,V.T).T
        V = np.reshape( V, (rk, n[k] * len(self.Js[k])) )
        
        # End decimation step by eps-truncated SVD of C
        ###############################################
        
        ###############################################
        # Recompute right maxvol and Pk
        V = np.reshape( V, ( rk, n[k] * len(self.Js[k]) ) ).T
        [V, R] = scla.qr(V, mode='economic')
        (J,QsqInv,it) = maxvol(V, mv_eps, mv_maxit)
        r1 = V[J,:]
        V = np.dot(V,QsqInv)
        V = np.reshape(V, (n[k], len(self.Js[k]), rk) )
        self.TT[k] = np.transpose(V, (2,0,1))
        r1 = np.dot(r1,R).T
        U = np.dot(U,r1)
        self.TT[k-1] = np.reshape( U, (len(self.Is[k-1]), n[k-1], rk) )
        
        # Recalculate P[k]
        V = self.TT[k].copy()
        V = np.tensordot( V, Pinv[k+1], ((2,),(0,)) )
        V = np.reshape( V, ( rk, n[k] * len(self.Js[k]) ) ).T
        [_,R] = scla.qr(V, mode='economic')
        R /= max(npla.norm(R,'fro'),1e-300)
        Pinv[k] = R

        # Retrive indices in folded tensor
        JC = [ idxfold( [ n[k], len(self.Js[k]) ], idx ) for idx in J ] # First retrive idx in folded C
        JT = [ (jc[0],) + self.Js[k][jc[1]] for jc in JC ] # Then reconstruct the idx in the tensor
        self.Js[k-1] = JT
        # End recompute left maxvol and Pk        
        ###############################################
        
        return (flist, Pinv, not_converged)

    def ttcross(self, eps, rs, Jinit, delta, maxit, mv_eps, mv_maxit, fix_rank = False):
        """ Construct a TT representation of A using TT cross. This routine manage the outer loops for incremental ttcross or passes everything to ttcross if rs are specified.

        :param float eps: tolerance with which to perform the TT-rounding and check the rank accuracy
        :param list rs: list of upper ranks (A.ndim)
        :param list Jinit: list (A.ndim-1) of lists of init indices
        :param float delta: TT-cross accuracy
        :param int maxit: maximum number of iterations for ttcross
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol
        :param bool fix_rank: Whether the rank is allowed to increase
        """

        self.method = 'ttcross'
        self.eps = eps

        INCREMENTAL = (rs == None) or (not fix_rank)
        if not INCREMENTAL:
            try:
                self.rs = rs
                self.TT = self.inner_ttcross( rs, Jinit, delta, maxit, mv_eps, mv_maxit)
                self.init = True
            except Exception as e:
                # If the size of the tensor A is small and can be fully stored, store it with also the last Js/Is used in ttcross
                if self.A.size <= 2*1e7/8: # Limit set to approx 20mb of file size
                    import cPickle as pkl
                    FILENAME = "TensorToolbox.log"
                    outdic = dict({'A': self.A[tuple([slice(None,None,None) for i in range(d)])],
                                   'Js': self.Js,
                                   'Is': self.Is,
                                   'Jinit': self.Jinit})
                    outfile = open(FILENAME,'wb')
                    pkl.dump(outdic,outfile)
                    outfile.close()
                    self.logger.warning( "TTvec: Log file stored" )
                raise e

        else:

            d = self.A.ndim
            n = self.A.shape
            
            MAXRANK = [ min( reduce(operator.mul,n[:i+1]), reduce(operator.mul,n[i+1:]) ) for i in range(d-1) ]
            MAXRANK.insert(0,1)
            MAXRANK.append(1)
            
            PassedRanks = False
            Js = Jinit

            if rs == None:
                r = 2
                self.rs = [min( r, MAXRANK[i+1]) for i in range(d-1)]
                self.rs.insert(0,1)
                self.rs.append(1)
            else:
                self.rs = rs

            counter = 0
            notpassidxs = None
            while not PassedRanks:
                store_init_Js = (counter == 0)
                counter += 1
                try:
                    Gnew = self.inner_ttcross(self.rs,Js,delta,maxit,mv_eps,mv_maxit,store_init=store_init_Js)
                except TTcrossLoopError as e:
                    # If a loop is detected, then ask for a rank increase
                    self.logger.warning( "Loop detected! Increasing ranks." )
                    PassedRanks = False
                    if notpassidxs == None: notpassidxs = range(d-1) # If this is the first run, mark all as not passed
                    notpassidxs_old = notpassidxs[:]
                    notpassidxs = []
                    for i in range(1,d):
                        if ((i-1) in notpassidxs_old) and (not self.rs[i] == MAXRANK[i]):
                            self.rs[i] += 1
                            notpassidxs.append(i-1)
                except ConvergenceError as e:
                    # If the ttcross reaches the maximum number of function iterations,
                    # increase the ranks like for TTcrossLoopError
                    self.logger.warning( "ttcross not converged, maximum num. of iterations reached. Increasing ranks" )
                    PassedRanks = False
                    if notpassidxs == None: notpassidxs = range(d-1) # If this is the first run, mark all as not passed
                    notpassidxs_old = notpassidxs[:]
                    notpassidxs = []
                    for i in range(1,d):
                        if ((i-1) in notpassidxs_old) and (not self.rs[i] == MAXRANK[i]):
                            self.rs[i] += 1
                            notpassidxs.append(i-1)
                except:
                    raise
                else:
                    TTapprox = TTvec( Gnew )
                    TTapprox.build()

                    crossRanks = TTapprox.ranks()
                    roundRanks = TTapprox.rounding(eps).ranks()
                    PassedRanks = True
                    notpassidxs = []
                    for i in range(1,d):
                        if not (crossRanks[i] > roundRanks[i] or roundRanks[i] == MAXRANK[i]):
                            notpassidxs.append(i-1) # i-1 because Js[0] is referred already to the first core
                            self.rs[i] += 1
                            PassedRanks = False 
                
                Js_old = self.Js
                Js = []
                if not PassedRanks:
                    
                    # Get last indices used copy them
                    for i in range(len(Js_old)):
                        Js.append( Js_old[i][:] )
                        
                    # Get last indices and augment them with one entry (possibly already computed)
                    for i in notpassidxs:
                        newidx = Js[i][0]
                        # Try first with already computed indices looking to all the history of Js
                        jtmp = len(self.Js_last)-1
                        while (newidx in Js[i]) and jtmp >= 0 :
                            Js_tmp = self.Js_last[jtmp]
                            Js_diff = set(Js_tmp[i]) - set(Js[i])
                            if len( Js_diff ) > 0:
                                # pick one randomly in the diff
                                newidx = random.sample(Js_diff,1)[0]
                            jtmp -= 1
                        
                        # Pick randomly if none was found in previously computed sequence
                        while newidx in Js[i]: newidx = tuple( [random.choice(range(n[j])) for j in range(i+1,d)] )
                        Js[i].append(newidx)
            
            self.TT = Gnew
            self.init = True

    def inner_ttcross(self, rs, Jinit, delta, maxit, mv_eps, mv_maxit, store_init=True):
        """ Construct a TT representation of A using TT cross

        :param list rs: list of upper ranks (A.ndim)
        :param list Jinit: list (A.ndim-1) of lists of init indices
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol
        :param bool store_init: indicates whether to store the init ``Jinit`` (see ``outer_ttcross``)
        """

        self.Js_last = []
        
        d = self.A.ndim
        n = self.A.shape
        
        if len(rs) != d+1:
            raise AttributeError("List of guessed ranks must be of length A.ndim")
        
        if rs[0] != 1 or rs[-1] != 1:
            raise ValueError("r[0] and r[-1] must be 1")
        
        Jinit = self.__check_create_Jinit(rs,Jinit)
            
        if store_init: self.Jinit = Jinit
        
        Gold = [ np.zeros((rs[k],n[k],rs[k+1])) for k in range(d) ]
        Gnew = [ npr.random((rs[k],n[k],rs[k+1])) for k in range(d) ]
        
        # Normalize Gnew so that we enter the first loop and check for inf or 0. value
        tt_Gnew = TTvec(Gnew)
        tt_Gnew.build()
        fro_new = mla.norm(tt_Gnew,'fro')
        if fro_new == np.inf or fro_new == 0.:
            raise OverflowError("TensorToolbox.TensorTrainVec: The Frobenious norm of the init TT is: %f" % fro_new )
        tt_Gnew = tt_Gnew * (1./fro_new)
        Gnew = tt_Gnew.TT
        
        Js = Jinit

        self.ltor_fiber_lists = []
        self.rtol_fiber_lists = []
        it = 0
        store_counter = 0
        while it < maxit and mla.norm(TTvec(Gold).build()-TTvec(Gnew).build(),'fro') > delta * mla.norm(TTvec(Gnew).build(), 'fro'):
            it += 1
            if isinstance(self.A,TensorWrapper) and self.A.twtype == 'array':
                try:
                    totsize = float(self.A.get_global_size())
                    self.logger.info("Ranks: %s - Iter: %d - Fill: %e%%" % (str(rs),it , float(self.A.get_fill_level())/totsize * 100.) )
                except OverflowError:
                    self.logger.info("Ranks: %s - Iter: %d - Fill: %d" % (str(rs),it , self.A.get_fill_level()) )
            else:
                self.logger.info("Ranks: %s - Iter: %d" % (str(rs),it) )

            Gold = Gnew
            Gnew = [None for i in range(d)]
            
            ######################################
            # left-to-right step
            ltor_fiber_list = []
            Is = [[()]]
            for k in range(d-1):
                (IT, flist, Q, QsqInv) = self.__left_right_ttcross_step(it, k, \
                                                                        rs, Is, Js, \
                                                                        mv_eps, mv_maxit)
                ltor_fiber_list.extend( flist )
                Is.append(IT)

            # end left-to-right step
            ###############################################

            # Store last Js indices (for restarting purposes)
            self.Js_last.append([ J[:] for J in Js ])
            
            ###############################################
            # right-to-left step
            rtol_fiber_list = []
            Js = [None for i in range(d)]
            Js[-1] = [()]
            for k in range(d,1,-1):
                (JT, flist, Q, QsqInv) = self.__right_left_ttcross_step(it, k, \
                                                                        rs, Is, Js, \
                                                                        mv_eps, mv_maxit)
                rtol_fiber_list.extend( flist )
                Js[k-2] = JT
                
                # Compute core
                Gnew[k-1] = np.dot(Q,QsqInv).T.reshape( (rs[k-1], n[k-1], rs[k]) )

            # Add the last core
            idx = (slice(None,None,None),) + tuple(zip(*Js[0]))
            if self.multidim_point == None:
                C = self.A[ idx ]
            else:
                C = np.asarray(self.A[ idx ].tolist())\
                    [(slice(None,None,None),slice(None,None,None)) + self.multidim_point]
            
            C = C.reshape(n[0], 1, rs[1])
            C = C.transpose( (1,0,2) )
            
            Gnew[0] = C 
            
            # end right-to-left step
            ################################################
            
            # Check that none of the previous iteration has already used the same fibers 
            # (this indicates the presence of a loop). 
            # If this is the case apply a random perturbation on one of the fibers
            loop_detected = False
            i = 0
            while (not loop_detected) and i < len(self.rtol_fiber_lists)-1:
                loop_detected = all(map( operator.eq, self.rtol_fiber_lists[i], rtol_fiber_list )) \
                    and all(map( operator.eq, self.ltor_fiber_lists[i], ltor_fiber_list ))
                i += 1
            
            if loop_detected:# and rtol_loop_detected:
                # If loop is detected, then an exception is raised
                # and the outer_ttcross will increase the rank
                self.Js = Js
                self.Is = Is
                raise TTcrossLoopError('Loop detected!')
            else:
                self.ltor_fiber_lists.append(ltor_fiber_list)
                self.rtol_fiber_lists.append(rtol_fiber_list)

        if it >= maxit:
            self.Js = Js
            self.Is = Is
            raise ConvergenceError('Maximum number of iterations reached.')

        if mla.norm(TTvec(Gold).build()-TTvec(Gnew).build(),'fro') > delta * mla.norm(TTvec(Gnew).build(), 'fro'):
            self.Js = Js
            self.Is = Is
            raise ConvergenceError('Low Rank Approximation algorithm did not converge.')
        
        self.Js = Js
        self.Is = Is

        # Final storage
        self.store()
        
        return Gnew
        
    def __check_create_Jinit( self, rs, Jinit):
        """ Check whether the input Jinit agrees with the ranks in ``rs`` or generates a random list of Jinit for the ranks ``rs``.

        :param list rs: list of upper ranks (A.ndim)
        :param list Jinit: list (A.ndim-1) of lists of init indices

        :returns: list (A.ndim-1) of lists of init indices
        """

        d = self.A.ndim
        n = self.A.shape
        MAXRANK = [ min( reduce(operator.mul,n[:i+1]), reduce(operator.mul,n[i+1:]) ) for i in range(d-1) ]
        MAXRANK.insert(0,1)
        MAXRANK.append(1)

        if Jinit == None: Jinit = [None for i in range(d)]

        if len(Jinit) != d:
            raise AttributeError("List of init indexes must be of length A.ndim-1")
        for k_Js in range(len(Jinit)-1):
            if Jinit[k_Js] == None:
                if rs[k_Js+1] > MAXRANK[k_Js+1]:
                    raise ValueError("Ranks selected exceed the dimension of the tensor")
                
                # Lazy selection of indices... this can be done better.
                Jinit[k_Js] = []
                for i in range(rs[k_Js+1]):
                    newidx = tuple( [ random.choice(range(n[j])) for j in range(k_Js+1,d) ] )
                    while newidx in Jinit[k_Js]:
                        newidx = tuple( [ random.choice(range(n[j])) for j in range(k_Js+1,d) ] )
                    
                    Jinit[k_Js].append(newidx)

            if len(Jinit[k_Js]) != rs[k_Js+1]:
                raise ValueError("Lenght of init right sequence must agree with the upper rank values #1")
            for idx in Jinit[k_Js]:
                if len(idx) != d - (k_Js + 1):
                    raise ValueError("Lenght of init right sequence must agree with the upper rank values #2")
        
        if Jinit[-1] == None: Jinit[-1] = [()]
        
        return Jinit
        
    def __left_right_ttcross_step(self, it, k, rs, Is, Js, mv_eps, mv_maxit):
        """ Compute one step of left-right sweep of ttcross.

        :param int it: the actual ttcross iteration
        :param int k: the actual sweep iteration
        :param list rs: list of upper ranks (A.ndim)
        :param list Is: list (A.ndim-1) of lists of left indices
        :param list Js: list (A.ndim-1) of lists of right indices
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol

        :returns: tuple containing: ``(IT,flist,Q,QsqInv)``, the list of new indices, the used fibers, the Q matrix and the inverse of the maxvol submatrix
        :rtype: tuple
        """

        d = self.A.ndim
        n = self.A.shape

        flist = []
   
        if isinstance(self.A,TensorWrapper) and self.A.twtype == 'array':
            try:
                totsize = float(self.A.get_global_size())
                self.logger.debug("Ranks: %s - Iter: %d - LR: %d - Fill: %e%%" % (str(rs),it,k, float(self.A.get_fill_level())/totsize * 100.) )
            except OverflowError:
                self.logger.debug("Ranks: %s - Iter: %d - LR: %d - Fill: %d" % (str(rs),it,k, self.A.get_fill_level()) )
        else:
            self.logger.debug("Ranks: %s - Iter: %d - LR: %d" % (str(rs),it,k))

        # Extract fibers
        for i in range(rs[k]):
            for j in range(rs[k+1]):
                fiber = Is[k][i] + (slice(None,None,None),) + Js[k][j]
                flist.append(fiber)

        if k == 0:      # Is[k] will be empty
            idx = (slice(None,None,None),) + tuple(zip(*Js[k]))
        else:
            it = itertools.product(Is[k],Js[k])
            idx = [ [] for i in range(d) ]
            for (lidx,ridx) in it:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+1+j].append(jj)
            idx[k] = slice(None,None,None)
            idx = tuple(idx)

        if self.multidim_point == None:
            C = self.A[ idx ]
        else:
            C = np.asarray(self.A[ idx ].tolist())\
                [(slice(None,None,None),slice(None,None,None)) + self.multidim_point]

        if k == 0:
            C = C.reshape(n[k], rs[k], rs[k+1])
            C = C.transpose( (1,0,2) )
        else:
            C = C.reshape(rs[k], rs[k+1], n[k])
            C = C.transpose( (0,2,1) )

        C = C.reshape(( rs[k] * n[k], rs[k+1] ))

        # Compute QR decomposition
        (Q,R) = scla.qr(C,mode='economic')
        # Maxvol
        (I,QsqInv,it) = maxvol(Q,mv_eps,mv_maxit)

        # Retrive indices in folded tensor
        IC = [ idxfold( [rs[k],n[k]], idx ) for idx in I ] # First retrive idx in folded C
        IT = [ Is[k][ic[0]] + (ic[1],) for ic in IC ] # Then reconstruct the idx in the tensor

        return (IT, flist, Q, QsqInv)
    
    def __right_left_ttcross_step(self, it, k, rs, Is, Js, mv_eps, mv_maxit):
        """ Compute one step of right-left sweep of ttcross.

        :param int it: the actual ttcross iteration
        :param int k: the actual sweep iteration
        :param list rs: list of upper ranks (A.ndim)
        :param list Is: list (A.ndim-1) of lists of left indices
        :param list Js: list (A.ndim-1) of lists of right indices
        :param float mv_eps: MaxVol accuracy
        :param int mv_maxit: maximum number of iterations for MaxVol

        :returns: tuple containing: ``(JT,flist,Q,QsqInv)``, the list of new indices, the used fibers, the Q matrix and the inverse of the maxvol submatrix
        :rtype: tuple
        """

        d = self.A.ndim
        n = self.A.shape

        flist = []

        if isinstance(self.A,TensorWrapper) and self.A.twtype == 'array':
            try:
                totsize = float(self.A.get_global_size())
                self.logger.debug("Ranks: %s - Iter: %d - RL: %d - Fill: %e%%" % (str(rs),it,k, float(self.A.get_fill_level())/totsize * 100.) )
            except OverflowError:
                self.logger.debug("Ranks: %s - Iter: %d - RL: %d - Fill: %d" % (str(rs),it,k, self.A.get_fill_level()) )
        else:
            self.logger.debug("Ranks: %s - Iter: %d - RL: %d" % (str(rs),it,k))

        # Extract fibers
        for i in range(rs[k-1]):
            for j in range(rs[k]):
                fiber = Is[k-1][i] + (slice(None,None,None),) + Js[k-1][j]
                flist.append(fiber)

        if k == d:      # Is[k] will be empty
            idx = tuple(zip(*Is[k-1])) + (slice(None,None,None),)
        else:
            it = itertools.product(Is[k-1],Js[k-1])
            idx = [ [] for i in range(d) ]
            for (lidx,ridx) in it:
                for j,jj in enumerate(lidx): idx[j].append(jj)
                for j,jj in enumerate(ridx): idx[len(lidx)+1+j].append(jj)
            idx[k-1] = slice(None,None,None)
            idx = tuple(idx)

        if self.multidim_point == None:
            C = self.A[ idx ]
        else:
            C = np.asarray(self.A[ idx ].tolist())\
                [(slice(None,None,None),slice(None,None,None)) + self.multidim_point]

        C = C.reshape(rs[k-1], rs[k], n[k-1])
        C = C.transpose( (0,2,1) )

        C = C.reshape( (rs[k-1],n[k-1]*rs[k]) ).T

        # Compute QR decomposition
        (Q,R) = scla.qr(C,mode='economic')
        # Maxvol
        (J,QsqInv,it) = maxvol(Q,mv_eps,mv_maxit)

        # Retrive indices in folded tensor
        JC = [ idxfold( [n[k-1],rs[k]], idx ) for idx in J ] # First retrive idx in folded C
        JT = [  (jc[0],) + Js[k-1][jc[1]] for jc in JC ] # Then reconstruct the idx in the tensor

        return (JT, flist, Q, QsqInv)
        
    def kron(self,A):
        if not self.init: raise NameError("TensorToolbox.TTvec.extend: TT not initialized correctly")
        if not isinstance(A,TTvec): raise NameError("TensorToolbox.TTvec.extend: input tensor is not in TT format")
        if not A.init: raise NameError("TensorToolbox.TTvec.extend: input tensor is not initialized correctly")
        self.TT.extend(A.TT)

    def rounding2(self,eps,max_ranks=None):
        """ TT-rounding """
        d = len(self.TT)
        n = self.shape()
        r = self.ranks()

        if max_ranks == None:
            max_ranks = [ min( reduce(operator.mul,n[:i+1]), reduce(operator.mul,n[i+1:]) ) for i in range(d-1) ]
            max_ranks.insert(0,1)
            max_ranks.append(1)
        
        nrm = np.zeros(d,dtype=np.float64)
        core0 = self.TT[0]
        for i in range(d-1):
            core0 = np.reshape(core0,(r[i]*n[i],r[i+1]))
            (core0,ru) = scla.qr(core0,mode='economic')
            nrm[i+1] = npla.norm(ru,'fro')
            ru /= max(nrm[i+1],1e-300)
            core1 = self.TT[i+1].reshape((r[i+1],n[i+1]*r[i+2]))
            core1 = np.dot(ru,core1)
            r[i+1] = core0.shape[1]
            self.TT[i] = np.reshape(core0,(r[i],n[i],r[i+1]))
            self.TT[i+1] = np.reshape(core1,(r[i+1],n[i+1],r[i+2]))
            core0 = core1
        
        ep = eps/np.sqrt(d-1)
        core0 = self.TT[d-1]
        for i in range(d-1,0,-1):
            core1 = self.TT[i-1].reshape((r[i-1]*n[i-1],r[i]))
            core0 = np.reshape(core0,(r[i],n[i]*r[i+1]))
            (U,S,V) = scla.svd(core0,full_matrices=False)
            r1 = self.__round_chop(S,npla.norm(S,2)*ep)            # Truncate
            if r1 > max_ranks[i]:
                r1 = max_ranks[i]
                self.logger.warning("MaxRank truncation. Cores %i-%i. Required accuracy delta=%e, accuracy met: %e" % (i,i-1,ep,npla.norm(S[r1:],2)) )
            U = U[:,:r1]
            S = S[:r1]
            V = V[:r1,:]
            U = np.dot(U,np.diag(S))
            r[i] = r1
            core1 = np.dot(core1,U)
            self.TT[i] = np.reshape(V,(r[i],n[i],r[i+1]))
            self.TT[i-1] = np.reshape(core1,(r[i-1],n[i-1],r[i]))
            core0 = core1.copy()
        
        pp = self.TT[0]
        nrm[0] = np.sqrt(np.sum(pp**2.))
        if np.abs(nrm[0]) > np.spacing(1):
            self.TT[0] /= nrm[0]
        # Ivan's trick to redistribute norms
        nrm0 = np.sum(np.log(np.abs(nrm)))
        nrm0 = nrm0/float(d)
        nrm0 = np.exp(nrm0)
        if nrm0 > np.spacing(1):
            for i in range(d-1):
                nrm[i+1] = nrm[i+1]*nrm[i]/nrm0
                nrm[i] = nrm0
        # Redistribute norm
        for i in range(d): self.TT[i] *= nrm[i]
        
        return self

    def __round_chop(self,S,eps):
        ss = np.cumsum(S[::-1]**2.)
        return len(S) - next(i for i,s in enumerate(ss) if s > eps**2. or i == len(S)-1)

    def rounding(self,eps,max_ranks=None):
        """ TT-rounding """
        d = len(self.TT)
        ns = self.shape()

        if max_ranks == None:
            max_ranks = [ min( reduce(operator.mul,ns[:i+1]), reduce(operator.mul,ns[i+1:]) ) for i in range(d-1) ]
            max_ranks.insert(0,1)
            max_ranks.append(1)

        # OBS: The truncation parameter could be computed during the right to left orthogonalization?
        delta = eps/np.sqrt(d-1) # Truncation parameter

        # Right to left orthogonalization
        nrm = np.zeros(d,dtype=np.float64)
        for k in range(d-1,0,-1):
            # Computation of rq components
            alphakm1 = self.TT[k].shape[0]
            betak = self.TT[k].shape[2]
            Gk = np.reshape(self.TT[k],(alphakm1,self.TT[k].shape[1]*betak))
            (R,Q) = scla.rq(Gk,mode='economic')
            nrm[k-1] = npla.norm(R,'fro')
            betakm1 = R.shape[1]
            R /= max(nrm[k-1],1e-300)
            self.TT[k] = np.reshape(Q,(betakm1,self.TT[k].shape[1],betak))
            # 3-mode product G[k-1] x_3 R 
            C = self.TT[k-1].reshape((self.TT[k-1].shape[0]*self.TT[k-1].shape[1],self.TT[k-1].shape[2]))
            self.TT[k-1] = np.dot(C,R).reshape((self.TT[k-1].shape[0],self.TT[k-1].shape[1],betakm1))
            # gc.collect()
        
        # Compression
        r = [1]
        for k in range(d-1):
            C = self.TT[k].copy().reshape((self.TT[k].shape[0]*self.TT[k].shape[1],self.TT[k].shape[2]))
            # Compute SVD
            (U,S,V) = npla.svd(C,full_matrices=False)
            # Truncate SVD
            rk = 0
            rk = self.__round_chop(S,npla.norm(S,2)*delta)            # Truncate
            if rk > max_ranks[k+1]:
                rk = max_ranks[k+1]
                self.logger.warning("MaxRank truncation. Cores %i-%i. Required accuracy delta=%e, accuracy met: %e" % (k,k+1,delta,npla.norm(S[rk:],2)) )

            r.append( rk )
            self.TT[k] = U[:,:r[k+1]].reshape((r[k],self.TT[k].shape[1],r[k+1]))
            # Update next core with 1-mode product.
            SV = np.tile(S[:r[k+1]],(V.shape[1],1)).T * V[:r[k+1],:]
            C = self.TT[k+1].reshape((self.TT[k+1].shape[0], self.TT[k+1].shape[1]*self.TT[k+1].shape[2]))
            self.TT[k+1] = np.dot(SV,C).reshape((r[k+1],ns[k+1],self.TT[k+1].shape[2]))
            # gc.collect()

        nrm[d-1] = npla.norm(self.TT[d-1].flatten(),2)
        self.TT[d-1] /= max(nrm[d-1],1e-300)

        # Oseledets Trick here!
        nrm0 = np.sum(np.log(np.abs(nrm)))
        nrm0 = nrm0/float(d)
        nrm0 = np.exp(nrm0)
        if np.abs(nrm0) > np.spacing(1):
            # Construct normalization of norm
            for i in range(d-1,0,-1):
                nrm[i-1] = nrm[i-1]*nrm[i]/nrm0
                nrm[i] = nrm0

        # Redistribute the norm
        for i in range(d-1,-1,-1): self.TT[i] *= nrm[i]
        
        return self

    def interpolate(self,Ms=None,eps=1e-8,is_sparse=None):
        """ Interpolates the values of the TTvec at arbitrary points, using the interpolation matrices ``Ms``.
        
        :param list Ms: list of interpolation matrices for each dimension. Ms[i].shape[1] == self.shape()[i]
        :param float eps: tolerance with which to perform the rounding after interpolation
        :param list is_sparse: is_sparse[i] is a bool indicating whether Ms[i] is sparse or not. If 'None' all matrices are non sparse
        
        :returns: TTvec interpolation
        :rtype: TTvec

        >>> from DABISpectralToolbox import DABISpectral1D as S1D
        >>> Ms = [ S1D.LinearInterpolationMatrix(X[i],XI[i]) for i in range(d) ]
        >>> is_sparse = [True]*d
        >>> TTapproxI = TTapprox.interpolate(Ms,eps=1e-8,is_sparse=is_sparse)
        
        """
        from TensorToolbox import TTmat

        if not self.init: raise NameError("TensorToolbox.TTvec.interpolate: TT not initialized correctly")
        
        if len(Ms) != self.ndim():
            raise AttributeError("The length of Ms and the dimension of the TTvec must be the same!")

        d = len(Ms)
        for i in range(d):
            if Ms[i].shape[1] != self.shape()[i]:
                raise AttributeError("The condition  Ms[i].shape[1] == self.shape()[i] must hold.")                        

        if isinstance(Ms,list) and np.all( [ isinstance(Ms[i],scsp.csr_matrix) for i in range(len(Ms)) ] ):
            sparse_ranks = [1] * (d+1)
            nrows = [Ms[i].shape[0] for i in range(d)]
            ncols = [Ms[i].shape[1] for i in range(d)]
            TT_MND = TTmat(Ms, nrows, ncols, sparse_ranks=sparse_ranks).build()
        else:
            if is_sparse == None: is_sparse = [False]*len(Ms)

            # Construct the interpolating TTmat
            TT_MND = TTmat(Ms[0].flatten(),nrows=Ms[0].shape[0],ncols=Ms[0].shape[1],is_sparse=[ is_sparse[0] ]).build()
            for M,s in zip(Ms[1:],is_sparse[1:]):
                TT_MND.kron( TTmat(M.flatten(),nrows=M.shape[0],ncols=M.shape[1],is_sparse=[s]).build() )
        
        # Perform interpolation
        return mla.dot(TT_MND,self).rounding(eps)
    
    def project(self, Vs=None, Ws=None, eps=1e-8,is_sparse=None):
        """ Project the TTvec onto a set of basis provided, using the Generalized Vandermonde matrices ``Vs`` and weights ``Ws``.
        
        :param list Vs: list of generalized Vandermonde matrices for each dimension. Ms[i].shape[1] == self.shape()[i]
        :param list Ws: list of weights for each dimension. Ws[i].shape[0] == self.shape()[i]
        :param float eps: tolerance with which to perform the rounding after interpolation
        :param list is_sparse: is_sparse[i] is a bool indicating whether Ms[i] is sparse or not. If 'None' all matrices are non sparse
        
        :returns: TTvec containting the Fourier coefficients
        :rtype: TTvec

        >>> from DABISpectralToolbox import DABISpectral1D as S1D
        >>> P = S1D.Poly1D(S1D.JACOBI,(0,0))
        >>> x,w = S1D.Quadrature(10,S1D.GAUSS)
        >>> X = [x]*d
        >>> W = [w]*d
        >>> # Compute here the TTapprox at points X
        >>> TTapprox = TTvec(....)
        >>> # Project
        >>> Vs = [ P.GradVandermonde1D(x,10,0,norm=False) ] * d
        >>> is_sparse = [False]*d
        >>> TTfourier = TTapprox.project(Vs,W,eps=1e-8,is_sparse=is_sparse)
        """
        from TensorToolbox import TTmat
        
        if not self.init: raise NameError("TensorToolbox.TTvec.project: TT not initialized correctly")
        
        if len(Vs) != len(Ws) or len(Ws) != self.ndim():
            raise AttributeError("The length of Vs, Ms and the dimension of the TTvec must be the same!")

        d = len(Vs)
        for i in range(d):
            if Vs[i].shape[1] != Ws[i].shape[0] or Ws[i].shape[0] != self.shape()[i]:
                raise AttributeError("The condition  Vs[i].shape[1] == Ws[i].shape[0] == self.shape()[i] must hold.")                

        if is_sparse == None: is_sparse = [False]*d
        
        # Prepare matrices
        VV = [ Vs[i].T * np.tile(Ws[i],(Vs[i].shape[0],1)) for i in range(d) ]

        TT_MND = TTmat(VV[0].flatten(),nrows=VV[0].shape[0],ncols=VV[0].shape[1],is_sparse=[ is_sparse[0] ]).build()
        for V,s in zip(VV[1:],is_sparse[1:]):
            TT_MND.kron( TTmat(V.flatten(),nrows=V.shape[0],ncols=V.shape[1],is_sparse=[s]).build() )
        
        # Perform projection
        return mla.dot(TT_MND,self).rounding(eps)

##########################################################
# Constructors of frequently used tensor vectors
##########################################################

def randvec(d,N):
    """ Returns the rank-1 multidimensional random vector in Tensor Train format
    
    Args:
       d (int): number of dimensions
       N (int or list): If int then uniform sizes are used for all the dimensions, if list of int then len(N) == d and each dimension will use different size
    
    Returns:
       TTvec The rank-1 multidim random vector in Tensor Train format
    """
    from TensorToolbox.core import Candecomp

    if isint(N):
        N = [ N for i in range(d) ]
    CPtmp = [npr.random(N[i]).reshape((1,N[i])) + 0.5 for i in range(d)]
    CP_rand = Candecomp(CPtmp)
    TT_rand = TTvec(CP_rand)
    TT_rand.build()
    return TT_rand

def zerosvec(d,N):
    """ Returns the rank-1 multidimensional vector of zeros in Tensor Train format
    
    Args:
       d (int): number of dimensions
       N (int or list): If int then uniform sizes are used for all the dimensions, if list of int then len(N) == d and each dimension will use different size
    
    Returns:
       TTvec The rank-1 multidim vector of zeros in Tensor Train format
    """
    from TensorToolbox.core import Candecomp

    if isint(N):
        N = [ N for i in range(d) ]
    CPtmp = [np.zeros((1,N[i])) for i in range(d)]
    CP_zeros = Candecomp(CPtmp)
    TT_zeros = TTvec(CP_zeros)
    TT_zeros.build()
    return TT_zeros

def onesvec(d,N):
    """ Returns the rank-1 multidimensional vector of ones in Tensor Train format
    
    Args:
       d (int): number of dimensions
       N (int or list): If int then uniform sizes are used for all the dimensions, if list of int then len(N) == d and each dimension will use different size
    
    Returns:
       TTvec The rank-1 multidim vector of ones in Tensor Train format
    """
    from TensorToolbox.core import Candecomp

    if isint(N):
        N = [ N for i in range(d) ]
    CPtmp = [np.ones((1,N[i])) for i in range(d)]
    CP_ones = Candecomp(CPtmp)
    TT_ones = TTvec(CP_rand)
    TT_ones.build()
    return TT_rand
