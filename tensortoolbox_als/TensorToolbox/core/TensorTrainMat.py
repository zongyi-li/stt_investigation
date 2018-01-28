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

__all__ = ['TTmat','eye','randmat']

import logging
import numpy as np
from scipy import sparse as scsp

from TensorToolbox.core import TTvec, WTTvec, mat_to_tt_idxs, isint, isfloat

class TTmat(TTvec):
    """ Constructor of multidimensional matrix in Tensor Train format

    :param Candecomp,ndarray,TT A: Available input formats are Candecomp, full tensor in numpy.ndarray, Tensor Train structure (list of cores), list of sparse matrices of sizes (r_{i-1}*r_{i}*nrows x ncols) (used for fast dot product - limited support for other functionalities)
    :param list,int nrows: If int then the row size will be the same in all dimensions, if list then len(nrows) == len(self.TT) (numer of cores) and row size will change for each dimension.
    :param list,int ncols: If int then the column size will be the same in all dimensions, if list then len(ncols) == len(self.TT) (numer of cores) and column size will change for each dimension.
    :param bool is_sparse: [default == False] if True it uses sparsity to accelerate some computations
    :param list sparse_ranks: [default==None] mandatory argument when A is a list of sparse matrices. It contains integers listing the TT-ranks of the matrix.

    .. note:: the method __getitem__ is not overwritten, thus the indices used to access the tensor refer to the flatten versions of the matrices composing the matrix tensor.

    """

    logger = logging.getLogger(__name__)

    def __init__(self,A,nrows,ncols,is_sparse=None,sparse_ranks=None,
                 store_location="",store_object=None,store_freq=1, store_overwrite=False):
        
        super(TTmat,self).__init__(A, 
                                   store_location=store_location,
                                   store_object=store_object,
                                   store_freq=store_freq,
                                   store_overwrite=store_overwrite)

        ############################
        # List of attributes
        self.nrows = None
        self.ncols = None
        self.is_sparse = None
        self.sparse_TT = None
        self.sparse_only = None
        self.sparse_ranks = None
        
        self.serialize_list.extend( ['nrows','ncols','is_sparse','sparse_TT','sparse_only','sparse_ranks'] )
        # End list of attributes
        ############################
        
        self.nrows = nrows
        self.ncols = ncols
        self.is_sparse = is_sparse
        self.sparse_ranks = sparse_ranks

    def __getstate__(self):
        return super(TTmat,self).__getstate__()
    
    def __setstate__(self,state):
        super(TTmat,self).__setstate__( state )

    def build(self, eps=1e-10, method='svd', rs=None, fix_rank=False, Jinit=None, delta=1e-4, maxit=100, mv_eps=1e-6, mv_maxit=100, max_ranks=None, kickrank=2):
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
        
        nrows = self.nrows
        ncols = self.ncols
        is_sparse = self.is_sparse
        
        self.nrows = []
        self.ncols = []
        self.is_sparse = []
        self.sparse_TT = []
        
        if isinstance(self.A,list) and np.all( [ (isinstance(self.A[i],scsp.csr_matrix) or
                                                  isinstance(self.A[i],scsp.csc_matrix) or
                                                  isinstance(self.A[i],scsp.dia_matrix)) for i in range(len(self.A)) ] ):
            if self.sparse_ranks == None:
                raise AttributeError("The parameter sparse_ranks must be defined for only-sparse initialization")
            
            if len(self.sparse_ranks) - 1 != len(self.A):
                raise AttributeError("The condition len(sparse_ranks)-1 == len(A) must hold.")
            
            self.sparse_only = True
            self.sparse_TT = self.A

            if isint(nrows) and isin(ncols):
                d = len(self.sparse_TT)
                self.nrows = [nrows for i in range(d)]
                self.ncols = [ncols for i in range(d)]
            elif isinstance(nrows,list) and isinstance(ncols,list):
                self.nrows = nrows
                self.ncols = ncols
            else:
                self.init = False
                raise TypeError("tensor.TTmat.__init__: types of nrows and ncols are inconsistent.")

            self.is_sparse = [True] * len(self.sparse_TT)
            self.TT = [None] * len(self.sparse_TT)
            
            self.init = True

        elif isinstance(self.A,list) and np.any( [ (isinstance(self.A[i],scsp.csr_matrix) or
                                                    isinstance(self.A[i],scsp.csc_matrix) or
                                                    isinstance(self.A[i],scsp.dia_matrix)) for i in range(len(self.A)) ] ):
            raise TypeError("Mixed sparse/full initialization not implemented yet")
        else:
            self.sparse_only = False
            super(TTmat,self).build( eps=eps, method=method, rs=rs, fix_rank=fix_rank, Jinit=Jinit, delta=delta, maxit=maxit, mv_eps=mv_eps, mv_maxit=mv_maxit, max_ranks=max_ranks, kickrank=kickrank )
            if isint(nrows) and isint(ncols):
                d = len(self.TT)
                self.nrows = [nrows for i in range(d)]
                self.ncols = [ncols for i in range(d)]
            elif isinstance(nrows,list) and isinstance(ncols,list):
                self.nrows = nrows
                self.ncols = ncols
            else:
                self.init = False
                raise TypeError("tensor.TTmat.__init__: types of nrows and ncols are inconsistent.")
        
            if is_sparse == None:
                self.is_sparse = [False]*len(self.TT)
            elif len(is_sparse) != len(self.TT):
                raise TypeError("tensor.TTmat.__init__: parameter is_sparse must be of length d=A.ndims.")
            else: self.is_sparse = is_sparse

            for i,(is_sp,Ai) in enumerate(zip(self.is_sparse,self.TT)):
                if is_sp:
                    Ai_rsh = np.reshape(Ai,(Ai.shape[0],self.nrows[i],self.ncols[i],Ai.shape[2]))
                    Ai_rsh = np.transpose(Ai_rsh,axes=(0,3,1,2))
                    Ai_rsh = np.reshape(Ai_rsh,(Ai.shape[0]*Ai.shape[2]*self.nrows[i],self.ncols[i]))
                    self.sparse_TT.append( scsp.csr_matrix(Ai_rsh) )
                else:
                    self.sparse_TT.append( None )
        
            # Check that all the mode sizes are equal to rows*cols
            for i,msize in enumerate(self.shape()):
                if msize != self.nrows[i]*self.ncols[i]:
                    self.init = False
                    raise NameError("tensor.TTmat.__init__: the %d-th TT mode size must be equal to nrows[%d]*ncols[%d]" % (i,i,i))
        
        return self

    def copy(self):
        newTT = []
        for TTi in self.TT: newTT.append(TTi.copy())
        return TTmat(newTT,self.nrows,self.ncols,is_sparse=self.is_sparse).build()

    def kron(self,A):
        if not self.init: raise NameError("tensor.TTmat.extend: TT not initialized correctly")
        if not isinstance(A,TTmat): raise NameError("tensor.TTmat.extend: input tensor is not in TT format")
        if not A.init: raise NameError("tensor.TTmat.extend: input tensor is not initialized correctly")
        self.TT.extend(A.TT)
        self.nrows.extend(A.nrows)
        self.ncols.extend(A.ncols)
        self.is_sparse.extend(A.is_sparse)
        self.sparse_TT.extend(A.sparse_TT)

    def ranks(self):
        if self.sparse_only:
            return self.sparse_ranks
        else:
            return super(TTmat,self).ranks()

    def __getitem__(self,idxs):
        """ 
        Return the item at a certain index. 
        The index is formed as follows:
           idxs = (rowidxs,colidxs) = ((i_1,...,i_d),(j_1,...,j_d))
        """
        if not self.init: raise NameError("tensor.TTmat.__getitem__: TTmat not initialized correctly")
        return TTvec.__getitem__(self,mat_to_tt_idxs(idxs[0],idxs[1],self.nrows,self.ncols))

    def __imul__(A,B):
        if isinstance(A,TTmat) and isinstance(B,TTmat):
            # Check dim consistency
            if A.nrows != B.nrows or A.ncols != B.ncols:
                raise NameError("tensor.TTmat.mul: Matrices of non consistent dimensions")
        
        return TTvec.__imul__(A,B)

    def dot(self,B):
        if isinstance(B,TTvec) and not isinstance(B,TTmat):
            if not self.init or not B.init: raise NameError("TensorToolbox.TTmat.dot: TT not initialized correctly")

            # TT matrix-vector dot product
            # Check consistency
            if (self.sparse_only and len(self.sparse_TT) != len(B.TT)) or (not self.sparse_only and len(self.TT) != len(B.TT)):
                raise NameError("TensorToolbox.TTmat.dot: A and B must have the same number of cores")

            for bsize,Acols in zip(B.shape(),self.ncols):
                if bsize != Acols:
                    raise NameError("TensorToolbox.TTmat.dot: Matrix and Vector mode dimensions are not consistent")

            Y = []
            for i,(Ai,Bi,is_sparse,sp_A) in enumerate(zip(self.TT,B.TT,self.is_sparse,self.sparse_TT)):
                Bi_rsh = np.transpose(Bi,axes=(1,0,2))
                Bi_rsh = np.reshape(Bi_rsh,(Bi.shape[1],Bi.shape[0]*Bi.shape[2]))

                if is_sparse:
                    Yi_rsh = sp_A.dot(Bi_rsh)
                else:
                    Ai_rsh = np.reshape(Ai,(Ai.shape[0],self.nrows[i],self.ncols[i],Ai.shape[2]))
                    Ai_rsh = np.transpose(Ai_rsh,axes=(0,3,1,2))
                    Ai_rsh = np.reshape(Ai_rsh,(Ai.shape[0]*Ai.shape[2]*self.nrows[i],self.ncols[i]))
                    Yi_rsh = np.dot(Ai_rsh,Bi_rsh)

                Ai0 = self.ranks()[i]
                Ai2 = self.ranks()[i+1]
                Yi_rsh = np.reshape(Yi_rsh,(Ai0,Ai2,self.nrows[i],Bi.shape[0],Bi.shape[2]))
                Yi_rsh = np.transpose(Yi_rsh,axes=(0,3,2,1,4))
                Yi = np.reshape(Yi_rsh,(Ai0*Bi.shape[0],self.nrows[i],Ai2*Bi.shape[2]))

                Y.append(Yi)
            
            if isinstance(B,WTTvec):
                return WTTvec(Y,B.sqrtW).build()
            else:
                return TTvec(Y).build()

        elif isinstance(B,TTmat):
            if not self.init or not B.init: raise NameError("TensorToolbox.TTmat.dot: TT not initialized correctly")

            # TT matrix-matrix dot product
            # Check consistency
            if len(self.TT) != len(B.TT):
                raise NameError("TensorToolbox.TTmat.dot: A and B must have the same number of cores")

            for Brows,Acols in zip(B.nrows,self.ncols):
                if Brows != Acols:
                    raise NameError("TensorToolbox.TTmat.dot: Matrices mode dimensions are not consistent")

            Y = []
            for i,(Ai,Bi) in enumerate(zip(self.TT,B.TT)):
                Ai_rsh = np.reshape(Ai,(Ai.shape[0],self.nrows[i],self.ncols[i],Ai.shape[2]))
                Ai_rsh = np.transpose(Ai_rsh,axes=(0,3,1,2))
                Ai_rsh = np.reshape(Ai_rsh,(Ai.shape[0]*Ai.shape[2]*self.nrows[i],self.ncols[i]))

                Bi_rsh = np.reshape(Bi,(Bi.shape[0],B.nrows[i],B.ncols[i],Bi.shape[2]))
                Bi_rsh = np.transpose(Bi_rsh,axes=(1,0,3,2))
                Bi_rsh = np.reshape(Bi_rsh,(B.nrows[i],Bi.shape[0]*Bi.shape[2]*B.ncols[i]))

                Yi_rsh = np.dot(Ai_rsh,Bi_rsh)

                Yi_rsh = np.reshape(Yi_rsh,(Ai.shape[0],Ai.shape[2],self.nrows[i],Bi.shape[0],Bi.shape[2],B.ncols[i]))
                Yi_rsh = np.transpose(Yi_rsh,axes=(0,3,2,5,1,4))
                Yi = np.reshape(Yi_rsh,(Ai.shape[0]*Bi.shape[0],self.nrows[i]*B.ncols[i],Ai.shape[2]*Bi.shape[2]))

                Y.append(Yi)

            return TTmat(Y,self.nrows,B.ncols).build()
        elif isinstance(B,np.ndarray):
            if not self.init: raise NameError("TensorToolbox.multilinalg.dot: TT not initialized correctly")

            # matrix-vector dot product with TTmat and full vector
            # Check consistency
            if len(self.shape()) != B.ndim:
                raise NameError("TensorToolbox.multilinalg.dot: A and B must have the same number of dimensions")
            for bsize,Acols in zip(B.shape,self.ncols):
                if bsize != Acols:
                    raise NameError("TensorToolbox.multilinalg.dot: Matrix and Vector mode dimensions are not consistent")

            Bshape = B.shape
            Y = np.reshape(B,( (1,) + Bshape ))
            Yshape = Y.shape

            for k, Ak in enumerate(self.TT):
                # Note: Ak(alpha_{k-1},(i_k,j_k),alpha_{k})
                # Reshape it to Ak((alpha_{k},i_k),(alpha_{k-1},j_k))
                alpha0 = Ak.shape[0]
                alpha1 = Ak.shape[2]
                Ak_rsh = np.reshape(Ak, (alpha0,self.nrows[k],self.ncols[k],alpha1))
                Ak_rsh = np.transpose(Ak_rsh, axes=(3,1,0,2)) # Ak(alpha_{k},i_k,alpha_{k},j_k)
                Ak_rsh = np.reshape(Ak_rsh, (alpha1 * self.nrows[k], alpha0 * self.ncols[k]))

                # Reshape Y to Y((alpha_{k-1},j_k),(i_1,..,i_k-1,j_k+1,..,j_d))
                Y = np.transpose(Y,axes=(0,k+1)+tuple(range(1,k+1))+tuple(range(k+2,len(Bshape)+1)))
                Y = np.reshape( Y, (alpha0 * Yshape[k+1],
                                    int(round(np.prod(Yshape[1:k+1]) * np.prod(Yshape[k+2:])))) )

                # Dot product
                Y = np.dot(Ak_rsh,Y)

                # Reshape Y
                Y = np.reshape(Y, (alpha1, self.nrows[k]) + Yshape[1:k+1] + Yshape[k+2:])
                Y = np.transpose(Y, axes=(0,) + tuple(range(2,k+2)) + (1,) + tuple(range(k+2,len(Bshape)+1)) )
                Yshape = Y.shape

            if Y.shape[0] != 1: raise NameError("TensorToolbox.multilinalg.dot: Last core dimenstion error")

            Y = np.reshape(Y,Y.shape[1:])

            return Y

        else:
            raise AttributeError("TensorToolbox.TTmat.dot: wrong input type")

##########################################################
# Constructors of frequently used tensor matrices
##########################################################

# Random rank-1 matrix
def randmat(d,nrows,ncols):
    """ Returns the rank-1 multidimensional random matrix in Tensor Train format
    
    Args:
       d (int): number of dimensions
       nrows (int or list): If int then uniform sizes are used for all the dimensions, if list of int then len(nrows) == d and each dimension will use different size
       ncols (int or list): If int then uniform sizes are used for all the dimensions, if list of int then len(ncols) == d and each dimension will use different size
    
    Returns:
       TTmat The rank-1 multidim random matrix in Tensor Train format
    """
    import numpy.random as npr
    from TensorToolbox.core import Candecomp

    if isint(nrows): nrows = [ nrows for i in range(d) ]
    if isint(ncols): ncols = [ ncols for i in range(d) ]
    CPtmp = [npr.random(nrows[i]*ncols[i]).reshape((1,nrows[i]*ncols[i])) + 0.5 for i in range(d)]
    CP_rand = Candecomp(CPtmp)
    TT_rand = TTmat(CP_rand,nrows,ncols).build()
    return TT_rand

# Identity tensor
def eye(d,N):
    """ Returns the multidimensional identity operator in Tensor Train format
    
    Args:
       d (int): number of dimensions
       N (int or list): If int then uniform sizes are used for all the dimensions, if list of int then len(N) == d and each dimension will use different size
    
    Returns:
       TTmat The multidim identity matrix in Tensor Train format

    Note:
       TODO: improve construction avoiding passage through Candecomp
    """
    from TensorToolbox.core import Candecomp
    if isint(N):
        If = np.eye(N).flatten().reshape((1,N**2))
        CPtmp = [If for i in range(d)]
    elif isinstance(N, list):
        CPtmp = [np.eye(N[i]).flatten().reshape((1,N[i]**2)) for i in range(d)]
    
    CP_id = Candecomp(CPtmp)
    TT_id = TTmat(CP_id,nrows=N,ncols=N,is_sparse=[True]*d).build()
    return TT_id

