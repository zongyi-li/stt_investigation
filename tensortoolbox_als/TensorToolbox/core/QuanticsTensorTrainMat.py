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


__all__ = ['QTTmat']

import logging
import operator
import numpy as np
from TensorToolbox.core import QTTvec
from TensorToolbox.core import TTmat
from TensorToolbox.core import idxfold, idxunfold, isint

class QTTmat(TTmat):
    """ Constructor of multidimensional matrix in Quantics Tensor Train format :cite:`Khoromskij2010,Khoromskij2011`.

    :param ndarray,TT A: Available input formats are full tensor in numpy.ndarray, Tensor Train structure (list of cores). If input is ndarray, then it must be in mattensor format (see aux.py)
    :param int base: folding base for QTT representation
    :param int nrows: If int then the row size will be the same in all dimensions, if list then len(nrows) == len(self.TT) (numer of cores) and row size will change for each dimension.
    :param int ncols: If int then the column size will be the same in all dimensions, if list then len(ncols) == len(self.TT) (numer of cores) and column size will change for each dimension.

    """

    logger = logging.getLogger(__name__)
    
    def __init__(self,A,base,nrows,ncols,is_sparse=None,
                 store_location="",store_object=None,store_freq=1, store_overwrite=False):
       
        super(QTTmat,self).__init__(A, base, base, is_sparse=is_sparse,
                                   store_location=store_location,
                                   store_object=store_object,
                                   store_freq=store_freq,
                                   store_overwrite=store_overwrite)
        
        ######################################
        # List of attributes
        self.base = None
        self.basemat = None
        self.L = None
        self.full_nrows = [] # Real sizes of the tensor matrices
        self.full_ncols = [] # Real sizes of the tensor matrices
        
        self.serialize_list.extend( ['base', 'basemat', 'L', 'full_nrows', 'full_ncols'] )
        # End list of attributes
        ######################################
 
        self.full_nrows = nrows
        self.full_ncols = ncols
        self.base = base
        self.basemat = base**2

    def __getstate__(self):
        return super(QTTmat,self).__getstate__()
    
    def __setstate__(self,state):
        super(QTTmat,self).__setstate__( state )
        
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
        
        nrows = self.full_nrows
        ncols = self.full_ncols
        
        if isint(nrows) and isint(ncols):
            nrows = [nrows]
            ncols = [ncols]

        if len(nrows) != len(ncols): raise NameError("TensorToolbox.QTTmat.__init__: len(nrows)!=len(ncols)")

        self.full_nrows = nrows
        self.full_ncols = ncols

        if isinstance(self.A,np.ndarray):
            
            for i,sizedim in enumerate(self.A.shape):
                if sizedim != self.full_nrows[i] * self.full_ncols[i]:
                    raise NameError("TensorToolbox.QTTmat.__init__: Array dimension not consistent with nrows and ncols")
                if np.remainder(np.log(sizedim)/np.log(self.basemat),1.0) > np.spacing(1):
                    raise NameError("TensorToolbox.QTTmat.__init__: base is not a valid base of A.size")
            
            self.L = int( np.log(self.A.size)/np.log(self.basemat) )
            
            # Prepare interleaved idxs (wtf!!!)
            Ls = [ int( np.log(self.full_nrows[i]*self.full_ncols[i])/np.log(self.basemat) ) for i in range(self.ndims()) ]
            idxs = []
            for j in range(self.ndims()):
                offset = np.sum(2 * Ls[:j],dtype=int)
                for i in range(Ls[j]):
                    idxs.append(offset + i)
                    idxs.append(offset + Ls[j]+i)

            # Fold, re-order and reshape
            self.A = np.reshape(self.A,[self.base for i in range(2*self.L)])
            self.A = np.transpose(self.A,axes=idxs)
            self.A = np.reshape(self.A,[self.basemat for i in range(self.L)])
            
            super(QTTmat,self).build( eps=eps, method=method, rs=rs, fix_rank=fix_rank, Jinit=Jinit, delta=delta, maxit=maxit, mv_eps=mv_eps, mv_maxit=mv_maxit, max_ranks=max_ranks, kickrank=kickrank )
            
        elif isinstance(self.A,list):
            
            super(QTTmat,self).build( eps=eps, method=method, rs=rs, fix_rank=fix_rank, Jinit=Jinit, delta=delta, maxit=maxit, mv_eps=mv_eps, mv_maxit=mv_maxit, max_ranks=max_ranks, kickrank=kickrank )
            
            # Check that unfolded nrows,ncols are consistent with the dimension of A
            if np.prod(self.shape()) != np.prod(self.full_nrows)*np.prod(self.full_ncols):
                self.init = False
                raise NameError("TensorToolbox.QTTmat.__init__: unfolded nrows,ncols not consistent with shape of A")
            for nrow,ncol in zip(self.full_nrows,self.full_ncols):
                if np.remainder(np.log(nrow*ncol)/np.log(self.basemat),1.0) > np.spacing(1):
                    self.init = False
                    raise NameError("TensorToolbox.QTTmat.__init__: base is not a valid base for the selected nrows,ncols")
            
            self.L = len(self.shape()) 
            
        return self
    
    def ndims(self):
        """ Return the number of dimensions of the tensor space
        """
        return len(self.full_nrows)

    def get_full_nrows(self):
        """ Returns the number of rows of the unfolded matrices 
        """
        return self.full_nrows

    def get_full_ncols(self):
        """ Returns the number of cols of the unfolded matrices
        """
        return self.full_ncols

    def get_nrows(self):
        """ Returns the number of rows of the folded matrices
        """
        return self.nrows

    def get_ncols(self):
        """ Returns the number of cols of the folded matrices
        """
        return self.nrows
    
    def copy(self):
        newTT = []
        for TTi in self.TT: newTT.append(TTi.copy())
        return QTTmat(newTT,self.base,nrows=list(self.get_full_nrows()),ncols=list(self.get_full_ncols())).build()
    
    def __getitem__(self,idxs):
        """ Get item function
        :param tuple,int idxs: ((i_1,..,i_d),(j_1,..,j_d)) with respect to the unfolded mode sizes

        :returns: item at that position
        """
        if not self.init: raise NameError("TensorToolbox.QTTmat.__getitem__: QTT not initialized correctly")
        
        # Check for out of bounds
        if any(map(operator.ge,idxs[0],self.get_full_nrows())) or any(map(operator.ge,idxs[1],self.get_full_ncols())):
            raise NameError("TensorToolbox.QTTmat.__getitem__: Index out of bounds")
        
        # Compute the index of the folding representation from the unfolded index
        return TTmat.__getitem__(self,
                                 ( idxfold( self.get_nrows(), idxunfold(self.get_full_nrows(), idxs[0])),
                                   idxfold( self.get_ncols(), idxunfold(self.get_full_ncols(), idxs[1]))) )

    def kron(self,A):
        if not self.init: raise NameError("TensorToolbox.QTTmat.kron: TT not initialized correctly")
        # Additional tests wrt the extend function of TTvec
        if not isinstance(A,QTTmat): raise NameError("TensorToolbox.QTTmat.kron: A is not of QTTmat type")
        if not A.init: raise NameError("TensorToolbox.QTTmat.kron: input tensor is not initialized correctly")
        if self.base != A.base: raise NameError("TensorToolbox.QTTmat.kron: kron product for QTT vectors is allowed only for the same bases")
        
        super(QTTmat,self).kron(A)

        self.full_nrows.extend(A.get_full_nrows())
        self.full_ncols.extend(A.get_full_ncols())
        self.L = len(self.shape())
            
    def dot(self,B):
        out = super(QTTmat,self).dot(B)
        
        if isinstance(B,QTTmat):
            out = QTTmat( out.TT, base=self.base, 
                          nrows=self.get_full_nrows(), ncols=B.get_full_ncols() )
            out.build()
        elif isinstance(B,QTTvec):
            out = QTTvec( out.TT, global_shape=self.get_full_nrows() )
            out.build()
        elif not isinstance(B,np.ndarray):
            raise AttributeError("TensorToolbox.QTTmat.dot: wrong input format")

        return out
