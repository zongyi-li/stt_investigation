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

__all__ = ['WTTvec']

import sys
import warnings
import logging 

import numpy as np
from scipy import sparse as scsp

from TensorToolbox.core import TTvec, Candecomp, TensorWrapper

class WTTvec(TTvec):
    """ Constructor of multidimensional tensor in Weighted Tensor Train format
    
    :param Candecomp,ndarray,TT,TensorWrapper A: Available input formats are Candecomp, full tensor in numpy.ndarray, Tensor Train structure (list of cores), or a Tensor Wrapper.
    :param list W: list of 1-dimensional ndarray containing the weights for each dimension.
    :param string store_location: Store computed values during construction on the specified file path. The stored values are ttcross_Jinit and the values used in the TensorWrapper. This permits a restart from already computed values. If empty string nothing is done. (method=='ttcross')
    :param string store_object: Object to be stored (default are the tensor wrapper and ttcross_Jinit)
    :param int store_freq: storage frequency. ``store_freq==1`` stores intermediate values at every iteration. The program stores data every ``store_freq`` internal iterations. If ``store_object`` is a SpectralTensorTrain, then ``store_freq`` determines the number of seconds every which to store values.
    :param int multidim_point: If the object A returns a multidimensional array, then this can be used to define which point to apply ttcross to.
    """

    logger = logging.getLogger(__name__)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    def __init__(self,A, W,
                 store_location="",store_object=None,store_freq=1, store_overwrite=False, 
                 multidim_point=None):
        super(WTTvec,self).__init__(A, 
                                    store_location=store_location,
                                    store_object=store_object,
                                    store_freq=store_freq, 
                                    store_overwrite=store_overwrite, 
                                    multidim_point=multidim_point)
        self._init(W)

    def _init(self, W):
        ##############################
        # List of attributes
        self.sqrtW = None           # It will contain the square root of the weights
        self.weights_flag = None    # Indicates whether the weights are being applied to the TT
        self.data_weights_flag = None # Indicates whether the weights are being applied to the data
        self.TTmat_sqrtW = None       # TT matrix of weights.
        self.TTmat_sqrtW_inv = None       # TT matrix of the reciprocal of the weights.

        self.serialize_list.extend( ['W', 'sqrtW','weights_flag','data_weights_flag',
                                     'TTmat_sqrtW','TTmat_sqrtW_inv'] )
        self.subserialize_list.extend( [] )   # Not serialized
        # End list of attributes
        ###############################

        self.W = W
        self.sqrtW = [ np.sqrt(wi) for wi in W ]
        self.weights_flag = False
        self.data_weights_flag = False
    
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
        self._build_preprocess()
        super(WTTvec,self).build(eps=eps, method=method, rs=rs, 
                                 fix_rank=fix_rank, Jinit=Jinit, 
                                 delta=delta, maxit=maxit, mv_eps=mv_eps,
                                 mv_maxit=mv_maxit,
                                 max_ranks=max_ranks, 
                                 kickrank=kickrank )
        self._build_postprocess()
        return self

    def _build_preprocess(self):
        from TensorToolbox.core import TTmat
        # Construct the sparse diagonal matrices of weights
        mats = []
        mats_inv = []
        nrows = []
        for wi in self.sqrtW:
            sh = wi.shape[0]
            mats.append( scsp.dia_matrix(( wi, np.array([0]) ), shape=(sh,sh)) )
            mats_inv.append( scsp.dia_matrix(( 1./wi, np.array([0]) ),
                                             shape=(sh,sh)) )
            nrows.append( sh )
        
        self.TTmat_sqrtW = TTmat( mats, nrows, nrows, sparse_ranks=[1]*(len(self.sqrtW)+1) )
        self.TTmat_sqrtW.build()
        self.TTmat_sqrtW_inv = TTmat( mats_inv, nrows, nrows,
                                      sparse_ranks=[1]*(len(self.sqrtW)+1) )
        self.TTmat_sqrtW_inv.build()
        if isinstance(self.A,np.ndarray) or isinstance(self.A,TensorWrapper):
            # We are building an approximation of the weighted data
            self.apply_weights_on_data()
            self.weights_flag = True

    def _build_postprocess(self):
        if isinstance(self.A,np.ndarray) or isinstance(self.A,TensorWrapper):
            # This remove the weights from the TT, returning the wanted approx.
            self.remove_weights_from_data()
            # self.remove_weights() 

    ######################################
    # Weighting routines

    def is_weighted(self):
        return self.weights_flag

    def is_data_weighted(self):
        return self.data_weights_flag

    def apply_weights_on_data(self):
        """ Apply the weights on the input data A
        
        .. note: The end user should not need to use this method unless he knows what he's doing.
        """
        if isinstance(self.A,Candecomp):
            raise NameError("Weights cannot be applied to a Candecomp type.")
        elif isinstance(self.A,np.ndarray):
            # Use numpy broadcasting
            for i,wi in zip(range(self.A.ndim),self.sqrtW):
                sh = [1]*self.A.ndim
                sh[i] = wi.shape[0]
                self.A *= wi.reshape( sh )
        elif isinstance(self.A,TensorWrapper):
            self.A.set_weights(self.sqrtW)
            self.A.set_active_weights(True)
        elif isinstance(self.A,list):
            raise NameError("Weights cannot be applied to a list type. Use ``TensorToolbox.WTTvec.apply_weights instead.")
        else:
            raise ValueError("Input type not allowed")
        self.data_weights_flag = True

    def remove_weights_from_data(self):
        """ Removes the weights from the input data A
        
        .. note: The end user should not need to use this method unless he knows what he's doing.
        """
        if isinstance(self.A,Candecomp):
            raise NameError("Weights cannot be removed from a Candecomp type.")
        elif isinstance(self.A,np.ndarray):
            # Use numpy broadcasting
            for i,wi in zip(range(self.A.ndim),self.sqrtW):
                sh = [1]*self.A.ndim
                sh[i] = wi.shape[0]
                self.A /= wi.reshape( sh )
        elif isinstance(self.A,TensorWrapper):
            self.A.set_active_weights(False)
        elif isinstance(self.A,list):
            raise NameError("Weights cannot be removed from a list type. Use ``TensorToolbox.WTTvec.apply_weights instead.")
        else:
            raise ValueError("Input type not allowed")
        self.data_weights_flag = False

    def apply_weights(self):
        self.TT = self.TTmat_sqrtW.dot(self).TT
        self.weights_flag = True
    
    def remove_weights(self):
        self.TT = self.TTmat_sqrtW_inv.dot(self).TT
        self.weights_flag = False
    
    # End weighting routines
    ########################################
    
    def rounding(self,eps,max_ranks=None):
        weights_applied = False
        if not self.is_weighted():
            self.apply_weights()
            weights_applied = True
        
        out = super(WTTvec,self).rounding(eps,max_ranks=max_ranks)
        
        if weights_applied:
            self.remove_weights()
        
        return out
        
    def rounding2(self,eps,max_ranks=None):
        weights_applied = False
        if not self.is_weighted():
            self.apply_weights()
            weights_applied = True
        
        out = super(WTTvec,self).rounding2(eps,max_ranks=max_ranks)
        
        if weights_applied:
            self.remove_weights()
        
        return out
