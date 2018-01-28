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

__all__ = ['STT', 'SQTT', 'SWTT', 'SWQTT',
           'LINEAR_INTERPOLATION', 'LAGRANGE_INTERPOLATION', 'PROJECTION',
           'AVAIL_SURROGATE_TYPES']

import logging
import os.path
import warnings
import time
import copy
import random

import numpy as np
import numpy.linalg as npla

from scipy import linalg as scla
from scipy import sparse as scsp
from scipy.sparse import linalg as spla

from TensorToolbox.core import storable_object, isint, isfloat
from TensorToolbox.core import TensorWrapper, TTvec, QTTvec, WTTvec, WQTTvec 
from TensorToolbox import multilinalg as mla

from SpectralToolbox import Spectral1D as S1D

LINEAR_INTERPOLATION = 'LinearInterpolation'
LAGRANGE_INTERPOLATION = 'LagrangeInterpolation'
PROJECTION    = 'Projection'
AVAIL_SURROGATE_TYPES = [ LINEAR_INTERPOLATION, LAGRANGE_INTERPOLATION, PROJECTION ]

class STT(storable_object):
    """ Constructor of the Spectral Tensor Train approximation :cite:`Bigoni2015`. Given a function ``f(x,theta,params):(Is, It) -> R``
    with ``dim(Is)=n`` and ``dim(It)=d``, construct an approximation of ``g(theta,params): It -> h_t(Is)``. For example ``Is`` could be the discretization of a spatial dimension, and ``It`` some parameter space, so that ``f(x,theta,params)`` describes a scalar field depending some parameters that vary in ``It``. The ``params`` in the definition of ``f`` can be constants used by the function or othere objects that must be passed to the function definition.
    
    :param function f: multidimensional function to be approximated with format ``f(x,theta,params)``
    :param list grids: this is a list with ``len(grids)=dim(Is)+dim(It)`` which can contain:
      a) 1-dimensional numpy.array of points discretizing the i-th dimension,
      b) a tuple ``(PolyType,QuadType,PolyParams,span)`` where ``PolyType`` is one of the polynomials available in :py:mod:`SpectralToolbox.Spectral1D` and ``QuadType`` is one of the quadrature rules associated to the selected polynomial and ``PolyParams`` are the parameters for the selected polynomial. ``span`` is a tuple defining the left and right end for dimension i (Example: ``(-3,np.inf)``)
      c) a tuple ``(QuadType,span)`` where ``QuadType`` is one of the quadrature rules available in :py:mod:`SpectralToolbox.Spectral1D` without the selection of a particular polynomial type, and ``span`` is defined as above.
    :param object params: any list of parameters to be passed to the function ``f``
    :param int range_dim: define the dimension of the spatial dimension ``Is``. For functionals ``f(theta,params)``, ``dim(Is)=0``. For scalar fileds in 3D, ``dim(Is)=3``.
    :param strung ftype: 'serial' if it can only evaluate the function pointwise,
       'vector' if it can evaluate many points at once.
    :param bool marshal_f: whether to marshal the function f or not. For MPI support, the function f must be marshalable (does this adverb exists??).
    :param bool surrogateONOFF: whether to construct the surrogate or not
    :param str surrogate_type: whether the surrogate will be an interpolating surrogate (``TensorTrain.LINEAR_INTERPOLATION`` or ``TensorTrain.LAGRANGE_INTERPOLATION``) or a projection surrogate (``TensorTrain.PROJECTION``)
    :param list orders: polynomial orders for each dimension if ``TensorTrain.PROJECTION`` is used. If ``orderAdapt==True`` then the ``orders`` are starting orders that can be increased as needed by the construction algorithm. If this parameter is not provided but ``orderAdapt==True``, then the starting order is 1 for all the dimensions.
    :param bool orderAdapt: whether the order is fixed or not.
    :param str stt_store_location: path to a file where function evaluations can be stored and used in order to restart the construction.
    :param bool stt_store_overwrite: whether to overwrite pre-existing files
    :param int stt_store_freq: storage frequency. Determines every how many seconds the state is stored. ``stt_store_freq==0`` stores every time it is possible.
    :param bool empty: Creates an instance without initializing it. All the content can be initialized using the ``setstate()`` function.

    .. note:: For a description of the remaining parameters see :py:class:`TTvec`.
    .. document private functions
    .. automethod:: __getitem__
    .. automethod:: __call__
    
    """

    logger = logging.getLogger(__name__)
    logger.propagate = False
    ch = logging.StreamHandler()
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)
    
    def __init__(self, f, grids, params, range_dim=0, ftype='serial', marshal_f=True,
                 surrogateONOFF=False, surrogate_type=None, orders=None, orderAdapt=None, 
                 eps=1e-4, method="ttdmrg",
                 rs=None, fix_rank=False, Jinit= None, delta=1e-4, maxit=100, 
                 mv_eps=1e-6, mv_maxit = 100,
                 kickrank = None,
                 store_location="",store_overwrite=False, store_freq=0):

        super(STT,self).__init__(store_location, store_freq, store_overwrite)

        ##########################################################
        # List of attributes
        #
        self.generic_approx = None       # np.array of TT approximation (can be TTvec or QTTvec)
        self.TTapprox = None             # np.array of TTvec approximations
        self.TTfour = None               # np.array of TT approximation of the Fourier coefficients
        self.Xs_space = None             # List of points (defining the space grid)
        self.space_shape = None          # Shape of the space dimension
        self.Xs_params = None            # np.array of List of points (defining the parameter grid)
        self.Ws = None                   # np.array of List of weights

        self.marshal_f = None            # Whether to marshal the function or not
        self.surrogateONOFF = None       # Whether to construct the surrogate or just the tt-approximation
        self.surr_type = None            # Can be Projection or Interpolation
        self.orders = None               # np.array of orders for each point
        self.poly_types = None           # List of Polynomial types for projection
        self.poly_params = None          # List of parameters used in the definition of the Polynomials
        self.quad_types = None           # List of quadrature types
        self.poly_Xs = None              # List of points used for the construction of the polynomial basis
        self.poly_Ws = None              # np.array of List of non-normalized weights
        self.Vs = None                   # np.array of List of projection matrices
        self.barycentric_weights = None  # np.array of List of barycentric weights for Lagrange Interp

        self.spans = None                # Span in each dimension
        self.range_dim = None            # Number of dimensions of the spatial space
        self.param_dim = None            # Number of dimensions of the parameter space
        self.TW = None                   # Tensor wrapper containing caching the function evaluations
        self.params = None               # Parameters to be passed to f

        self.method = None
        self.eps = None
        self.rs = None
        self.fix_rank = None
        self.Jinit = None
        self.delta = None
        self.maxit = None
        self.mv_eps = None
        self.mv_maxit = None
        self.kickrank = None

        self.init = False                # Flags whether the construction of the approximation/surrogate is done.

        self.start_build_time = None
        self.stop_build_time = None

        # Parameters to be reset on restarting
        self.f = None                    # Function to be approximated
        self.polys = None                # List of polynomials (from Spectral1D.AVAIL_POLY)
        #
        # End list of attributes
        ##########################################################
        
        self.serialize_list.extend( ['generic_approx', 'TTapprox', 'TTfour', 'Xs_space', 'space_shape', 'Xs_params', 'Ws', 'poly_Ws', 'Vs', 'barycentric_weights', 'surrogateONOFF', 'surr_type', 'orders', 'poly_types', 'poly_params', 'quad_types', 'poly_Xs', 'spans', 'range_dim', 'param_dim', 'params', 'method', 'eps', 'rs', 'fix_rank', 'Jinit', 'delta', 'maxit', 'mv_eps', 'mv_maxit', 'kickrank', 'init','marshal_f'] )
        self.subserialize_list.extend( ['TW'] )

        self.f = f
        self.params = params
        self.marshal_f = marshal_f
        self.surrogateONOFF = surrogateONOFF
        self.surr_type = surrogate_type

        # Store all the tt approximation parameters
        self.method = method
        self.eps = eps
        self.rs = rs
        self.fix_rank = fix_rank
        self.Jinit = Jinit
        self.delta = delta
        self.maxit = maxit
        self.mv_eps = mv_eps
        self.mv_maxit = mv_maxit
        self.kickrank = kickrank
        
        self.range_dim = range_dim
        self.param_dim = len(grids) - self.range_dim
        if self.param_dim < 0: raise AttributeError("The grids argument must respect len(grids) >= range_dim")

        self.poly_types = [None for i in range(self.param_dim)]
        self.poly_params = [None for i in range(self.param_dim)]
        self.quad_types = [None for i in range(self.param_dim)]
        self.spans = [None for i in range(self.param_dim)]
        self.polys = [None for i in range(self.param_dim)]
        
        self.set_grids(grids,orders)

        # Definition of the Tensor Wrapper (works for homogeneous grid on param space)
        if   self.range_dim == 0: dtype = np.float64
        elif self.range_dim > 0: dtype = np.ndarray
        self.TW = TensorWrapper(self.f, \
                                self.Xs_params[tuple([0]*max(self.range_dim,1))],\
                                params = self.params, \
                                ftype = ftype, \
                                dtype = dtype, \
                                store_file = self.store_location, store_object = self,
                                marshal_f=marshal_f)

    def __getstate__(self):
        dd = super(STT,self).__getstate__()
        dd['TW'] = self.TW.getstate()
        return dd
    
    def __setstate__(self,state,f = None):
        super(STT,self).__setstate__( state )
        # Reset subserialized attributes
        tw = TensorWrapper( None, None, None, empty=True)
        tw.setstate(self.TW)
        self.TW = tw
        
        # Reset additional parameters
        self.set_f(f,self.marshal_f)
        self.reset_store_object(self)
        self.reset_store_time()
        self.polys = [None for i in range(self.param_dim)]
        for i, (poly_type, poly_param) in enumerate(zip(self.poly_types,self.poly_params)):
            if poly_type != None:
                self.polys[i] = S1D.generate(poly_type,poly_param)
    
    def __getitem__(self):
        raise NotImplementedError("To be implemented! (Maybe it's better to just use __call__")
    
    def __call__(self,x_in,verbose=False):
        """ Evaluate the surrogate on points ``x_in``
        
        :param np.ndarray x_in: 1 or 2 dimensional array of points in the parameter space where to evaluate the function. In 2 dimensions, each row is an entry, i.e. ``x_in.shape[1] == self.param_dim``
        
        :return: an array with dimension equal to the space dimension (``range_dim``) plus one. If ``A`` is the returned vector and ``range_dim=2``, then ``A[i,:,:]`` is the value of the surrogate for ``x_in[i,:]``
        
        """
        if not (self.init and self.surrogateONOFF):
            raise RuntimeError("The SpectralTensorTrain approximation is not initialized or is not set to construct a surrogate")
        else:
            if not isinstance(x_in,np.ndarray) or x_in.ndim not in [1,2]: raise AttributeError("The input variable must be a 1 or 2 dimensional numpy.ndarray")
            orig_ndim = x_in.ndim
            x = x_in.copy()
            if x.ndim == 1:
                x = np.array([x])

            if x.shape[1] != self.param_dim: raise AttributeError("The input variable has dimension x.shape[1]==%d, while self.param_dim==%d" % (x.shape[1],self.param_dim))

            Np = x.shape[0]
            
            for i in range(self.param_dim):
                if not (np.inf in self.spans[i]):
                    # Check all the values are in the span range
                    if any(np.logical_or(x[:,i] < self.spans[i][0], x[:,i] > self.spans[i][1])):
                        raise AttributeError("The input value " + str(i) + " exceeds the span for dimension " + str(i) )
                    if self.surr_type == PROJECTION:
                        # rescale points to the span [-1,1]
                        x[:,i] = 2. * (x[:,i] - self.spans[i][0]) / (self.spans[i][1] - self.spans[i][0]) - 1.
            
            output = np.empty((Np,) + self.space_shape)

            mat_cache = [ {} for i in range(self.param_dim) ]
            MsI = [ None for i in range(self.param_dim) ]
            
            if self.surr_type == PROJECTION: # Projection
                for ((point,TTf),(_,orders)) in \
                        zip( np.ndenumerate(self.TTfour), np.ndenumerate(self.orders) ):
                    self.logger.debug("Computing point %s" % (str(point)))
                    for i in range(self.param_dim):
                        try: MsI[i] = mat_cache[i][orders[i]]
                        except KeyError:
                            mat_cache[i][orders[i]] = self.polys[i].GradVandermonde(x[:,i], orders[i], 0, norm=True)
                            MsI[i] = mat_cache[i][orders[i]]
                    TTval = TTf.interpolate(MsI)
                    output[ (slice(None,None,None),) + point ] = np.asarray( [ TTval[tuple([i]*self.param_dim)] for i in range(Np) ] ) 
            elif self.surr_type == LINEAR_INTERPOLATION: # Linear Interpolation
                for ((point,TTapp),(_,Xs)) in \
                        zip( np.ndenumerate(self.TTapprox),np.ndenumerate(self.Xs_params) ):
                    self.logger.debug("Computing point %s" % (str(point)))
                    for i in range(self.param_dim):
                        try: MsI[i] = mat_cache[i][tuple(Xs[i])]
                        except KeyError:
                            mat_cache[i][tuple(Xs[i])] = S1D.SparseLinearInterpolationMatrix(Xs[i],x[:,i]).tocsr()
                            MsI[i] = mat_cache[i][tuple(Xs[i])]
                    is_sparse = [True] * self.param_dim
                    TTval = TTapp.interpolate(MsI, is_sparse=is_sparse)
                    output[ (slice(None,None,None),) + point ] = np.asarray( [ TTval[tuple([i]*self.param_dim)] for i in range(Np) ] ) 
            elif self.surr_type == LAGRANGE_INTERPOLATION: # Lagrange Interpolation
                for ((point,TTapp),(_,Xs),(_,bw)) in \
                        zip( np.ndenumerate(self.TTapprox),np.ndenumerate(self.Xs_params),np.ndenumerate(self.barycentric_weights) ):
                    self.logger.debug("Computing point %s" % (str(point)))
                    for i in range(self.param_dim):
                        try: MsI[i] = mat_cache[i][tuple(Xs[i])]
                        except KeyError:
                            mat_cache[i][tuple(Xs[i])] = S1D.LagrangeInterpolationMatrix(Xs[i],bw[i],x[:,i])
                            MsI[i] = mat_cache[i][tuple(Xs[i])]
                    is_sparse = [False] * self.param_dim
                    TTval = TTapp.interpolate(MsI, is_sparse=is_sparse)
                    output[ (slice(None,None,None),) + point ] = np.asarray( [ TTval[tuple([i]*self.param_dim)] for i in range(Np) ] ) 
            else:
                raise AttributeError("Type of interpolation not defined")

            self.logger.debug("Computing DONE")

            if orig_ndim == 1:
                if self.range_dim == 0:
                    return output[ (0,) + tuple( [slice(None,None,None)]*self.range_dim )][0]
                else:
                    return output[ (0,) + tuple( [slice(None,None,None)]*self.range_dim )]
            else:
                if self.range_dim == 0:
                    return output[:,0]
                else:
                    return output

    #####################################################################################
    # Multi-linear Algebra
    #####################################################################################
    
    def __add__(A,B):
        # if not (A.init and B.init): raise RuntimeError("The STT are not initialized")
        C = A.copy()
        C += B
        return C
    
    def __iadd__(A,B):
        if not ( (isinstance(A, STT) and A.init) and \
                     ((isinstance(B,STT) and B.init) or isinstance(B, np.ndarray) or isfloat(B))): 
            raise RuntimeError("The STT are not initialized")
        if isinstance(A, STT) and isinstance(B, STT):
            for (point,TTa),(_,TTb) in \
                    zip( np.ndenumerate(A.TTapprox), np.ndenumerate(B.TTapprox) ):
                A.TTapprox[point] = (TTa + TTb).rounding( min(A.eps,B.eps) )

        elif isinstance(A, STT) and isinstance(B, np.ndarray):
            for (point,TTa),(_,b) in \
                    zip( np.ndenumerate(A.TTapprox), np.ndenumerate(B) ):
                A.TTapprox[point] = (TTa + b).rounding( A.eps )

        elif isinstance(A, STT) and isfloat(B):
            for (point,TTa) in np.ndenumerate(A.TTapprox):
                A.TTapprox[point] = (TTa + B).rounding( A.eps )

        A.prepare_surrogate(force_redo=True)        
        return A
    
    def __neg__(A):
        if not (A.init): raise RuntimeError("The STT is not initialized")
        B = -1. * A
        return B

    def __sub__(A,B):
        # if not (A.init and B.init): raise RuntimeError("The STT are not initialized")
        return A + (-B)
    
    def __isub__(A,B):
        # if not (A.init and B.init): raise RuntimeError("The STT are not initialized")
        A += -B
        return A
    
    def __mul__(A,B):
        # if not (A.init and B.init): raise RuntimeError("The STT are not initialized")
        C = A.copy()
        C *= B
        return C
    
    def __rmul__(A,B):
        # if not (A.init and B.init): raise RuntimeError("The STT are not initialized")
        return (A * B)
    
    def __imul__(A,B):
        if not ( (isinstance(A, STT) and A.init) and \
                     ((isinstance(B,STT) and B.init) or isinstance(B, np.ndarray) or isfloat(B))): 
            raise RuntimeError("The STT are not initialized")
        if isinstance(A,STT) and isinstance(B,STT):
            for (point,TTa),(_,TTb) in \
                zip( np.ndenumerate(A.TTapprox), np.ndenumerate(B.TTapprox) ):
                A.TTapprox[point] = (TTa * TTb).rounding( min(A.eps,B.eps) )
        
        elif isfloat(B) and isinstance(A,STT):
            for (point,TTa) in np.ndenumerate(A.TTapprox):
                A.TTapprox[point] = (TTa * B).rounding( min(A.eps) )

        elif isinstance(A, STT) and isinstance(B, np.ndarray):
            for (point,TTa),(_,b) in \
                    zip( np.ndenumerate(A.TTapprox), np.ndenumerate(B) ):
                A.TTapprox[point] = (TTa * b).rounding( A.eps )

        A.prepare_surrogate(force_redo=True)
        return A
    
    def __pow__(A,n):
        if not A.init: raise RuntimeError("The STT is not initialized")
        if not isint(n): raise RuntimeError("n must be an integer")
        B = A.copy()
        for (point,TTb) in np.ndenumerate(B.TTapprox):
            B.TTapprox[point] = (TTb**n).rounding( B.eps )
        B.prepare_surrogate(force_redo=True)
        return B
    
    #####################################################################################
    # End Multi-linear Algebra
    #####################################################################################

    def getstate(self):
        return self.__getstate__()
    
    def setstate(self,state,f = None):
        self.__setstate__(state, f)

    def h5store(self, h5file):
        self.TW.h5store(h5file)
    
    def h5load(self, h5file):
        self.TW.h5load(h5file)
    
    def to_v_0_3_0(self, store_location):
        """ Upgrade to v0.3.0
        
        :param string filename: path to the filename. This must be the main filename with no extension.
        """
        super(STT,self).to_v_0_3_0(store_location)
        # Call the upgrade for the Tensor Wrapper
        self.TW.to_v_0_3_0(store_location)
        # Call the upgrade for all the TTapproximations
        for (_,tt) in np.ndenumerate(self.TTapprox):
            if tt != None:
                tt.to_v_0_3_0(store_location)
        for (_,tt) in np.ndenumerate(self.TTfour):
            if tt != None:
                tt.to_v_0_3_0(store_location)
        for (_,tt) in np.ndenumerate(self.generic_approx):
            if tt != None:
                tt.to_v_0_3_0(store_location)
    
    def copy(self):
        C = copy.deepcopy(self)
        # Reset additional parameters
        C.set_f(self.f,self.marshal_f)
        C.polys = [None for i in range(self.param_dim)]
        for i, (poly_type, poly_param) in enumerate(zip(self.poly_types,self.poly_params)):
            if poly_type != None:
                C.polys[i] = S1D.generate(poly_type,poly_param)
        return C
 
    def set_f(self,f,marshal_f=True):
        if f is None:
            self.TW.reset_f_marshal()
        else:
            self.TW.set_f(f,marshal_f)
        self.f = self.TW.f
    
    def set_params(self, params):
        self.params = params
        self.TW.set_params(params)

    def reset_store_object(self, obj):
        self.set_store_object( None )
        self.TW.set_store_object( obj )
        for _, TTapp in np.ndenumerate(self.generic_approx):
            if TTapp != None: TTapp.set_store_object(obj)

    def set_grids(self,grids,orders=None):
        # Store grid for spatial space
        self.Xs_space = []
        for i in range(self.range_dim):
            if isinstance(grids[i],np.ndarray): self.Xs_space.append(grids[i])
            else: raise AttributeError("The grids argument must contain np.ndarray in the elements grids[:range_dim]")
        if self.range_dim == 0:
            self.Xs_space.append( np.array([0]) )
        
        # Set the shape of the space
        self.space_shape = tuple([ len(x) for x in self.Xs_space ])
        
        # Initialize variables
        self.Xs_params = np.empty(self.space_shape, dtype=list)
        self.Ws = np.empty(self.space_shape, dtype=list)
        self.poly_Xs = np.empty(self.space_shape, dtype=list)
        self.poly_Ws = np.empty(self.space_shape, dtype=list)
        self.orders = np.empty(self.space_shape, dtype=list)
        self.Vs = np.empty(self.space_shape, dtype=list)
        
        if orders is None:
            for p, value in np.ndenumerate(self.orders): self.orders[p] = [2 for i in range(self.param_dim)]
        else:
            if not (len(orders) == self.param_dim):
                raise AttributeError("The condition len(orders) == len(grids)-range_dim must hold.")
            for p, value in np.ndenumerate(self.orders): self.orders[p] = orders

        # Extract types
        for i,grid in enumerate(grids[self.range_dim:]):
            if isinstance(grid,np.ndarray):
                self.spans[i] = (grid[0], grid[-1])
            elif isinstance(grid,tuple) and len(grid)==4:
                (self.poly_types[i],self.quad_types[i],self.poly_params[i],self.spans[i]) = grid
                self.polys[i] = S1D.generate(self.poly_types[i],self.poly_params[i])
            elif isinstance(grid,tuple) and len(grid)==2:
                (self.quad_types[i],self.spans[i]) = grid

        # Construct grids and weights if needed for parameter space
        for point, val in np.ndenumerate(self.Xs_params):
            self.Xs_params[point] = [None for i in range(self.param_dim)]
            self.Ws[point] = [None for i in range(self.param_dim)]
            self.poly_Xs[point] = [None for i in range(self.param_dim)]
            self.poly_Ws[point] = [None for i in range(self.param_dim)]
            self.Vs[point] = [None for i in range(self.param_dim)]
            
            for i,grid in enumerate(grids[self.range_dim:]):
                if isinstance(grid,np.ndarray): 
                    self.Xs_params[point][i] = grid
                elif isinstance(grid,tuple) and len(grid)==4:
                    (x,w) = self.polys[i].Quadrature(self.orders[point][i],quadType=self.quad_types[i], norm=False)
                    self.poly_Xs[point][i] = x.copy()
                    self.poly_Ws[point][i] = w.copy()
                    if not (np.inf in self.spans[i]):
                        # rescale points to the span
                        x = self.spans[i][0] + (x+1.)/2. * (self.spans[i][1] - self.spans[i][0])
                        w = w / np.sum(w) * (self.spans[i][1] - self.spans[i][0])
                    self.Xs_params[point][i] = x
                    self.Ws[point][i] = w
                elif isinstance(grid,tuple) and len(grid)==2:
                    (x,w) = S1D.QUADS[self.quad_types[i]](self.orders[i],norm=False)
                    self.poly_Xs[point][i] = x.copy()
                    if not (np.inf in self.spans[i]):
                        # rescale points to the span
                        x = self.spans[i][0] + (x+1.)/2. * (self.spans[i][1] - self.spans[i][0])
                        w = w / np.sum(w) * (self.spans[i][1] - self.spans[i][0])
                    self.Xs_params[point][i] = x
                    self.Ws[point][i] = w
                else:
                    raise AttributeError("The %d argument of grid is none of the types accepted." % i)
    
    def empty_generic_approx(self):
        self.generic_approx = np.empty(self.space_shape, dtype=TTvec)
    
    def new_generic_approx(self,multidim_point):
        return TTvec( self.TW,
                      store_location=self.store_location,
                      store_freq=self.store_freq,
                      store_object=self,
                      multidim_point=multidim_point )

    def build(self,maxprocs=None,force_redo=False):
        self.start_build_time = time.clock()
        
        if self.generic_approx is None or force_redo:
            self.empty_generic_approx()
        
        self.TW.set_maxprocs(maxprocs) # Set the number of processors for MPI
        for point, val in np.ndenumerate(self.generic_approx):
            if val is None or not val.init:
                try:
                    totsize = float(self.TW.get_size())
                    self.logger.info( "Point %s - Fill: %f%%" % (str(point),float(self.TW.get_fill_level())/totsize * 100.) )
                except OverflowError:
                    self.logger.info( "Point %s - Fill: %d" % (str(point),self.TW.get_fill_level()) )
                multidim_point = point if self.range_dim > 0 else None
                # Build generic_approx for the selected point
                if val is None or self.generic_approx[point].Jinit is None:
                    # Find all the back neighbors and select the first found to start from
                    neigh = None
                    for i in range(self.range_dim-1,-1,-1): 
                        if point[i]-1 >= 0:
                            pp = list(point)
                            pp[i] -= 1
                            neigh = self.generic_approx[tuple(pp)]
                            break
                    # If a neighbor is found, select the rank and the fibers to be used
                    if neigh != None:
                        if self.method == 'ttcross':
                            rs = neigh.ranks()
                            for i in range(1,len(rs)-1):
                                rs[i] += 1
                            Js = neigh.Js_last[-2]
                            # Trim the fibers according to rs (This allow the rank to decrease as well)
                            for r,(j,J) in zip(rs[1:-1],enumerate(Js)):
                                Js[j] = random.sample(J,r)
                            self.rs = rs
                            self.Jinit = Js
                        elif self.method == 'ttdmrg':
                            self.Jinit = neigh.Jinit
                            

                    self.generic_approx[point] = self.new_generic_approx(multidim_point)
                    self.generic_approx[point].build(eps=self.eps, method=self.method,
                                                     rs=self.rs, 
                                                     fix_rank=self.fix_rank, 
                                                     Jinit= self.Jinit, 
                                                     delta=self.delta, maxit=self.maxit, 
                                                     mv_eps=self.mv_eps,
                                                     mv_maxit = self.mv_maxit,
                                                     kickrank=self.kickrank)
                else:           # Use stored Jinit to restart
                    Jinit = self.generic_approx[point].Jinit
                    rs = [ len(Jinit[i]) for i in range(len(self.generic_approx[point].Jinit)) ]
                    rs.insert(0,1)
                    self.generic_approx[point].A = self.TW
                    self.generic_approx[point].build(eps=self.eps, method=self.method,
                                                     rs=rs, 
                                                     fix_rank=self.fix_rank, 
                                                     Jinit=Jinit, 
                                                     delta=self.delta, maxit=self.maxit, 
                                                     mv_eps=self.mv_eps,
                                                     mv_maxit = self.mv_maxit,
                                                     kickrank=self.kickrank)

                self.store()

        self.store(force=True)

        self.prepare_TTapprox()
        
        self.store()
        
        self.prepare_surrogate()
        
        self.init = True

        self.store(force=True)

        self.stop_build_time = time.clock()
    
    def prepare_TTapprox(self, force_redo=False):
        """ Prepares the TTapprox from the generic_approx
        """
        if self.TTapprox is None or force_redo:
            self.TTapprox = np.empty(self.space_shape, dtype=TTvec)
        
        for (point,gen_app) in np.ndenumerate(self.generic_approx):
            self.TTapprox[point] = self.generic_to_TTvec( gen_app )

    def generic_to_TTvec(self, gen_app):
        return gen_app

    def prepare_surrogate(self, force_redo=False):
        if self.surrogateONOFF and self.surr_type == PROJECTION:
            self.build_projection_surrogate(force_redo)
        
        if self.surrogateONOFF and self.surr_type == LAGRANGE_INTERPOLATION:
            self.build_barycentric_weights(force_redo)
    
    def build_projection_surrogate(self,force_redo=False):
        # Create the TT approximation of the Fourier coefficients (Project)
        if self.TTfour is None or force_redo:
            self.TTfour = np.empty(self.space_shape, dtype=TTvec)

        for (point, val),(_,ttapp) in zip(np.ndenumerate(self.TTfour), np.ndenumerate(self.TTapprox)) :
            if val is None and ttapp != None:
                self.Vs[point] = [ self.polys[i].GradVandermonde(self.poly_Xs[point][i], self.orders[point][i], 0, norm=True) for i in range(self.param_dim) ]
                self.logger.debug("Projecting point %s" % (str(point)))
                self.TTfour[point] = ttapp.project(self.Vs[point],self.poly_Ws[point])
        self.logger.debug("Projection Done")
    
    def build_barycentric_weights(self, force_redo=False):
        if self.barycentric_weights is None or force_redo:
            self.barycentric_weights = np.empty(self.space_shape, dtype=list)
        
        for (point, _),(_,X) in \
                zip( np.ndenumerate(self.barycentric_weights), np.ndenumerate(self.Xs_params) ):
            self.barycentric_weights[point] = [ S1D.BarycentricWeights(X[i]) for i in range(self.param_dim) ]

    def integrate(self):
        """ Compute the integral of the approximated function
        
        :return: an array with dimension equal to the space dimension (``range_dim``), containing the value of the integral.

        """
        
        if not self.init or not np.all( np.all( [np.not_equal(W, None) for W in self.Ws]) ):
            raise RuntimeError("The SpectralTensorTrain approximation is not initialized or is not for computing integrals.")
        else:
            output = np.zeros(self.space_shape)
            for (point,TTapp),(_,Ws) in \
                    zip( np.ndenumerate(self.TTapprox), np.ndenumerate(self.Ws) ):
                output[point] = mla.contraction(TTapp, Ws)
            if self.range_dim == 0: return output[0]
            else: return output
            
class SQTT(STT):
    """ Constructor of the Spectral Quantics Tensor Train approximation. Given a function ``f(x,theta,params):(Is, It) -> R``
    with ``dim(Is)=n`` and ``dim(It)=d``, construct an approximation of ``g(theta,params): It -> h_t(Is)``. For example ``Is`` could be the discretization of a spatial dimension, and ``It`` some parameter space, so that ``f(x,theta,params)`` describes a scalar field depending some parameters that vary in ``It``. The ``params`` in the definition of ``f`` can be constants used by the function or othere objects that must be passed to the function definition.
    
    :param function f: multidimensional function to be approximated with format ``f(x,theta,params)``
    :param list grids: this is a list with ``len(grids)=dim(Is)+dim(It)`` which can contain:
      a) 1-dimensional numpy.array of points discretizing the i-th dimension,
      b) a tuple ``(PolyType,QuadType,PolyParams,span)`` where ``PolyType`` is one of the polynomials available in :py:mod:`SpectralToolbox.Spectral1D` and ``QuadType`` is one of the quadrature rules associated to the selected polynomial and ``PolyParams`` are the parameters for the selected polynomial. ``span`` is a tuple defining the left and right end for dimension i (Example: ``(-3,np.inf)``)
      c) a tuple ``(QuadType,span)`` where ``QuadType`` is one of the quadrature rules available in :py:mod:`SpectralToolbox.Spectral1D` without the selection of a particular polynomial type, and ``span`` is defined as above.
    :param object params: any list of parameters to be passed to the function ``f``
    :param int range_dim: define the dimension of the spatial dimension ``Is``. For functionals ``f(theta,params)``, ``dim(Is)=0``. For scalar fileds in 3D, ``dim(Is)=3``.
    :param strung ftype: 'serial' if it can only evaluate the function pointwise,
       'vector' if it can evaluate many points at once.
    :param bool marshal_f: whether to marshal the function f or not. For MPI support, the function f must be marshalable (does this adverb exists??).
    :param int base: ``base`` parameter for Quantics Tensor Train
    :param bool surrogateONOFF: whether to construct the surrogate or not
    :param str surrogate_type: whether the surrogate will be an interpolating surrogate (``TensorTrain.LINEAR_INTERPOLATION`` or ``TensorTrain.LAGRANGE_INTERPOLATION``) or a projection surrogate (``TensorTrain.PROJECTION``)
    :param list orders: polynomial orders for each dimension if ``TensorTrain.PROJECTION`` is used. If ``orderAdapt==True`` then the ``orders`` are starting orders that can be increased as needed by the construction algorithm. If this parameter is not provided but ``orderAdapt==True``, then the starting order is 1 for all the dimensions.
    :param bool orderAdapt: whether the order is fixed or not.
    :param str stt_store_location: path to a file where function evaluations can be stored and used in order to restart the construction.
    :param bool stt_store_overwrite: whether to overwrite pre-existing files
    :param int stt_store_freq: storage frequency. Determines every how many seconds the state is stored. ``stt_store_freq==0`` stores every time it is possible.
    :param bool empty: Creates an instance without initializing it. All the content can be initialized using the ``setstate()`` function.

    .. note:: For a description of the remaining parameters see :py:class:`TTvec`.
    .. document private functions
    .. automethod:: __getitem__
    .. automethod:: __call__
    
    """

    def __init__(self, f, grids, params, range_dim=0, ftype='serial', marshal_f=True,
                 base = 2,
                 surrogateONOFF=False, surrogate_type=None, orders=None, orderAdapt=None, 
                 eps=1e-4, method="ttdmrg",
                 rs=None, fix_rank=False, Jinit= None, delta=1e-4, maxit=100, 
                 mv_eps=1e-6, mv_maxit = 100,
                 kickrank = None,
                 store_location="",store_overwrite=False, store_freq=0):
        
        super(SQTT,self).__init__(f, grids, params, range_dim=range_dim,
                                  ftype=ftype,
                                  marshal_f=marshal_f,
                                  surrogateONOFF=surrogateONOFF,
                                  surrogate_type=surrogate_type,
                                  orders=orders, orderAdapt=orderAdapt, 
                                  eps=eps, method=method,
                                  rs=rs, fix_rank=fix_rank, Jinit=Jinit,
                                  delta=delta, maxit=maxit, 
                                  mv_eps=mv_eps, mv_maxit=mv_maxit,
                                  kickrank=kickrank,
                                  store_location=store_location,
                                  store_overwrite=store_overwrite, store_freq=store_freq)
        self._init(base)

    def _init(self, base):
        ##########################################################
        # List of attributes
        #
        self.base = None        
        #
        # End list of attributes
        ##########################################################
        self.serialize_list.extend( ['base'] )
        self.base = base
    
    def empty_generic_approx(self):
        self.generic_approx = np.empty(self.space_shape, dtype=QTTvec)
    
    def new_generic_approx(self,multidim_point):
        return QTTvec( self.TW, self.base,
                       store_location=self.store_location,
                       store_freq=self.store_freq,
                       store_object=self,
                       multidim_point=multidim_point )
    
    def generic_to_TTvec(self, gen_app):
        return gen_app.to_TTvec()

class SWTT(STT):
    """ Constructor of the Spectral Weighted Tensor Train approximation :cite:`Bigoni2015`. Given a function ``f(x,theta,params):(Is, It) -> R``
    with ``dim(Is)=n`` and ``dim(It)=d``, construct an approximation of ``g(theta,params): It -> h_t(Is)``. For example ``Is`` could be the discretization of a spatial dimension, and ``It`` some parameter space, so that ``f(x,theta,params)`` describes a scalar field depending some parameters that vary in ``It``. The ``params`` in the definition of ``f`` can be constants used by the function or othere objects that must be passed to the function definition.
    
    :param function f: multidimensional function to be approximated with format ``f(x,theta,params)``
    :param list grids: this is a list with ``len(grids)=dim(Is)+dim(It)`` which can contain:
      a) 1-dimensional numpy.array of points discretizing the i-th dimension,
      b) a tuple ``(PolyType,QuadType,PolyParams,span)`` where ``PolyType`` is one of the polynomials available in :py:mod:`SpectralToolbox.Spectral1D` and ``QuadType`` is one of the quadrature rules associated to the selected polynomial and ``PolyParams`` are the parameters for the selected polynomial. ``span`` is a tuple defining the left and right end for dimension i (Example: ``(-3,np.inf)``)
      c) a tuple ``(QuadType,span)`` where ``QuadType`` is one of the quadrature rules available in :py:mod:`SpectralToolbox.Spectral1D` without the selection of a particular polynomial type, and ``span`` is defined as above.
    :param object params: any list of parameters to be passed to the function ``f``
    :param int range_dim: define the dimension of the spatial dimension ``Is``. For functionals ``f(theta,params)``, ``dim(Is)=0``. For scalar fileds in 3D, ``dim(Is)=3``.
    :param strung ftype: 'serial' if it can only evaluate the function pointwise,
       'vector' if it can evaluate many points at once.
    :param bool marshal_f: whether to marshal the function f or not. For MPI support, the function f must be marshalable (does this adverb exists??).
    :param bool surrogateONOFF: whether to construct the surrogate or not
    :param str surrogate_type: whether the surrogate will be an interpolating surrogate (``TensorTrain.LINEAR_INTERPOLATION`` or ``TensorTrain.LAGRANGE_INTERPOLATION``) or a projection surrogate (``TensorTrain.PROJECTION``)
    :param list orders: polynomial orders for each dimension if ``TensorTrain.PROJECTION`` is used. If ``orderAdapt==True`` then the ``orders`` are starting orders that can be increased as needed by the construction algorithm. If this parameter is not provided but ``orderAdapt==True``, then the starting order is 1 for all the dimensions.
    :param bool orderAdapt: whether the order is fixed or not.
    :param str stt_store_location: path to a file where function evaluations can be stored and used in order to restart the construction.
    :param bool stt_store_overwrite: whether to overwrite pre-existing files
    :param int stt_store_freq: storage frequency. Determines every how many seconds the state is stored. ``stt_store_freq==0`` stores every time it is possible.
    :param bool empty: Creates an instance without initializing it. All the content can be initialized using the ``setstate()`` function.

    .. note:: For a description of the remaining parameters see :py:class:`TTvec`.
    .. document private functions
    .. automethod:: __getitem__
    .. automethod:: __call__
    
    """
    def empty_generic_approx(self):
        self.generic_approx = np.empty(self.space_shape, dtype=WTTvec)
    
    def new_generic_approx(self,multidim_point):
        return WTTvec( self.TW, self.Ws[0],
                       store_location=self.store_location,
                       store_freq=self.store_freq,
                       store_object=self,
                       multidim_point=multidim_point )
    
    def generic_to_TTvec(self, gen_app):
        return gen_app.to_TTvec()

class SWQTT(SQTT, SWTT):
    """ Constructor of the Spectral Quantics Tensor Train approximation. Given a function ``f(x,theta,params):(Is, It) -> R``
    with ``dim(Is)=n`` and ``dim(It)=d``, construct an approximation of ``g(theta,params): It -> h_t(Is)``. For example ``Is`` could be the discretization of a spatial dimension, and ``It`` some parameter space, so that ``f(x,theta,params)`` describes a scalar field depending some parameters that vary in ``It``. The ``params`` in the definition of ``f`` can be constants used by the function or othere objects that must be passed to the function definition.
    
    :param function f: multidimensional function to be approximated with format ``f(x,theta,params)``
    :param list grids: this is a list with ``len(grids)=dim(Is)+dim(It)`` which can contain:
      a) 1-dimensional numpy.array of points discretizing the i-th dimension,
      b) a tuple ``(PolyType,QuadType,PolyParams,span)`` where ``PolyType`` is one of the polynomials available in :py:mod:`SpectralToolbox.Spectral1D` and ``QuadType`` is one of the quadrature rules associated to the selected polynomial and ``PolyParams`` are the parameters for the selected polynomial. ``span`` is a tuple defining the left and right end for dimension i (Example: ``(-3,np.inf)``)
      c) a tuple ``(QuadType,span)`` where ``QuadType`` is one of the quadrature rules available in :py:mod:`SpectralToolbox.Spectral1D` without the selection of a particular polynomial type, and ``span`` is defined as above.
    :param object params: any list of parameters to be passed to the function ``f``
    :param int range_dim: define the dimension of the spatial dimension ``Is``. For functionals ``f(theta,params)``, ``dim(Is)=0``. For scalar fileds in 3D, ``dim(Is)=3``.
    :param strung ftype: 'serial' if it can only evaluate the function pointwise,
       'vector' if it can evaluate many points at once.
    :param bool marshal_f: whether to marshal the function f or not. For MPI support, the function f must be marshalable (does this adverb exists??).
    :param int base: ``base`` parameter for Quantics Tensor Train
    :param bool surrogateONOFF: whether to construct the surrogate or not
    :param str surrogate_type: whether the surrogate will be an interpolating surrogate (``TensorTrain.LINEAR_INTERPOLATION`` or ``TensorTrain.LAGRANGE_INTERPOLATION``) or a projection surrogate (``TensorTrain.PROJECTION``)
    :param list orders: polynomial orders for each dimension if ``TensorTrain.PROJECTION`` is used. If ``orderAdapt==True`` then the ``orders`` are starting orders that can be increased as needed by the construction algorithm. If this parameter is not provided but ``orderAdapt==True``, then the starting order is 1 for all the dimensions.
    :param bool orderAdapt: whether the order is fixed or not.
    :param str stt_store_location: path to a file where function evaluations can be stored and used in order to restart the construction.
    :param bool stt_store_overwrite: whether to overwrite pre-existing files
    :param int stt_store_freq: storage frequency. Determines every how many seconds the state is stored. ``stt_store_freq==0`` stores every time it is possible.
    :param bool empty: Creates an instance without initializing it. All the content can be initialized using the ``setstate()`` function.

    .. note:: For a description of the remaining parameters see :py:class:`TTvec`.
    .. document private functions
    .. automethod:: __getitem__
    .. automethod:: __call__
    
    """
    def __init__(self, f, grids, params, range_dim=0, ftype='serial', marshal_f=True,
                 base = 2,
                 surrogateONOFF=False, surrogate_type=None, orders=None, orderAdapt=None, 
                 eps=1e-4, method="ttdmrg",
                 rs=None, fix_rank=False, Jinit= None, delta=1e-4, maxit=100, 
                 mv_eps=1e-6, mv_maxit = 100,
                 kickrank = None,
                 store_location="",store_overwrite=False, store_freq=0):
        SQTT.__init__(self, f, grids, params, range_dim=range_dim,
                      marshal_f=marshal_f, ftype=ftype, base=base,
                      surrogateONOFF=surrogateONOFF,
                      surrogate_type=surrogate_type, orders=orders,
                      orderAdapt=orderAdapt, 
                      eps=eps, method=method,
                      rs=rs, fix_rank=fix_rank, Jinit=Jinit,
                      delta=delta, maxit=maxit, 
                      mv_eps=mv_eps, mv_maxit=mv_maxit,
                      kickrank=kickrank,
                      store_location=store_location,
                      store_overwrite=store_overwrite,
                      store_freq=store_freq)

    def empty_generic_approx(self):
        self.generic_approx = np.empty(self.space_shape, dtype=WQTTvec)

    def new_generic_approx(self,multidim_point):
        return WQTTvec( self.TW, self.Ws[0], base=self.base,
                        store_location=self.store_location,
                        store_freq=self.store_freq,
                        store_object=self,
                        multidim_point=multidim_point )
    
    def generic_to_TTvec(self, gen_app):
        wttvec = gen_app.to_TTvec()
        wttvec.remove_weights()
        return wttvec