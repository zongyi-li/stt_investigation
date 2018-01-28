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

__all__ = ['TensorWrapper']

import sys
import time
import datetime
import logging
import operator
import itertools
try:
    import itertools.imap as map
    import itertools.ifilter as filter
except ImportError:
    pass
import random
import shutil
import os.path
import numpy as np
import numpy.linalg as npla
import math
import scipy.linalg as scla
import marshal, types
import warnings
import h5py
try:
    import mpi_map
    MPI_SUPPORT = True
except ImportError:
    MPI_SUPPORT = False
from functools import reduce

from TensorToolbox.core import idxunfold, idxfold, expand_idxs, storable_object

class TensorWrapper(storable_object):
    """ A tensor wrapper is a data structure W that given a multi-dimensional scalar function f(X,params), and a set of coordinates {{x1}_i1,{x2}_i2,..,{xd}_id} indexed by the multi index {i1,..,id}, let you access f(x1_i1,..,xd_id) by W[i1,..,id]. The function evaluations are performed "as needed" and stored for future accesses.

    :param f: multi-dimensional scalar function of type f(x,params), x being a list.
    :param list X: list of arrays with coordinates for each dimension
    :param tuple params: parameters to be passed to function f
    :param list W: list of arrays with weights for each dimension
    :param string twtype: 'array' values are stored whenever computed, 'view' values are never stored and function f is always called
    :param strung ftype: 'serial' if it can only evaluate the function pointwise,
       'vector' if it can evaluate many points at once.
    :param dict data: initialization data of the Tensor Wrapper (already computed entries)
    :param type dtype: type of output to be expected from f
    :param str store_file: file where to store the data
    :param object store_object: a storable object that must be stored in place of the TensorWrapper
    :param bool store_freq: how frequently to store the TensorWrapper (seconds)
    :param bool store_overwrite: whether to overwrite pre-existing files.
    :param bool empty: Creates an instance without initializing it. All the content can be initialized using the ``setstate()`` function.
    :param int maxprocs: Number of processors to be used in the function evaluation (MPI)
    :param bool marshal_f: whether to marshal the function or not

    Several shape parameters are used by the TensorWrapper in order to keep track of reshaping and slicings, without affecting the underlying shape of the tensor which is always preserved. The following table lists the existing shapes and their meaning.

    +------------------------------------------------+--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | Shape attribute/function                       | Applied transformations (ordered)    | Description                                                                                                                                                                                                                                                      |
    +================================================+======================================+==================================================================================================================================================================================================================================================================+
    | :py:meth:`~TensorWrapper.get_global_shape`     | None                                 | The original shape of the tensor. This shape can be modified only  through a refinement of the grid using the function :py:meth:`~TensorWrapper.refine`.                                                                                                         |
    +------------------------------------------------+--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`~TensorWrapper.get_view_shape`       | VIEW                                 | The particular view of the tensor, defined by the view in :py:attr:`TensorWrapper.maps` set active using :py:meth:`~TensorWrapper.set_active_view`.                                                                                                              |
    +------------------------------------------------+--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`~TensorWrapper.get_extended_shape`   | VIEW, QUANTICS                       | The shape of the extended tensor in order to allow for the quantics folding with basis :py:attr:`TensorWrapper.Q`.                                                                                                                                               |
    +------------------------------------------------+--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`~TensorWrapper.get_ghost_shape`      | VIEW, QUANTICS, RESHAPE              | The shape of the tensor reshaped using :py:meth:`~TensorWrapper.reshape`. If a Quantics folding is pre-applied, then the reshape is on the extended shape.                                                                                                       |
    +------------------------------------------------+--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+
    | :py:meth:`~TensorWrapper.get_shape`            | VIEW, QUANTICS, RESHAPE, FIX_IDXS    | The shape of the tensor with :py:meth:`~TensorWrapper.fix_indices` and :py:meth:`~TensorWrapper.release_indices`. This is the view that is always used when the tensor is accessed through the function :py:meth:`~TensorWrapper.__getitem__` (i.e. ``TW[...]``) |
    +------------------------------------------------+--------------------------------------+------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------------+

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

    # FILL_VALUE = 0.0            # Value used to fill powQ tensors
    
    def __init__(self, f, X, params=None, W=None, twtype='array', ftype='serial',
                 data=None, dtype=np.float,
                 store_file = "", store_object = None, store_freq=None,
                 store_overwrite=False,
                 empty=False,
                 maxprocs=None,
                 marshal_f=True):

        super(TensorWrapper,self).__init__(store_file, 
                                           store_freq=store_freq, 
                                           store_overwrite=store_overwrite)

        #######################################
        # List of attributes
        self.f_code = None               # Marshal string of the function f
        self.X = None
        self.W = None
        self.params = None
        self.dtype = np.float
        self.twtype = None
        self.ftype = None
        self.data = None

        self.maps = {}          # Multiple views
        self.view = None        # Multiple views

        self.serialize_list.extend( ['X', 'W', 'dtype', 'params', 'twtype', 'ftype',
                                     'f_code', 'maps', 'view'] )
        self.subserialize_list.extend( [] )

        # Attributes which are not serialized and need to be reset on reload
        self.f = None
        self.store_object = None
        self.__maxprocs = None             # Number of processors to be used (MPI)

        self.shape = None
        self.ndim = None
        self.size = None
        
        self.stored_keys = None # Set of stored keys (to improve the saving speed)

        self.active_weights = None
        # End list of attributes
        #################################

        self.active_weights = False        
        self.stored_keys = set()
        if not empty: 
            self.set_f(f,marshal_f)
            self.ftype = ftype
            self.params = params
            self.X = X
            self.W = W
            self.maps['full'] = { 'X_map': self.X, 
                                  'idx_map': [ range( len(x) ) for x in self.X ],
                                  'Q': None,
                                  'ghost_shape': None,
                                  'fix_idxs': [],
                                  'fix_dims': [] }
            self.set_active_view('full')
            self.shape = self.get_shape()
            self.ndim = self.get_ndim()
            self.size = self.get_size()
            self.twtype = twtype
            self.dtype = dtype

            self.set_store_object(store_object)

            self.set_maxprocs(maxprocs)
            if self.twtype == 'array':
                if data == None:
                    self.data = {} # Dictionary in python behave as a hash-table
                else:
                    self.data = data
            elif self.twtype != 'view':
                raise ValueError("Tensor Wrapper type not existent. Use 'array' or 'view'")
    
    def __getstate__(self):
        return super(TensorWrapper,self).__getstate__()
        
    def __setstate__(self,state,f = None, store_object = None):
        super(TensorWrapper,self).__setstate__( state, store_object )
        # Reset parameters
        if f == None: self.reset_f_marshal()
        else: self.set_f(f)
        self.set_store_object( store_object )
        self.fix_idxs = []
        self.fix_dims = []
        self.__maxprocs = None
        self.reset_ghost_shape()
        self.active_weights = False

    def set_weights(self, W):
        """ Set a new list of weights for the tensor
        :param list W: list of np.ndarray with weights for each dimension
        """
        if len(W) != len(self.get_global_shape()):
            raise ValueError("The provided set of weights has not the right dimension: len(W)=%d, dim=%d" % (len(W),len(self.get_global_shape())))
        if any( [ len(wi) != si for wi,si in zip(W,self.get_global_shape()) ] ):
            raise ValueError("The provided set of weights contains at least one dimension which is not conformal with the tensor grid.")
        self.W = W

    def set_active_weights(self,flag):
        """ Set whether to use the weights or not.

        :param bool flag: If ``True`` the items returned by the Tensor Wrapper will be weighted according to the weights provided at construction time. If ``False`` the original values of the function will be returned.
        """
        self.active_weights = flag

    def getstate(self):
        return self.__getstate__();
    
    def setstate(self,state,f = None, store_object = None):
        return self.__setstate__(state, f, store_object)

    def h5store(self, h5file):
        # Store the data table in the h5 file. Update if possible.
        try:
            tw_grp = h5file['TW']
        except KeyError:
            # Create the group, the data structure and dump data
            if len(self.data) > 0:
                tw_grp = h5file.create_group('TW')
                keys = list(self.data.keys())
                values = list(self.data.values())
                tw_grp.create_dataset("keys", data=keys, maxshape=(None,self.get_ndim()) )
                tw_grp.create_dataset("values", data=values,
                                      maxshape=(None,)+self.data[keys[0]].shape )
                self.stored_keys = set( keys )
        else:
            # Increase the shape of the datasets to accommodate for the new data
            tw_grp["keys"].resize(len(self.data),axis=0)
            tw_grp["values"].resize(len(self.data),axis=0)

            # Store by data chunk
            N = 100000
            it = 0
            dvals = len(tw_grp["values"].shape)
            while len(self.stored_keys) < len(self.data):
                # Get the missing data
                new_data = dict(filter(lambda i:i[0] not in self.stored_keys, 
                                       itertools.islice( self.data.items(), 
                                                         it*N, 
                                                         min((it+1)*N,len(self.data)) 
                                                     ) 
                                   ))
                it += 1
                # Assign new values to the datasets
                keys = list(new_data.keys())
                values = list(new_data.values())
                tw_grp["keys"][len(self.stored_keys):len(self.stored_keys)+len(new_data),:] = keys
                tw_grp["values"][ (slice(len(self.stored_keys),len(self.stored_keys)+len(new_data),None),) + tuple([slice(None,None,None)]*(dvals-1)) ] = values
                self.stored_keys |= set( keys )

    def h5load(self, h5file):
        try:
            tw_grp = h5file['TW']
        except KeyError:
            # The data structure is empty. Do nothing.
            pass
        else:
            # Load by data chunk
            N = 100000
            it = 0
            dvals = len(tw_grp["values"].shape)
            Ndata = tw_grp["keys"].shape[0]
            self.data = {}
            while len(self.data) < Ndata:
                keys = tw_grp["keys"][it*N:min((it+1)*N,Ndata), :]
                values = tw_grp["values"][ (slice(it*N, min((it+1)*N,Ndata),None),) + tuple([slice(None,None,None)]*(dvals-1))]
                def f(i,dvals=dvals):
                    return ( tuple(keys[i,:]), values[(i,) + tuple([slice(None,None,None)]*(dvals-1))] )
                self.data.update( map( f, range(keys.shape[0]) ) )
                it += 1
            self.stored_keys = set( self.data.keys() )
    
    def to_v_0_3_0(self, store_location):
        """ Upgrade to v0.3.0
        
        :param string filename: path to the filename. This must be the main filename with no extension.
        """
        super(TensorWrapper,self).to_v_0_3_0(store_location)
        # Upgrade serialize list
        self.serialize_list.remove( 'data' )

    def copy(self):
        return TensorWrapper(self.f, self.X, params=self.params, twtype=self.twtype, data=self.data.copy())

    #####################################################
    #               SHAPES AND VIEWS                    #
    #####################################################

    ##########
    # GLOBAL #
    ##########
    def get_global_shape(self):
        """ Always returns the shape of the underlying tensor
        """
        dim = [ len(coord) for coord in self.X ]
        return tuple(dim)
    
    def get_global_ndim(self):
        """ Always returns the ndim of the underlying tensor
        """
        return len(self.get_global_shape())
    
    def get_global_size(self):
        """ Always returns the size of the underlying tensor
        """
        return reduce(operator.mul, self.get_global_shape(), 1)
    
    #########
    # VIEWS #
    #########
    def get_view_shape(self):
        """ Always returns the shape of the current view
        """
        dim = [ len(coord) for coord in self.maps[self.view]['X_map'] ]
        return tuple(dim)

    def get_view_ndim(self):
        """ Always returns the ndim of the current view
        """
        return len(self.maps[self.view]['X_map'])
    
    def get_view_size(self):
        """ Always returns the size of the current view
        """
        return reduce(operator.mul, self.get_view_shape(), 1)

    def set_active_view(self, view):
        """ Set a view among the ones in ``self.maps``.
        
        :param str view: name of the view to be set as active
        """
        self.view = view
        self.update_shape()
        
    def set_view(self, view, X_map, tol=None):
        """ Set or add a view to ``self.maps``. This resest all the existing reshape parameters in existing views.
        
        :param str view: name of the view to be added
        :param list X_map: list of coordinates of the new view
        :param float tol: tolerance for the matching of coordinates
        """
        if tol == None: tol = np.spacing(1)
        idx_map = []
        for d, x in enumerate(X_map):
            if any(x[i] > x[i+1] for i in range( len(x)-1 )):
                raise ValueError("TensorWrapperView: the input coordinates must be sorted")
            
            idx_map.append([])
            j = 0
            for val in x:
                while j < len(self.X[d]):
                    if abs( val - self.X[d][j] ) <= tol :
                        idx_map[-1].append( j )
                        break
                    j += 1
                if j == len(self.X[d]):
                    raise ValueError("TensorWrapperView: the input coordinates are not a subset of the full coordinates")
        self.maps[view] = { 'X_map': X_map, 
                            'idx_map': idx_map,
                            'Q': None,
                            'ghost_shape': None,
                            'fix_idxs': [],
                            'fix_dims': []}

    def refine(self, X_new, tol=None):
        """ Refine the global discretization. The new discretization must contain the old one.

        This function takes care of updating all the indices in the global view as well in all the other views.
        
        :param list X_new: list of coordinates of the new refinement
        :param float tol: tolerance for the matching of coordinates

        .. warning:: Any existing reshaping of the views is discarded.
        
        """
        if tol == None: tol = np.spacing(1)
        top_map = []            # Map from the old full coord to X_new
        for d, x in enumerate(X_new):
            if any(x[i] > x[i+1] for i in range( len(x)-1 )):
                raise ValueError("TensorWrapperView: the input coordinates must be sorted")
            
            top_map.append([])
            j = 0
            for val in self.X[d]:
                while j < len(x):
                    if abs( val - x[j] ) <= tol:
                        top_map[-1].append( j )
                        break
                    j += 1
                if j == len(x):
                    raise ValueError("TensorWrapperView: the full coordinates are not a subset of the new coordinates")
            
        # Update all keys in data
        new_data = {}
        while len(self.data) > 0:
            (old_key, value) = self.data.popitem()
            new_key = tuple( [ top_map[i][k] for i,k in enumerate(old_key) ] )
            new_data[new_key] = value
        self.data = new_data
        
        # Update the coordinates
        self.X = X_new

        # Update all the views
        self.maps['full'] = { 'X_map': self.X, 
                              'idx_map': [ range( len(x) ) for x in self.X ],
                              'Q': None,
                              'ghost_shape': None,
                              'fix_idxs': [],
                              'fix_dims': [] }
        for view in self.maps:
            if view != 'full':
                self.set_view( view, self.maps[view]['X_map'] )

    #########################
    # EXTENDED for Quantics #
    #########################
    def get_extended_shape(self):
        """ If the quantics folding has been performed on the current view, then this returns the shape of the extended tensor to the next power of Q. If the folding has not been performed, this returns the view shape.
        """
        if self.maps[self.view]['Q'] == None:
            return self.get_view_shape()
        else:
            return tuple( [ self.maps[self.view]['Q']**(int(math.log(s-0.5,self.maps[self.view]['Q']))+1) for s in self.get_view_shape() ] )

    def get_extended_ndim(self):
        """ If the quantics folding has been performed on the current view, then this returns the number of dimensions of the extended tensor to the next power of Q. If the folding has not been performed, this returns an error.
        """
        return len(self.get_extended_shape())
    
    def get_extended_size(self):
        """ If the quantics folding has been performed on the current view, then this returns the size of the extended tensor to the next power of Q. If the folding has not been performed, this returns an error.
        """
        return reduce(operator.mul, self.get_extended_shape())
    
    def set_Q(self, Q):
        """ Set the quantics folding base for the current view.

        This will unset any fixed index for the current view set using :py:meth:`~TensorWrapper.fix_indices`.
        
        :param int Q: folding base.
        """
        self.maps[self.view]['Q'] = Q
        self.maps[self.view]['ghost_shape'] = [ Q**(int(math.log(s-0.5, Q))+1) for s in self.get_view_shape() ]
        self.maps[self.view]['fix_idxs'] = []
        self.maps[self.view]['fix_dims'] = []
        self.update_shape()

    def reset_shape(self):
        """ Reset the shape of the tensor erasing the reshape and quantics foldings.
        """
        self.maps[self.view]['Q'] = None
        self.maps[self.view]['ghost_shape'] = None
        self.update_shape()

    #########
    # GHOST #
    #########
    def get_ghost_shape(self):
        """ If the ``ghost_shape`` is set for this view, then it returns the shape obtained after quantics folding by the function :py:meth:`~TensorWrapper.set_Q` or after reshaping by the function :py:meth:`~TensorWrapper.reshape`. Otherwise the shape of the extended shape is returned.
        """
        if self.maps[self.view]['ghost_shape'] != None:
            return self.maps[self.view]['ghost_shape']
        else:
            return self.get_extended_shape()
    
    def get_ghost_ndim(self):
        """ If the ``ghost_shape`` is set for this view, then it returns the number of dimensions obtained after quantics folding by the function :py:meth:`~TensorWrapper.set_Q` or after reshaping by the function :py:meth:`~TensorWrapper.reshape`. Otherwise the number of dimensions of the view is returned.
        """
        return len(self.get_ghost_shape())

    def get_ghost_size(self):
        """ If the ``ghost_shape`` is set for this view, then it returns the size obtained after quantics folding by the function :py:meth:`~TensorWrapper.set_Q` or after reshaping by the function :py:meth:`~TensorWrapper.reshape`. Otherwise the size of the view is returned.
        """
        return reduce(operator.mul, self.get_ghost_shape())

    def reset_ghost_shape(self):
        """ Reset the shape of the tensor erasing the reshape and quantics foldings.
        """
        self.maps[self.view]['ghost_shape'] = None
        self.update_shape()
    
    def reshape(self,newshape):
        """ Reshape the tensor. The number of items in the new shape must be consistent with :py:meth:`~TensorWrapper.get_extended_size`, i.e. with the number of items in the extended quantics size or the view size if ``Q`` is not set for this view.

        This will unset any fixed index for the current view set using :py:meth:`~TensorWrapper.fix_indices`.

        :param list newshape: new shape to be applied to the tensor.
        """
        if reduce(operator.mul, newshape, 1) == self.get_extended_size():
            self.maps[self.view]['ghost_shape'] = tuple(newshape)
            self.maps[self.view]['fix_idxs'] = []
            self.maps[self.view]['fix_dims'] = []
            self.update_shape()
            return self
        else:
            raise ValueError('TensorWrapper.reshape: total size of new tensor must be unchanged')
    
    ###########
    # FIX_IDX #
    ###########
    def get_shape(self):
        """ Always returns the shape of the actual tensor view
        
        .. note: use :py:meth:`TensorWrapper.get_global_shape` to get the shape of the original tensor
        """
        return tuple([ s for dim,s in enumerate(self.get_ghost_shape()) if not (dim in self.maps[self.view]['fix_dims']) ])
    
    def get_ndim(self):
        """ Always returns the number of dimensions of the tensor view
        
        .. note: use :py:meth:`TensorWrapper.get_global_ndim` to get the number of dimensions of the original tensor
        """
        return len(self.get_shape())

    def get_size(self):
        """ Always returns the size of the tensor view
        
        .. note: use :py:meth:`TensorWrapper.get_global_size` to get the size of the original tensor
        """
        return reduce(operator.mul, self.get_shape(), 1)
    
    def update_shape(self):
        self.shape = self.get_shape()
        self.size = self.get_size()
        self.ndim = self.get_ndim()
    
    def fix_indices(self, idxs, dims):
        """ Fix some of the indices in the tensor wrapper and reshape/resize it accordingly. The internal storage of the data is still done with respect to the global indices, but once some indices are fixed, the TensorWrapper can be accessed using just the remaining free indices.
        
        :param list idxs: list of indices to be fixed
        :param list dims: list of dimensions to which the indices refer to
        
        .. note: ``len(idxs) == len(dims)``
        """
        if len(idxs) != len(dims):
            raise AttributeError("TensorToolbox.TensorWrapper.fix_indices: len(idxs) == len(dims) violated")
        if len(dims) != len(set(dims)):
            raise AttributeError("TensorToolbox.TensorWrapper.fix_indices: the list of dimensions must contain unique entries only.")
        
        # Reorder the lists
        i_ord = sorted(range(len(dims)), key=dims.__getitem__)
        self.maps[self.view]['fix_idxs'] = [ idxs[i] for i in i_ord ]
        self.maps[self.view]['fix_dims'] = [ dims[i] for i in i_ord ]
        # Update shape, ndim and size
        self.update_shape()
    
    def release_indices(self):
        """ Release all the indices in the tensor wrapper which were fixed using :py:meth:`~TensorWrapper.fix_indices`.
        """
        self.maps[self.view]['fix_idxs'] = []
        self.maps[self.view]['fix_dims'] = []
        self.update_shape()

    #####################################################
    #            INDEX TRANSFORMATIONS                  #
    #####################################################

    ###################
    # GLOBAL to SHAPE #
    ###################
    def global_to_view(self, idxs):
        """ This maps the index from the global shape to the view shape.
        
        :param tuple idxs: tuple representing an index to be transformed.
        
        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        .. note:: this returns an error if the ``idxs`` do not belong to the index mapping of the current view.
        """
        return tuple( [ self.idx_max[d].index(i) for d,i in enumerate(idxs) ] )
    
    def view_to_ghost(self, idxs):
        """ This maps the index from the view to the ghost shape.
        
        :param list idxs: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        .. note:: this returns an error if the ghost shape is obtained by quantics folding, because the one view index can be pointing to many indices in the folding.
        """
        if self.maps[self.view]['Q'] != None:
            raise NotImplemented("This operation is undefined because one view idx can point to many q indices")
        else:
            return idxfold( self.get_ghost_shape(), idxunfold( self.get_view_shape(), idxs ) )
    
    def global_to_ghost(self,idxs):
        """ This maps the index from the global shape to the ghost shape.
        
        :param list idxs: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.

        For :py:meth:`TensorWrapper` ``A``, this corresponds to:
        
        >>> A.view_to_ghost( A.global_to_view( idxs ) )

        """
        return self.view_to_ghost( self.global_to_view( idxs ) )

    ###################
    # SHAPE to GLOBAL #
    ###################
    def shape_to_ghost(self, idxs_in):
        """ This maps the index from the current shape of the view (fixed indices) to the ghost shape.
        
        :param list idxs_in: list of indices to be transformed

        .. note:: slicing is admitted here.
        """
        idxs = idxs_in[:]
        # Insert the fixed indices
        for i in self.maps[self.view]['fix_dims']:
            idxs.insert(i, self.maps[self.view]['fix_idxs'][ self.maps[self.view]['fix_dims'].index(i)] )
        return idxs
    
    def ghost_to_extended(self, idxs):
        return idxfold( self.get_extended_shape(), idxunfold( self.get_ghost_shape(), idxs ) )

    def ghost_to_view(self, idxs):
        """ This maps the index from the current ghost shape of the view to the view shape.
        
        :param list idxs_in: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        """
        if self.maps[self.view]['Q'] != None:
            return tuple( [ ( i if i<N else N-1 ) for i,N in zip(self.ghost_to_extended(idxs),self.get_view_shape()) ] )
        else:
            idxs = idxfold( self.get_view_shape(), idxunfold( self.get_ghost_shape(), idxs ) )
            return tuple( idxs )
    
    def shape_to_view(self, idxs):
        """ This maps the index from the current shape of the view to the view shape.
        
        :param list idxs_in: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        """
        return self.ghost_to_view( self.shape_to_ghost( idxs ) )

    def view_to_global(self, idxs):
        """ This maps the index in view to the global indices of the full tensor wrapper.
        
        :param list idxs_in: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        """
        return tuple( [ self.maps[self.view]['idx_map'][d][i] for d,i in enumerate(idxs) ] )

    def ghost_to_global(self, idxs):
        """ This maps the index from the current ghost shape of the view to the global shape.
        
        :param list idxs_in: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        """
        return self.view_to_global( self.ghost_to_view( idxs ) )
    
    def shape_to_global(self, idxs):
        """ This maps the index from the current shape of the view to the global shape.
        
        :param list idxs_in: list of indices to be transformed

        .. note:: no slicing is admitted here. Preprocess ``idxs`` with :py:meth:`expand_idxs` if slicing is required.
        """
        return self.view_to_global( self.ghost_to_view( self.shape_to_ghost( idxs ) ) )

    ######################
    # CHECKS ON EXTENDED #
    ######################
    def extended_is_view(self, idxs):
        """ 
        :return: True if the idxs is in the view shape. False if it is outside
        """
        return all( [ i<N for i,N in zip(idxs,self.get_view_shape()) ] )
    
    def full_is_view(self,idxs):
        return self.extended_is_view( self.ghost_to_extended( idxs ) )

    ###############################################
    #            DATA AND FUNCTIONS               #
    ###############################################
    def get_fill_level(self):
        if self.twtype == 'view': return 0
        else: return len(self.data)
        
    def get_fill_idxs(self):
        return self.data.keys()
    
    def get_data(self):
        return self.data
    
    def get_X(self):
        return self.X

    def get_params(self):
        return self.params
    
    def set_f(self,f, marshal_f):
        self.f = f
        if self.f != None and marshal_f:
            self.f_code = marshal.dumps(self.f.__code__)
    
    def reset_f_marshal(self):
        if self.f_code != None:
            code = marshal.loads(self.f_code)
            self.f = types.FunctionType(code, globals(), "f")
        else:
            warnings.warn(
                "TensorToolbox.TensorWrapper: The tensor wrapper has not function " + \
                "code to un-marshal. The function is undefined. Define it using " + \
                "TensorToolbox.TensorWrapper.set_f", RuntimeWarning)
    
    def set_params(self, params):
        self.params = params
    
    def set_maxprocs(self,maxprocs):
        self.__maxprocs = maxprocs
        try:
            import mpi_map
            MPI_SUPPORT = True
        except ImportError:
            MPI_SUPPORT = False
        
        if self.__maxprocs != None and not MPI_SUPPORT:
            warnings.warn("TensorToolbox.TensorWrapper: MPI is not supported on this " + \
                          "machine. The program will run without it.", RuntimeWarning)
        
    def set_store_object(self, store_object):
        self.store_object = store_object

    def __getitem__(self,idxs_in):
        
        (lidxs,out_shape,transpose_list_shape) = expand_idxs(
            idxs_in, self.shape, self.get_ghost_shape(),
            self.maps[self.view]['fix_dims'], self.maps[self.view]['fix_idxs'])
        
        # Allocate output array
        if len(out_shape) > 0:
            out = np.empty(out_shape, dtype=self.dtype)
            if self.active_weights:
                out_weights = np.empty(out_shape, dtype=self.dtype)
            
            # MPI code
            eval_is =[]
            eval_idxs = []
            eval_xx = []
            # End MPI code

            for i,idxs in enumerate(lidxs):

                # Map ghost indices to global indices
                idxs = self.ghost_to_global( idxs )
                
                # Compute the weight corresponding to idxs
                if self.active_weights:
                    out_weights[idxfold(out_shape,i)] = np.prod([self.W[j][jj] for j,jj in enumerate(idxs)])
                
                # Separate field idxs from parameter idxs                
                if self.twtype == 'array':
                    # Check whether the value has already been computed
                    try:
                        out[idxfold(out_shape,i)] = self.data[idxs]
                    except KeyError:
                        if idxs not in eval_idxs:
                            # Evaluate function
                            xx = np.array( [self.X[ii][idx] for ii,idx in enumerate(idxs)] )
                            # MPI code
                            eval_is.append([i])
                            eval_idxs.append(idxs)
                            eval_xx.append(xx)
                            # End MPI code
                        else:
                            pos = eval_idxs.index(idxs)
                            eval_is[pos].append(i)

                else:
                    # Evaluate function
                    xx = np.array([self.X[ii][idx] for ii,idx in enumerate(idxs)])
                    out[idxfold(out_shape,i)] = self.f(xx,self.params)

                # # Check that the idxs belong to the real tensor
                # isout_flag = not self.full_is_view( idxs )
                #
                # if isout_flag:
                #     out[idxfold(out_shape,i)] = TensorWrapper.FILL_VALUE
                # else:
                #     # Map ghost indices to global indices
                #     idxs = self.full_to_global( idxs )
				# 
                #     # Separate field idxs from parameter idxs                
                #     if self.twtype == 'array':
                #         # Check whether the value has already been computed
                #         try:
                #             out[idxfold(out_shape,i)] = self.data[idxs]
                #         except KeyError:
                #             if idxs not in eval_idxs:
                #                 # Evaluate function
                #                 xx = np.array( [self.X[ii][idx] for ii,idx in enumerate(idxs)] )
                #                 # MPI code
                #                 eval_is.append([i])
                #                 eval_idxs.append(idxs)
                #                 eval_xx.append(xx)
                #                 # End MPI code
                #             else:
                #                 pos = eval_idxs.index(idxs)
                #                 eval_is[pos].append(i)
				# 
                #     else:
                #         # Evaluate function
                #         xx = np.array([self.X[ii][idx] for ii,idx in enumerate(idxs)])
                #         out[idxfold(out_shape,i)] = self.f(xx,self.params)
            
            # Evaluate missing values
            if len(eval_xx) > 0:
                self.logger.debug(" [START] Num. of func. eval.: %d " % len(eval_xx))
                start_eval = time.time()
                if self.__maxprocs == None or not MPI_SUPPORT:
                    if self.ftype == 'serial':
                        # Serial evaluation
                        for (ii,idxs,xx) in zip(eval_is, eval_idxs, eval_xx):
                            self.data[idxs] = self.f(xx,self.params)
                            self.store()
                            for i in ii:
                                out[idxfold(out_shape,i)] = self.data[idxs]
                    elif self.ftype == 'vector':
                        # Vectorized evaluation
                        eval_xx_mat = np.vstack(eval_xx)
                        data_mat = self.f(eval_xx_mat, self.params)
                        for j, (ii,idxs) in enumerate(zip(eval_is, eval_idxs)):
                            self.data[idxs] = data_mat[j]
                            for i in ii:
                                out[idxfold(out_shape,i)] = self.data[idxs]
                        self.store()
                else:
                    # MPI code
                    eval_res = mpi_map.mpi_map_code( self.f_code, eval_xx, self.params, self.__maxprocs )
                    for (ii,idxs,res) in zip(eval_is, eval_idxs, eval_res):
                        self.data[idxs] = res
                        for i in ii:
                            out[idxfold(out_shape,i)] = self.data[idxs]
                    self.store()
                    # End MPI code
                stop_eval = time.time()
                self.logger.debug(" [DONE] Num. of func. eval.: %d - Avg. time of func. eval.: %fs - Tot. time: %s" % (len(eval_xx),(stop_eval-start_eval)/len(eval_xx)*(min(self.__maxprocs,len(eval_xx)) if self.__maxprocs != None else 1), str(datetime.timedelta(seconds=(stop_eval-start_eval))) ))
            
            # Apply weights if needed
            if self.active_weights:
                out *= out_weights
            
            if transpose_list_shape:
                out = np.transpose( out , tuple( list(range(1,len(out_shape))) + [0] ) )
            
        else:
            idxs = tuple(itertools.chain(*lidxs))
            # Map ghost indices to global indices
            idxs = self.ghost_to_global( idxs )
            # Compute weight if necessary
            if self.active_weights:
                w = np.prod([self.W[j][jj] for j,jj in enumerate(idxs)])
            if self.twtype == 'array':
                try:
                    out = self.data[idxs]
                except KeyError:
                    # Evaluate function
                    xx = np.array([self.X[ii][idx] for ii,idx in enumerate(idxs)])
                    self.data[idxs] = self.f(xx,self.params)
                    self.store()
                    out = self.data[idxs]
            else:
                out = self.f(np.array([self.X[ii][idx] for ii,idx in enumerate(idxs)]),self.params)
            # Apply the weight if necessary
            if self.active_weights:
                out *= w
        
        return out
