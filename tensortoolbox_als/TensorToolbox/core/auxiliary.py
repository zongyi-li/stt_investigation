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

__all__ = ['NumericsError','ConvergenceError','TTcrossLoopError',
           'idxunfold','idxfold','expand_idxs',
           'matkron_to_mattensor','mat_to_tt_idxs','tt_to_mat_idxs',
           'maxvol','lowrankapprox','reort',
           'isint', 'isfloat']

import operator
import itertools
try:
    import itertools.izip as zip
except ImportError:
    pass
import random
import numpy as np
import numpy.linalg as npla
import scipy.linalg as scla

class NumericsError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class ConvergenceError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

class TTcrossLoopError(Exception):
    def __init__(self, value):
        self.value = value
    def __str__(self):
        return repr(self.value)

def isint(a):
    return type(a) in [int, np.int32, np.int64]

def isfloat(a):
    return type(a) in [float, np.float32, np.float64]

def matkron_to_mattensor(A,nrows,ncols):
    """ This function reshapes a 2D-matrix obtained as kron product of len(nrows)==len(ncols) matrices, to a len(nrows)-tensor that can be used as input for the TTmat constructor. Applies the Van Loan-Pitsianis reordering of the matrix elements.
    
    :param ndarray A: 2-dimensional matrix
    :param list,int nrows,ncols: number of rows and number of columns of the original matrices used for the kron product
    """
    
    if A.ndim != 2: raise NameError("TensorToolbox.core.matkron_to_mattensor: The input ndarray has the wrong number of dimension. 2d-ndarray required.")
    if A.shape[0] != np.prod(nrows) or A.shape[1] != np.prod(ncols): raise NameError("TensorToolbox.core.matkron_to_mattensor: dimension of A not consistent with nrows or ncols")

    if isint(nrows) and isint(ncols):
        nrows = [nrows]
        ncols = [ncols]

    if len(nrows) != len(ncols): raise NameError("TensorToolbox.core.matkron_to_mattensor: len(nrows)!=len(ncols)")
    
    d = len(nrows)
    dims = list(nrows)
    dims.extend(ncols)

    # Prepare interleaved idxs
    idxs = [[i,d+i] for i in range(d)]
    idxs = list(itertools.chain(*idxs))

    # Reshape and re-order
    A = A.reshape(dims)
    A = np.transpose(A,axes=idxs)
    A = A.reshape([nrows[i]*ncols[i] for i in range(d)])
    
    return A

def idxunfold(dlist,idxs):
    """ Find the index corresponding to the unfolded (flat) version of a tensor
    
    :param list,int dlist: list of integers containing the dimensions of the tensor
    :param list,int idxs: tensor index
    
    :returns: index for the flatten tensor
    """
    
    # Check whether index out of bounds
    import operator
    if any(map(operator.ge,idxs,dlist)):
        raise NameError("TensorToolbox.core.idxunfold: Index out of bounds")
    
    n = len(dlist)

    plist = [1]
    for i in range(n-1,0,-1):
        plist.append( plist[-1] * dlist[i] )

    ii = sum( [ p * int(idxs[i]) for i,p in enumerate(reversed(plist)) ] )

    return ii

def idxfold(dlist,idx):
    """ Find the index corresponding to the folded version of a tensor from the flatten version

    :param list,int dlist: list of integers containing the dimensions of the tensor
    :param int idx: tensor flatten index

    :returns: list of int -- the index for the folded version

    :note: this routine can be used to get the indexes of a TTmat from indices of a matkron (matrix obtained using np.kron): (i,j) \in N^d x N^d -> ((i_1,..,i_d),(j_1,..,j_d)) \in (N x .. x N) x (N x .. x N)

    """
    n = len(dlist)

    cc = [1]
    for val in reversed(dlist): cc.append( cc[-1] * val )
    if idx >= cc[-1]: raise NameError("TensorToolbox.core.idxunfold: Index out of bounds")

    ii = []
    tmp = idx

    for i in range(n):
        ii.append( tmp//cc[n-i-1] )
        tmp = tmp % cc[n-i-1]

    return tuple(ii)


def expand_idxs(idxs_in,shape,ghost_shape=None,fix_dims=None,fix_idxs=None):
    """ From a tuple of indicies, apply all the unslicing transformations and restore the fixed indices in order to extract values from a tensor with a certain ``ghost_shape``. If ``ghost_shape==None``, ``fix_dims==None`` and ``fix_idxs==None``, then this performs only an unslicing of the index and it is assumed ``ghost_shape=shape``.
    
    :param tuple idxs_in: indexing tuple. The admissible slicing format is the same used in np.ndarray.
    :param tuple shape: shape of the tensor
    :param tuple ghost_shape: shape of the tensor without fixed indices
    :param list fix_dims: whether there are dimensions which had been fixed and need to be added.
    :param list fix_idxs: fixed idxs for each fixed dimension.
    
    :return: tuple ``(lidxs,out_shape,transpose_list_shape)``. ``lidxs`` is an iterator of the indices. ``out_shape`` is a tuple containing the shape of the output tensor. ``transpose_list_shape`` is a flag indicating whether the output format need to be transposed (behaving accordingly to np.ndarray).
    """
    if (ghost_shape == None and fix_dims == None and fix_idxs == None ):
        ghost_shape = shape
        fix_dims = []
        fix_idxs = []
    elif (ghost_shape == None or fix_dims == None or fix_idxs == None ):
        raise ValueError('If any of the ghost_shape, fix_dims, and fix_idxs are specified, then the other must be specified too')

    # Transform the tuple to a list for convinience
    idxs_in = list(idxs_in)

    # Slice notation can be used. Remember: slice(start:stop:step)
    if len(idxs_in) != len(shape):
        raise IndexError('wrong number of indices')

    # Check that all the lists are of the same length
    int_idx = []
    llen = None
    for i,idx in enumerate(idxs_in):
        if isint(idx):
            int_idx.append(i)
        if isinstance(idx, list) or isinstance(idx,tuple):
            idxs_in[i] = list(idx)
            if llen == None:
                llen = len(idx)
            elif llen != len(idx):
                raise IndexError('List of indices must have the same length.')

    if llen == None: llen = 1

    # Expand single indices in idxs_in to llen
    for i in int_idx: idxs_in[i] = [idxs_in[i]] * llen

    # Update input indices of slices and lists
    list_idx_in = []
    slice_idx_in = []
    for i,idx in enumerate(idxs_in):
        if isinstance(idx, list) or isinstance(idx,tuple):
            list_idx_in.append(i)
        if isinstance(idx, slice):
            slice_idx_in.append(i)

    # Insert fixed indices
    for i in fix_dims:
        idxs_in.insert(i, [fix_idxs[fix_dims.index(i)]] * llen)

    # Construct list of indices which are lists and slices
    list_idx = []
    list_IDXs = []
    slice_idx = []
    slice_IDXs = []
    out_shape = []
    for i,idx in enumerate(idxs_in):
        if isinstance(idx, list) or isinstance(idx,tuple):
            list_idx.append(i)
            list_IDXs.append( idx )
        if isinstance(idx, slice):
            slice_idx.append(i)
            IDXs = range(idx.start if idx.start != None else 0,
                             idx.stop  if idx.stop  != None else ghost_shape[i],
                             idx.step  if idx.step  != None else 1)
            slice_IDXs.append( IDXs )
            out_shape.append(len(IDXs))

    if len(list_idx) == 0: list_IDXs.append( [-1] ) # Ghost element added to make the full slicing work
    unlistIdxs = zip(*list_IDXs)

    transpose_list_shape = False
    if llen > 1: 
        out_shape.insert(0,llen)
        if len(slice_idx_in) > 0 and len(list_idx_in) > 0 and min(list_idx_in) > max(slice_idx_in):
            transpose_list_shape = True

    # Un-slice sliced idxs
    unslicedIdxs = itertools.product(*slice_IDXs)

    # Final list of indices (iterator)
    lidxs_iter = itertools.product(unlistIdxs, unslicedIdxs)
    
    lidxs = []
    for i,(lidx,sidx) in enumerate(lidxs_iter):
        # Reorder the idxs
        idxs = [None for j in range(len(list_idx)+len(slice_idx))] 
        for j,jj in enumerate(list_idx): idxs[jj] = lidx[j]
        for j,jj in enumerate(slice_idx): idxs[jj] = sidx[j]
        lidxs.append( tuple(idxs) )
    
    return (lidxs,out_shape,transpose_list_shape)


def mat_to_tt_idxs(rowidxs,colidxs,nrows,ncols):
    """ Mapping from the multidimensional matrix indexing to the tt matrix indexing

    (rowidxs,colidxs) = ((i_1,...,i_d),(j_1,...,j_d)) -> (l_1,...,l_d)
    
    :param tuple,int rowidxs,colidxs: list of row and column indicies. len(rowidxs) == len(colidxs)
    :param tuple,int nrows,ncols: dimensions of matrices

    :returns: tuple,int indices in the tt format
    """
    
    if isint(rowidxs): rowidxs = (rowidxs,)
    if isint(colidxs): colidxs = (colidxs,)
    if isint(nrows): nrows = (nrows,)
    if isint(ncols): ncols = (ncols,)

    if not (len(rowidxs) == len(colidxs) == len(nrows) == len(ncols)):
        raise NameError("TensorToolbox.core.mat_to_tt_idxs: not consistent dimensions in the input arguments")
    
    import operator
    if any(map(operator.ge,rowidxs,nrows)) or any(map(operator.ge,colidxs,ncols)):
        raise NameError("TensorToolbox.core.mat_to_tt_idxs: Index out of bound")
    
    return tuple(np.asarray(rowidxs) * np.asarray(ncols) + np.asarray(colidxs))

def tt_to_mat_idxs(idxs,nrows,ncols):
    """ Mapping from the tt matrix indexing to the multidimensional matrix indexing

     (l_1,...,l_d) -> (rowidxs,colidxs) = ((i_1,...,i_d),(j_1,...,j_d))
    
    :param tuple,int idxs: list of tt indicies. len(idxs) == len(nrows) == len(ncols)
    :param tuple,int nrows,ncols: dimensions of matrices

    :returns: (rowidxs,colidxs) = ((i_1,..,i_d),(j_1,..,j_d)) indices in the matrix indexing
    """
    
    if isint(idxs): idxs = (idxs,)
    if isint(nrows): nrows = (nrows,)
    if isint(ncols): ncols = (ncols,)

    if not (len(idxs) == len(nrows) == len(ncols)):
        raise NameError("TensorToolbox.core.tt_to_mat_idxs: not consistent dimensions in the input arguments")
    
    import operator
    if any(map(operator.ge,idxs,np.asarray(nrows)*np.asarray(ncols))):
        raise NameError("TensorToolbox.core.tt_to_mat_idxs: Index out of bound")

    return (tuple(np.asarray(idxs) // np.asarray(ncols)),tuple(np.asarray(idxs) % np.asarray(ncols)))

def maxvol(A,delta=1e-2,maxit=100):
    """ Find the rxr submatrix of maximal volume in A(nxr), n>=r

    :param ndarray A: two dimensional array with (n,r)=shape(A) where r<=n
    :param float delta: stopping cirterion [default=1e-2]
    :param int maxit: maximum number of iterations [default=100]

    :returns: ``(I,AsqInv,it)`` where ``I`` is the list or rows of A forming the matrix with maximal volume, ``AsqInv`` is the inverse of the matrix with maximal volume and ``it`` is the number of iterations to convergence 

    :raises: raise exception if the dimension of A is r>n or if A is singular
    :raises: ConvergenceError if convergence is not reached in maxit iterations
    """
    
    (n,r) = A.shape

    if r>n :
        raise TypeError("TensorToolbox.core.maxvol: A(nxr) must be a thin matrix, i.e. n>=r")
    
    # Find an arbitrary non-singular rxr matrix in A
    (P,L,U) = scla.lu(A)
    # Check singularity 
    if np.min(np.abs(np.diag(U))) < np.spacing(1):
        raise NumericsError("TensorToolbox.core.maxvol: Matrix A is singular")
    
    # Reorder A so that the non-singular matrix is on top
    I = np.arange(n,dtype=int) # set of swapping indices
    I = np.dot(P.astype(int).T,I)
    # Select Asq the top square matrix
    Asq = A[I[:r],:]
    
    # Compute inverse of Asq: Asq^-1 = (PLU)^-1
    LU = L[:r,:r] - np.eye(r) + U
    AsqInv = scla.lu_solve((LU,np.arange(r)), np.eye(r))
    # Compute B
    B = np.dot(A[I,:],AsqInv)
    
    # Find maximum and row
    maxidx = np.argmax(np.abs(B))
    maxB = np.abs(B).flatten()[maxidx]
    (maxrow,maxcol) = (maxidx // r, maxidx % r)
    
    it = 0
    eps = 1.+ delta
    while it < maxit and maxB > eps:
        it += 1

        # Update AsqInv
        q = np.zeros((r,1),dtype=np.float64)
        q[maxcol] = 1.
        vT = A[[I[maxrow],],:] - A[[I[maxcol],],:]
        AsqInv -= np.dot(np.dot(AsqInv,q), np.dot(vT,AsqInv)) / (1. + np.dot(vT,np.dot(AsqInv,q))) # Eq (8) in "How to find a good submatrix"
        
        # Update B using Sherman-Woodbury-Morrison formula
        Bj = B[:,[maxcol,]]
        # Bj[maxcol,0] -= 1.
        Bj[maxrow,0] += 1.
        Bi = B[[maxrow,],:]
        Bi[0,maxcol] -= 1.
        B[r:,:] -= np.dot(Bj[r:],Bi)/B[maxrow,maxcol]

        # Update index of maxvol matrix I
        tmp = I[maxcol]
        I[maxcol] = I[maxrow]
        I[maxrow] = tmp

        # # Manual update TO BE REMOVED
        # AsqInv = npla.inv(A[I[:r],:])
        # B = np.dot(A[I,:], AsqInv)

        # Find new maximum in B
        maxidx = np.argmax(np.abs(B))
        maxB = np.abs(B).flatten()[maxidx]
        (maxrow,maxcol) = (maxidx // r, maxidx % r)

    if maxB > eps:
        raise ConvergenceError('Maxvol algorithm did not converge.')
    
    # Return max-vol submatrix Asq
    return ([np.asscalar(i) for _,i in np.ndenumerate(I[:r])],AsqInv,it)


def lowrankapprox(A, r, Jinit=None, delta=1e-5, maxit=100, maxvoleps=1e-2, maxvolit=100):
    """ Given a matrix A nxm, find the maximum volume submatrix with rank r<n,m.
    
    :param ndarray A: two dimensional array with dimension nxm
    :param int r: rank of the maxvol submatrix
    :param list Jinit: list of integers containing the r starting columns. If ``None`` then pick them randomly.
    :param float delta: accuracy parameter
    :param int maxit: maximum number of iterations in the lowrankapprox routine
    :param float maxvoleps: accuracy parameter for each usage of the maxvol algorithm
    :parma int maxvolit: maximum number of iterations in the maxvol routine
    
    :return: ``(I,J,AsqInv,it)`` where ``I`` and ``J`` are the list of rows and columns of A that compose the submatrix of maximal volume, ``AsqInv`` is the inverse of such matrix and ``it`` is the number of iteration to convergence
    """

    import random
    
    (n,m) = A.shape
    if r>n or r>m:
        raise AttributeError('Rank r bigger than shape of A')
    
    # Pick first column indices J at random
    if Jinit == None:
        J = random.sample(range(m),r)
    else:
        J = Jinit
    if len(J) != r:
        raise AttributeError('Invalid number of init columns: len(J) != r')
    J.sort()
    Aold = np.ones((n,m))
    Anew = np.zeros((n,m))
    it = 0
    while it < maxit and npla.norm( Anew-Aold, 'fro' ) > delta * npla.norm(Anew,'fro'):
        it += 1
        # Row cycle
        R = A[:,J]
        if r == 1: R = np.reshape(R,(len(R),1))
        # QR decomposition
        (Q,R) = scla.qr(R,mode='economic')
        # Maxvol
        (I,QsqInv,it) = maxvol(Q,maxvoleps,maxvolit)
        # Column cycle
        C = A[I,:].T
        if r == 1: C = np.reshape(C,(len(C),1))
        # QR decomposition
        (Q,R) = scla.qr(C,mode='economic')
        # Maxvol
        (J,QsqInv,it) = maxvol(Q,maxvoleps,maxvolit)
        # New approximation
        Aold = Anew
        Atmp = A[:,J]
        if r == 1: Atmp = np.reshape(Atmp,(len(Atmp),1))
        Anew = np.dot(Atmp, np.dot(Q, QsqInv).T)
    
    if npla.norm( Anew-Aold, 'fro' ) > delta * npla.norm(Anew,'fro'):
        raise ConvergenceError('Low Rank Approximation algorithm did not converge.')

    if np.min(np.abs(np.diag(R))) <= np.spacing(1):
        raise ConvergenceError('Low Rank Approximation algorithm converged to a singular solution. Rank r over-estimated.')
    # Compute AsqInv
    AsqInv = scla.solve_triangular(R,QsqInv).T
    
    return (I,J,AsqInv,it)

def reort(u,uadd):
    """ Golub-Kahan reorthogonalization
    
    .. note: See Oseledets' TT-Toolbox
    """
    if uadd.shape[1] == 0 or u.shape[0] == u.shape[1]:
        return u
    
    if u.shape[1] + uadd.shape[1] >= u.shape[0]:
        uadd = uadd[:, :u.shape[0]-u.shape[1]]
    
    radd = uadd.shape[1]
    
    mvr = np.dot( u.T, uadd)
    unew = uadd - np.dot( u, mvr )
    reort_flag = True
    while reort_flag:
        reort_flag = False
        j = 1
        norm_unew = np.sum(unew**2., axis=0)
        norm_uadd = np.sum(uadd**2., axis=0)
        reort_flag = (0 < len([ True for (nn,na) in itertools.product(norm_unew,norm_uadd) if nn <= .25 * na]))
        [unew,_] = scla.qr(unew,mode='economic')
        if reort_flag:
            su = np.dot( u.T, unew )
            uadd = unew.copy()
            unew -= np.dot(u, su)
    
    return np.hstack( (u,unew) )
            
    
