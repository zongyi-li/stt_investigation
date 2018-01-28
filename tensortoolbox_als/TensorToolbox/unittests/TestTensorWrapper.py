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

#
# DESCRIPTION
# 
# Construct a tensor wrapper around an numpy.ndarray and check that all the 
# slicing functions work.
# The test checks that the output of the tensor wrapper is equal to the output
# of the original array and that the number of function evaluations are consistent.
#

import sys
from TensorToolbox.unittests.auxiliary import bcolors, print_summary

def print_ok(testn,string):
    print(bcolors.OKGREEN + "[SUCCESS] " + str(testn) + " Tensor Wrapper: " + string + bcolors.ENDC)

def print_fail(testn,string,msg=''):
    print(bcolors.FAIL + "[FAILED] " + str(testn) + " Tensor Wrapper: " + string + bcolors.ENDC)
    if msg != '':
        print(bcolors.FAIL + msg + bcolors.ENDC)

feval = 0

def run(maxprocs, loglev=logging.WARNING):

    logging.basicConfig(level=loglev)

    import os
    import os.path
    import pickle as pkl
    import numpy as np
    import numpy.random as npr
    import math
    import random
    from functools import reduce
    import TensorToolbox as TT

    store_location = "tw"
    if os.path.isfile(store_location + ".pkl"):
        os.remove(store_location + ".pkl")
    if os.path.isfile(store_location + ".h5"):
        os.remove(store_location + ".h5")
    if os.path.isfile(store_location + ".pkl.old"):
        os.remove(store_location + ".pkl.old")
    if os.path.isfile(store_location + ".h5.old"):
        os.remove(store_location + ".h5.old")
    
    testn = 0

    ###################################################################
    # 00: Construction of the multidimensional array and the corresponding tensor wrapper
    #

    global feval
    global nsucc
    global nfail

    nsucc = 0
    nfail = 0

    shape = [2,3,4,5]
    d = len(shape)
    A = np.arange(np.prod(shape)).reshape(shape)
    Aglobal = A.copy()              # Used in f in order to test fix_indices

    def f(X, params):
        global feval
        feval += 1
        return Aglobal[tuple(X)]

    X = [ np.arange(s,dtype=int) for s in shape ]
    TW = TT.TensorWrapper(f, X, None,dtype=A.dtype,marshal_f=False)

    testn += 1
    print_ok(testn,"Construction")
    nsucc += 1

    def test(testn, title, idx):
        global feval
        global nsucc
        global nfail
        feval = 0
        TW.data = {}
        out = TW[idx]
        if np.any(A[idx].shape != out.shape) or (not np.allclose(A[idx], out, rtol=1e-10, atol=1e-12)):
            print_fail(testn,title,msg='Different output - idx: ' + str(idx))
            nfail += 1
        elif feval != np.prod(np.unique(A[idx]).shape):
            print_fail(testn,title,msg='Wrong number of function evaluations - idx: ' + str(idx))
            nfail += 1
        else:
            print_ok(testn,title)
            nsucc += 1

    ###################################################################
    # 01: Single address access
    #
    idx = (1,2,3,4)
    feval = 0
    out = TW[idx]
    testn += 1
    if A[idx] != out:
        print_fail(testn,"Single address access",msg='Different output')
        nfail += 1
    elif feval != 1:
        print_fail(testn,"Single address access",msg='Wrong number of function evaluations')
        nfail += 1
    else:
        print_ok(testn,"Single address access")
        nsucc += 1

    ###################################################################
    # Storage
    testn += 1
    TW.data = {}
    TW.store_location = store_location
    TW[:,:,:,0]
    TW.store(force=True)
    print_ok(testn, "Storage")
    nsucc += 1
    # Reload
    testn += 1
    TW = TT.load(store_location)
    TW.set_f(f,False)
    idx = tuple( [slice(None,None,None)]*d )
    test(testn,"Reload",idx)
    if os.path.isfile(store_location + ".pkl"):
        os.remove(store_location + ".pkl")
    if os.path.isfile(store_location + ".h5"):
        os.remove(store_location + ".h5")
    if os.path.isfile(store_location + ".pkl.old"):
        os.remove(store_location + ".pkl.old")
    if os.path.isfile(store_location + ".h5.old"):
        os.remove(store_location + ".h5.old")

    ###################################################################
    # Single slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,4)
    test(testn,"Single slice",idx)

    ###################################################################
    # Partial slice
    #
    testn += 1
    idx = (1,2,slice(1,3,1),4)
    test(testn,"Partial slice",idx)

    ###################################################################
    # Partial stepping slice
    #
    testn += 1
    idx = (1,2,slice(0,4,2),4)
    test(testn,"Partial stepping slice",idx)

    ###################################################################
    # Multiple slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,slice(0,4,2))
    test(testn,"Multiple slice",idx)

    ###################################################################
    # Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(shape))
    test(testn,"Full slice",idx)

    ###################################################################
    # List 
    #
    testn += 1
    idx = ([0,1],[1,2],[1,3],[0,4])
    test(testn,"Lists",idx)

    ###################################################################
    # Single list 
    #
    testn += 1
    idx = (0,1,[1,3],3)
    test(testn,"Single list",idx)

    ###################################################################
    # Double list 
    #
    testn += 1
    idx = (0,[0,2],[1,3],3)
    test(testn,"Double list",idx)

    ###################################################################
    # Single list slice
    #
    testn += 1
    idx = (0,[0,2],slice(None,None,None),3)
    test(testn,"Single list slice",idx)

    testn += 1
    idx = (0,slice(None,None,None),[0,2],3)
    test(testn,"Single list slice",idx)

    testn += 1
    idx = (slice(None,None,None),0,[0,2,3],3)
    test(testn,"Single list slice",idx)

    ###################################################################
    # Double list slice
    #
    testn += 1
    idx = ([0,1],slice(None,None,None),[0,2],3)
    test(testn,"Double list slice",idx)

    testn += 1
    idx = (slice(None,None,None),0,slice(None,None,None),[0,2,3])
    test(testn,"Double slice list",idx)

    ###################################################################
    # Lists slice
    #
    testn += 1
    idx = ([0,1],[0,2],slice(None,None,None),[1,3])
    test(testn,"Lists slice",idx)

    ###################################################################
    # Fix indices
    #
    testn += 1
    fix_idxs = [0,2]
    fix_dims = [0,2]
    Atmp = A.copy()
    A = A[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Fix indices", idx)

    ###################################################################
    # Release indices
    #
    testn += 1
    A = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Release indices", idx)

    ###################################################################
    # Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = A.copy()
    A = A[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Fix indices - second test", idx)

    ###################################################################
    # Release indices
    #
    testn += 1
    A = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Release indices", idx)

    ######################################################################################
    ######################################################################################
    ## Reshape function
    ##
    Atmp_shape = A.copy()             # Storing original array
    A = np.reshape( A, [5,4,3,2] )
    TW.reshape( [5,4,3,2] )

    ###################################################################
    # Reshaped - Single slice
    #
    testn += 1
    idx = (4,slice(None,None,None),2,1)
    test(testn,"Reshaped - Single slice",idx)

    ###################################################################
    # Reshaped - Partial slice
    #
    testn += 1
    idx = (4,slice(1,3,1),2,1)
    test(testn,"Reshaped - Partial slice",idx)

    ###################################################################
    # Reshaped - Partial stepping slice
    #
    testn += 1
    idx = (4,slice(0,4,2),2,1)
    test(testn,"Reshaped - Partial stepping slice",idx)

    ###################################################################
    # Reshaped - Multiple slice
    #
    testn += 1
    idx = (slice(0,4,2),3,slice(None,None,None),1)
    test(testn,"Reshaped - Multiple slice",idx)

    ###################################################################
    # Reshaped - Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(A.shape))
    test(testn,"Reshaped - Full slice",idx)

    ###################################################################
    # Reshaped - List 
    #
    testn += 1
    idx = ([0,4],[1,3],[1,2],[0,1])
    test(testn,"Reshaped - Lists",idx)

    ###################################################################
    # Reshaped - Single list 
    #
    testn += 1
    idx = (3,[1,3],1,0)
    test(testn,"Reshaped - Single list",idx)

    ###################################################################
    # Reshaped - Double list 
    #
    testn += 1
    idx = (3,[1,3],[0,2],0)
    test(testn,"Reshaped - Double list",idx)

    ###################################################################
    # Reshaped - Single list slice
    #
    testn += 1
    idx = (3,slice(None,None,None),[0,2],0)
    test(testn,"Reshaped - Single list slice",idx)

    testn += 1
    idx = (3,[0,2],slice(None,None,None),0)
    test(testn,"Reshaped - Single list slice",idx)

    testn += 1
    idx = (3,[0,2,3],0,slice(None,None,None))
    test(testn,"Reshaped - Single list slice",idx)

    ###################################################################
    # Reshaped - Double list slice
    #
    testn += 1
    idx = (3,[0,2],slice(None,None,None),[0,1])
    test(testn,"Reshaped - Double list slice",idx)

    testn += 1
    idx = (slice(None,None,None),0,slice(None,None,None),[0,1])
    test(testn,"Reshaped - Double slice list",idx)

    ###################################################################
    # Reshaped - Lists slice
    #
    testn += 1
    idx = ([0,1],[0,2],slice(None,None,None),[1,0])
    test(testn,"Reshaped - Lists slice",idx)

    ###################################################################
    # Reshaped - Fix indices
    #
    testn += 1
    fix_idxs = [0,2]
    fix_dims = [0,2]
    Atmp = A.copy()
    A = A[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped - Fix indices", idx)

    ###################################################################
    # Reshaped - Release indices
    #
    testn += 1
    A = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped - Release indices", idx)

    ###################################################################
    # Reshaped - Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = A.copy()
    A = A[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Reshaped - Fix indices - second test", idx)

    ###################################################################
    # Reshaped - Release indices
    #
    testn += 1
    A = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped - Release indices", idx)

    ###################################################################
    # Reshaped - Restore original shape
    #
    testn += 1
    A = Atmp_shape.copy()
    TW.reset_shape()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped - Restore original shape", idx)
    
    ##
    ## Shape restored
    #############################################################################
    #############################################################################

    #############################################################################
    #############################################################################
    ## Reshape function 2 
    ##
    shape = [2,4,4,4]
    d = len(shape)
    A = np.arange(np.prod(shape)).reshape(shape)
    Aglobal = A.copy()              # Used in f in order to test fix_indices

    def f(X, params):
        global feval
        feval += 1
        return Aglobal[tuple(X)]

    X = [ np.arange(s,dtype=int) for s in shape ]
    TW = TT.TensorWrapper(f, X, None,dtype=A.dtype,marshal_f=False)
    Atmp_shape = A.copy()             # Storing original array
    newshape = [2]*int(round(np.log2(np.prod(shape))))
    A = np.reshape( A, newshape )
    TW.reshape( newshape )

    ###################################################################
    # Reshaped 2 - Single slice
    #
    testn += 1
    idx = (0,slice(None,None,None),1,1,0,1,0)
    test(testn,"Reshaped 2 - Single slice",idx)

    ###################################################################
    # Reshaped 2 - Partial slice
    #
    testn += 1
    idx = (0,slice(0,1,1),1,1,1,0,1)
    test(testn,"Reshaped 2 - Partial slice",idx)

    ###################################################################
    # Reshaped 2 - Multiple slice
    #
    testn += 1
    idx = (slice(0,1,1),0,slice(None,None,None),1,1,0,1)
    test(testn,"Reshaped 2 - Multiple slice",idx)

    ###################################################################
    # Reshaped 2 - Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(A.shape))
    test(testn,"Reshaped 2 - Full slice",idx)

    ###################################################################
    # Reshaped 2 - List 
    #
    testn += 1
    idx = ([1,0],[0,1],[1,0],[0,1],[0,0],[1,1],[0,1])
    test(testn,"Reshaped 2 - Lists",idx)

    ###################################################################
    # Reshaped 2 - Single list 
    #
    testn += 1
    idx = (1,[0,0],1,0,0,1,1)
    test(testn,"Reshaped 2 - Single list",idx)

    ###################################################################
    # Reshaped 2 - Double list 
    #
    testn += 1
    idx = (1,[1,0],[0,0],0,1,0,1)
    test(testn,"Reshaped 2 - Double list",idx)

    ###################################################################
    # Reshaped 2 - Single list slice
    #
    testn += 1
    idx = (1,slice(None,None,None),[0,1],0,0,1,1)
    test(testn,"Reshaped 2 - Single list slice",idx)

    testn += 1
    idx = (1,[0,0],slice(None,None,None),0,1,1,1)
    test(testn,"Reshaped 2 - Single list slice",idx)

    testn += 1
    idx = (1,[0,1,1],0,slice(None,None,None),1,0,1)
    test(testn,"Reshaped 2 - Single list slice",idx)

    ###################################################################
    # Reshaped 2 - Double list slice
    #
    testn += 1
    idx = (1,[0,1],slice(None,None,None),[0,1],0,0,1)
    test(testn,"Reshaped 2 - Double list slice",idx)

    testn += 1
    idx = (slice(None,None,None),0,slice(None,None,None),[0,1],0,1,1)
    test(testn,"Reshaped 2 - Double slice list",idx)

    ###################################################################
    # Reshaped 2 - Lists slice
    #
    testn += 1
    idx = ([0,1],[0,1],slice(None,None,None),[1,0],[0,0],[1,1],[1,0])
    test(testn,"Reshaped 2 - Lists slice",idx)

    ###################################################################
    # Reshaped 2 - Fix indices
    #
    testn += 1
    fix_idxs = [0,0,1]
    fix_dims = [0,3,2]
    Atmp = A.copy()
    A = A[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped 2 - Fix indices", idx)

    ###################################################################
    # Reshaped 2 - Release indices
    #
    testn += 1
    A = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped 2 - Release indices", idx)

    ###################################################################
    # Reshaped 2 - Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = A.copy()
    A = A[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Reshaped 2 - Fix indices - second test", idx)

    ###################################################################
    # Reshaped 2 - Release indices
    #
    testn += 1
    A = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped 2 - Release indices", idx)

    ###################################################################
    # Reshaped 2 - Restore original shape
    #
    testn += 1
    A = Atmp_shape.copy()
    TW.reset_shape()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped 2 - Restore original shape", idx)
    
    ##
    ## Shape restored
    #############################################################################
    #############################################################################

    #############################################################################
    #############################################################################
    ## PowQ: Check funtionalities for power of Q extension
    ##
    shape = [3,5,5,5]
    Q = 2
    qshape = [ Q**(int(math.log(s,Q))+1) for s in shape ]
    d = len(shape)
    
    # Create A
    vs = [ 1 + npr.random(size=shape[i]) for i in range(d) ]
    A = reduce(np.multiply, np.ix_(*vs))

    # Create Atest
    vs_test = [ np.hstack( (vs[i], np.ones(qshape[i]-shape[i])*vs[i][-1]) ) for i in range(d) ]
    # vs_test = [ np.hstack( (vs[i], np.zeros(qshape[i]-shape[i])) ) for i in range(d) ]
    Atest = reduce(np.multiply, np.ix_(*vs_test))

    Aglobal = A.copy()              # Used in f in order to test fix_indices

    def f(X, params):
        global feval
        feval += 1
        return Aglobal[tuple(X)]

    def test(testn, title, idx):
        global feval
        global nsucc
        global nfail
        feval = 0
        TW.data = {}
        out = TW[idx]
        if np.any(Atest[idx].shape != out.shape) or (not np.allclose(Atest[idx], out, rtol=1e-10, atol=1e-12)):
            print_fail(testn,title,msg='Different output - idx: ' + str(idx))
            nfail += 1
        elif feval != np.prod(np.unique(Atest[idx]).shape):
        # elif feval != np.prod(np.unique(Atest[idx][Atest[idx]>0]).shape):
            print_fail(testn,title,msg='Wrong number of function evaluations - idx: ' + str(idx))
            nfail += 1
        else:
            print_ok(testn,title)
            nsucc += 1

    X = [ np.arange(s,dtype=int) for s in shape ]
    TW = TT.TensorWrapper(f, X, None, dtype=A.dtype, marshal_f=False)
    TW.set_Q( Q )

    ###################################################################
    # 01: Single address access
    #
    idx = (1,2,3,4)
    feval = 0
    out = TW[idx]
    testn += 1
    if Atest[idx] != out:
        print_fail(testn,"PowQ - Single address access",msg='Different output')
        nfail += 1
    elif feval != 1:
        print_fail(testn,"PowQ - Single address access",msg='Wrong number of function evaluations')
        nfail += 1
    else:
        print_ok(testn,"PowQ - Single address access")
        nsucc += 1

    ###################################################################
    # Storage
    testn += 1
    TW.data = {}
    TW.store_location = store_location
    TW[:,:,:,0]
    TW.store(force=True)
    print_ok(testn, "PowQ - Storage")
    nsucc += 1
    # Reload
    testn += 1
    TW = TT.load(store_location)
    TW.set_f(f,False)
    idx = tuple( [slice(None,None,None)]*d )
    test(testn,"PowQ - Reload",idx)
    if os.path.isfile(store_location + ".pkl"):
        os.remove(store_location + ".pkl")
    if os.path.isfile(store_location + ".h5"):
        os.remove(store_location + ".h5")
    if os.path.isfile(store_location + ".pkl.old"):
        os.remove(store_location + ".pkl.old")
    if os.path.isfile(store_location + ".h5.old"):
        os.remove(store_location + ".h5.old")

    ###################################################################
    # Single slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,4)
    test(testn,"PowQ - Single slice",idx)

    ###################################################################
    # Partial slice
    #
    testn += 1
    idx = (1,2,slice(1,3,1),4)
    test(testn,"PowQ - Partial slice",idx)

    ###################################################################
    # Partial stepping slice
    #
    testn += 1
    idx = (1,2,slice(0,4,2),4)
    test(testn,"PowQ - Partial stepping slice",idx)

    ###################################################################
    # Multiple slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,slice(0,4,2))
    test(testn,"PowQ - Multiple slice",idx)

    ###################################################################
    # Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(shape))
    test(testn,"PowQ - Full slice",idx)

    ###################################################################
    # List 
    #
    testn += 1
    idx = ([0,1],[1,2],[1,3],[0,4])
    test(testn,"PowQ - Lists",idx)

    ###################################################################
    # Single list 
    #
    testn += 1
    idx = (0,1,[1,3],3)
    test(testn,"PowQ - Single list",idx)

    ###################################################################
    # Double list 
    #
    testn += 1
    idx = (0,[0,2],[1,3],3)
    test(testn,"PowQ - Double list",idx)

    ###################################################################
    # Single list slice
    #
    testn += 1
    idx = (0,[0,2],slice(None,None,None),3)
    test(testn,"PowQ - Single list slice",idx)

    testn += 1
    idx = (0,slice(None,None,None),[0,2],3)
    test(testn,"PowQ - Single list slice",idx)

    testn += 1
    idx = (slice(None,None,None),0,[0,2,3],3)
    test(testn,"PowQ - Single list slice",idx)

    ###################################################################
    # Double list slice
    #
    testn += 1
    idx = ([0,1],slice(None,None,None),[0,2],3)
    test(testn,"PowQ - Double list slice",idx)

    testn += 1
    idx = (slice(None,None,None),0,slice(None,None,None),[0,2,3])
    test(testn,"PowQ - Double slice list",idx)

    ###################################################################
    # Lists slice
    #
    testn += 1
    idx = ([0,1],[0,2],slice(None,None,None),[1,3])
    test(testn,"PowQ - Lists slice",idx)

    ###################################################################
    # Fix indices
    #
    testn += 1
    fix_idxs = [0,2]
    fix_dims = [0,2]
    Atmp = Atest.copy()
    Atest = Atest[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "PowQ - Fix indices", idx)

    ###################################################################
    # Release indices
    #
    testn += 1
    Atest = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "PowQ - Release indices", idx)

    ###################################################################
    # Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = Atest.copy()
    Atest = Atest[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "PowQ - Fix indices - second test", idx)

    ###################################################################
    # Release indices
    #
    testn += 1
    Atest = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "PowQ - Release indices", idx)

    #############################################################################
    #############################################################################
    ## Reshaped PowQ: Check funtionalities for power of Q extension
    ##
    shape = [3,5,5,5]
    Q = 2
    qshape = [ Q**(int(math.log(s,Q))+1) for s in shape ]
    d = len(shape)
    
    # Create A
    vs = [ 1 + npr.random(size=shape[i]) for i in range(d) ]
    A = reduce(np.multiply, np.ix_(*vs))

    # Create Atest
    vs_test = [ np.hstack( (vs[i], np.ones(qshape[i]-shape[i])*vs[i][-1]) ) for i in range(d) ]
    # vs_test = [ np.hstack( (vs[i], np.zeros(qshape[i]-shape[i])) ) for i in range(d) ]
    Atest = reduce(np.multiply, np.ix_(*vs_test))

    Aglobal = A.copy()              # Used in f in order to test fix_indices

    def f(X, params):
        global feval
        feval += 1
        return Aglobal[tuple(X)]

    def test(testn, title, idx):
        global feval
        global nsucc
        global nfail
        feval = 0
        TW.data = {}
        out = TW[idx]
        if np.any(Atest[idx].shape != out.shape) or (not np.allclose(Atest[idx], out, rtol=1e-10, atol=1e-12)):
            print_fail(testn,title,msg='Different output - idx: ' + str(idx))
            nfail += 1
        elif feval != np.prod(np.unique(Atest[idx]).shape):
        # elif feval != np.prod(np.unique(Atest[idx][Atest[idx]>0]).shape):
            print_fail(testn,title,msg='Wrong number of function evaluations - idx: ' + str(idx))
            nfail += 1
        else:
            print_ok(testn,title)
            nsucc += 1

    X = [ np.arange(s,dtype=int) for s in shape ]
    TW = TT.TensorWrapper(f, X, None, dtype=A.dtype, marshal_f=False)
    TW.set_Q( Q )
    Atest_shape = Atest.copy()             # Storing original array
    newshape = [Q]*int(math.log(np.prod(qshape),Q))
    Atest = np.reshape( Atest, newshape )
    TW.reshape( newshape )

    ###################################################################
    # Reshaped PowQ - Single slice
    #
    testn += 1
    idx = (0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-2) ])
    test(testn,"Reshaped PowQ - Single slice",idx)

    ###################################################################
    # Reshaped PowQ - Partial slice
    #
    testn += 1
    idx = (0,slice(0,1,1)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-2) ])
    test(testn,"Reshaped PowQ - Partial slice",idx)

    ###################################################################
    # Reshaped PowQ - Multiple slice
    #
    testn += 1
    idx = (slice(0,1,1),0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-3) ])
    test(testn,"Reshaped PowQ - Multiple slice",idx)

    ###################################################################
    # Reshaped PowQ - Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(Atest.shape))
    test(testn,"Reshaped PowQ - Full slice",idx)

    ###################################################################
    # Reshaped PowQ - List 
    #
    testn += 1
    nn = 2
    idx = tuple( [ [random.randint(0,Q-1) for j in range(nn)] for i in range(len(newshape)) ] )
    test(testn,"Reshaped PowQ - Lists",idx)

    ###################################################################
    # Reshaped PowQ - Single list 
    #
    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-2) ])
    test(testn,"Reshaped PowQ - Single list",idx)

    ###################################################################
    # Reshaped PowQ - Double list 
    #
    testn += 1
    nn = 2
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-3) ])
    test(testn,"Reshaped PowQ - Double list",idx)

    ###################################################################
    # Reshaped PowQ - Single list slice
    #
    testn += 1
    nn = 2
    idx = (1,slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-3) ])
    test(testn,"Reshaped PowQ - Single list slice",idx)

    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-3) ])
    test(testn,"Reshaped PowQ - Single list slice",idx)

    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-4) ])
    test(testn,"Reshaped PowQ - Single list slice",idx)

    ###################################################################
    # Reshaped PowQ - Double list slice
    #
    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-4) ])
    test(testn,"Reshaped PowQ - Double list slice",idx)

    testn += 1
    nn = 3
    idx = (slice(None,None,None),0,slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape)-4) ])
    test(testn,"Reshaped PowQ - Double slice list",idx)

    ###################################################################
    # Reshaped PowQ - Lists slice
    #
    testn += 1
    nn = 3
    idx = ([random.randint(0,Q-1) for j in range(nn)],[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None)) + tuple( [ [random.randint(0,Q-1) for j in range(nn)] for i in range(len(newshape)-3) ] )
    test(testn,"Reshaped PowQ - Lists slice",idx)

    ###################################################################
    # Reshaped PowQ - Fix indices
    #
    testn += 1
    fix_idxs = [0,0,1]
    fix_dims = [0,3,2]
    Atmp = Atest.copy()
    Atest = Atest[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Fix indices", idx)

    ###################################################################
    # Reshaped PowQ - Release indices
    #
    testn += 1
    Atest = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Release indices", idx)

    ###################################################################
    # Reshaped PowQ - Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = Atest.copy()
    Atest = Atest[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Reshaped PowQ - Fix indices - second test", idx)

    ###################################################################
    # Reshaped PowQ - Release indices
    #
    testn += 1
    Atest = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Release indices", idx)

    ###################################################################
    # Reshaped PowQ - Restore original shape
    #
    testn += 1
    Atest = Atest_shape.copy()
    TW.reset_ghost_shape()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Restore original shape", idx)
    ##
    ## Shape restored
    #############################################################################
    #############################################################################

    
    #############################################################################
    #############################################################################
    ## Test sub views: construct a sub view and check its consistency
    ##
    
    coarse_shape = [3,5,5,5]
    full_shape = [ 2*s-1 for s in coarse_shape ]
    X_full = [ np.linspace(0,1,s) for s in full_shape ]
    X_coarse = [ np.linspace(0,1,s) for s in coarse_shape ]
    d = len(full_shape)

    rnd_p = npr.random(size=d)
    
    tmp = [ x * p for x,p in zip(X_full, rnd_p) ]
    Atest_full = reduce( np.add, np.ix_(*tmp) )
    Atest_coarse = Atest_full[::2,::2,::2,::2]

    def f(X, params):
        global feval
        feval += 1
        return np.dot(rnd_p,X)
    
    def test(testn, title, idx, view):
        global feval
        global nsucc
        global nfail
        feval = 0
        TW.data = {}
        TW.set_active_view(view)
        out = TW[idx]
        if view == 'full': A = Atest_full
        elif view == 'coarse': A = Atest_coarse
        if np.any(A[idx].shape != out.shape) or (not np.allclose(A[idx], out, rtol=1e-10, atol=1e-12)):
            print_fail(testn,title,msg='Different output - idx: ' + str(idx))
            nfail += 1
        elif feval != np.prod(np.unique(A[idx]).shape):
            print_fail(testn,title,msg='Wrong number of function evaluations - idx: ' + str(idx))
            nfail += 1
        else:
            print_ok(testn,title)
            nsucc += 1
            
    TW = TT.TensorWrapper(f, X_full, None, dtype=Atest_full.dtype, marshal_f=False)
    TW.set_view( 'coarse', X_coarse )
    
    ######################################################
    # Single index full
    testn += 1
    idx = (1,3,3,3)
    test( testn, "View - Single index full", idx, 'full' )

    ######################################################
    # Single index coarse
    testn += 1
    idx = (1,3,3,3)
    test( testn, "View - Single index coarse", idx, 'coarse' )

    ######################################################
    # Single slice full
    testn += 1
    idx = (1,slice(None,None,None),3,3)
    test( testn, "View - Single slice full", idx, 'full' )

    ######################################################
    # Single slice coarse
    testn += 1
    idx = (1,3,slice(None,None,None),3)
    test( testn, "View - Single slice coarse", idx, 'coarse' )

    ######################################################
    # Partial slice full
    testn += 1
    idx = (1,slice(1,3,1),3,3)
    test( testn, "View - Partial slice full", idx, 'full' )

    ######################################################
    # Partial index coarse
    testn += 1
    idx = (1,3,slice(0,2,1),3)
    test( testn, "View - Partial slice coarse", idx, 'coarse' )

    ######################################################
    # Multiple slice full
    testn += 1
    idx = (1,slice(1,3,1),slice(None,None,None),3)
    test( testn, "View - Multiple slice full", idx, 'full' )

    ######################################################
    # Multiple slice coarse
    testn += 1
    idx = (1,slice(None,None,None),slice(0,2,1),3)
    test( testn, "View - Multiple slice coarse", idx, 'coarse' )

    ######################################################
    # Full slice full
    testn += 1
    idx = tuple([slice(None,None,None)] * d)
    test( testn, "View - Full slice full", idx, 'full' )

    ######################################################
    # Full slice coarse
    testn += 1
    idx = tuple([slice(None,None,None)] * d)
    test( testn, "View - Full slice coarse", idx, 'coarse' )

    ######################################################
    # List full
    testn += 1
    nn = 2
    idx = tuple( [ [random.randint(0,full_shape[i]-1) for j in range(nn)] for i in range(d) ] )
    test( testn, "View - List full", idx, 'full' )

    ######################################################
    # List coarse
    testn += 1
    nn = 2
    idx = tuple( [ [random.randint(0,coarse_shape[i]-1) for j in range(nn)] for i in range(d) ] )
    test( testn, "View - List coarse", idx, 'coarse' )

    ######################################################
    # Single list slice full
    testn += 1
    nn = 2
    idx = (1, [random.randint(0,full_shape[1]-1) for j in range(nn)], slice(None,None,None), 2)
    test( testn, "View - Single list slice full", idx, 'full' )

    ######################################################
    # Single list slice coarse
    testn += 1
    nn = 2
    idx = (1, [random.randint(0,coarse_shape[1]-1) for j in range(nn)], slice(None,None,None), 2)
    test( testn, "View - Single list slice coarse", idx, 'coarse' )

    ######################################################
    # Single list slice full
    testn += 1
    nn = 2
    idx = (1, slice(None,None,None), [random.randint(0,full_shape[2]-1) for j in range(nn)], 2)
    test( testn, "View - Single list slice full", idx, 'full' )

    ######################################################
    # Single list slice coarse
    testn += 1
    nn = 2
    idx = (1, slice(None,None,None), [random.randint(0,coarse_shape[2]-1) for j in range(nn)], 2)
    test( testn, "View - Single list slice coarse", idx, 'coarse' )

    ######################################################
    # Single list slice full
    testn += 1
    nn = 2
    idx = (1, slice(None,None,None), 2, [random.randint(0,full_shape[3]-1) for j in range(nn)])
    test( testn, "View - Single list slice full", idx, 'full' )

    ######################################################
    # Single list slice coarse
    testn += 1
    nn = 2
    idx = (1, slice(None,None,None), 2, [random.randint(0,coarse_shape[3]-1) for j in range(nn)])
    test( testn, "View - Single list slice coarse", idx, 'coarse' )

    ######################################################
    # Sub view and PowQ
    ######################################################
    Q = 2
    ext_full_shape = [ Q**(int(math.log(s,Q))+1) for s in full_shape ]
    ext_coarse_shape = [ Q**(int(math.log(s,Q))+1) for s in coarse_shape ]
    # Create folded tensors
    X_ext_full = [ np.hstack( (X_full[i], np.ones(ext_full_shape[i]-full_shape[i])*X_full[i][-1]) ) for i in range(d) ]
    X_ext_coarse = [ np.hstack( (X_coarse[i], np.ones(ext_coarse_shape[i]-coarse_shape[i])*X_coarse[i][-1]) ) for i in range(d) ]
    tmp_ext_full = [ x * p for x,p in zip(X_ext_full, rnd_p) ]
    Atest_ext_full = reduce( np.add, np.ix_(*tmp_ext_full) )
    tmp_ext_coarse = [ x * p for x,p in zip(X_ext_coarse, rnd_p) ]
    Atest_ext_coarse = reduce( np.add, np.ix_(*tmp_ext_coarse) )

    def f(X, params):
        global feval
        feval += 1
        return np.dot(rnd_p,X)

    def test(testn, title, idx, view):
        global feval
        global nsucc
        global nfail
        feval = 0
        TW.data = {}
        TW.set_active_view(view)
        out = TW[idx]
        if view == 'full': A = Atest_ext_full
        elif view == 'coarse': A = Atest_ext_coarse
        if np.any(A[idx].shape != out.shape) or (not np.allclose(A[idx], out, rtol=1e-10, atol=1e-12)):
            print_fail(testn,title,msg='Different output - idx: ' + str(idx))
            nfail += 1
        elif feval != np.prod(np.unique(A[idx]).shape):
            print_fail(testn,title,msg='Wrong number of function evaluations - idx: ' + str(idx))
            nfail += 1
        else:
            print_ok(testn,title)
            nsucc += 1
    
    TW.set_active_view('coarse')
    TW.set_Q(2)
    TW.set_active_view('full')
    TW.set_Q(2)

    ########## 
    # Full
    TW.set_active_view('full')
    TW.data = {}

    ###################################################################
    # 01: Single address access
    #
    idx = (1,2,3,4)
    feval = 0
    out = TW[idx]
    testn += 1
    if not np.isclose(Atest_ext_full[idx], out, rtol=1e-10, atol=1e-12):
        print_fail(testn,"Sub view and PowQ - Single address access - full",msg='Different output')
        nfail += 1
    elif feval != 1:
        print_fail(testn,"Sub view and PowQ - Single address access - full",msg='Wrong number of function evaluations')
        nfail += 1
    else:
        print_ok(testn,"Sub view and PowQ - Single address access - full")
        nsucc += 1

    ###################################################################
    # Storage
    testn += 1
    TW.data = {}
    TW.store_location = store_location
    TW[:,:,:,0]
    TW.store(force=True)
    print_ok(testn, "Sub view and PowQ - Storage - full")
    nsucc += 1
    # Reload
    testn += 1
    TW = TT.load(store_location)
    TW.set_f(f,False)
    idx = tuple( [slice(None,None,None)]*d )
    test(testn,"Sub view and PowQ - Reload - full",idx, 'full')
    if os.path.isfile(store_location + ".pkl"):
        os.remove(store_location + ".pkl")
    if os.path.isfile(store_location + ".h5"):
        os.remove(store_location + ".h5")
    if os.path.isfile(store_location + ".pkl.old"):
        os.remove(store_location + ".pkl.old")
    if os.path.isfile(store_location + ".h5.old"):
        os.remove(store_location + ".h5.old")

    ###################################################################
    # Single slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,4)
    test(testn,"Sub view and PowQ - Single slice - full",idx, 'full')

    ###################################################################
    # Partial slice
    #
    testn += 1
    idx = (1,2,slice(1,3,1),4)
    test(testn,"Sub view and PowQ - Partial slice - full",idx, 'full')

    ###################################################################
    # Partial stepping slice
    #
    testn += 1
    idx = (1,2,slice(0,4,2),4)
    test(testn,"Sub view and PowQ - Partial stepping slice - full",idx, 'full')

    ###################################################################
    # Multiple slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,slice(0,4,2))
    test(testn,"Sub view and PowQ - Multiple slice - full",idx, 'full')

    ###################################################################
    # Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(shape))
    test(testn,"Sub view and PowQ - Full slice - full",idx, 'full')

    ###################################################################
    # List 
    #
    testn += 1
    idx = ([0,1],[1,2],[1,3],[0,4])
    test(testn,"Sub view and PowQ - Lists - full",idx, 'full')

    ###################################################################
    # Single list 
    #
    testn += 1
    idx = (0,1,[1,3],3)
    test(testn,"Sub view and PowQ - Single list - full",idx, 'full')

    ###################################################################
    # Double list 
    #
    testn += 1
    idx = (0,[0,2],[1,3],3)
    test(testn,"Sub view and PowQ - Double list - full",idx, 'full')

    ###################################################################
    # Single list slice
    #
    testn += 1
    idx = (0,[0,2],slice(None,None,None),3)
    test(testn,"Sub view and PowQ - Single list slice - full",idx, 'full')

    testn += 1
    idx = (0,slice(None,None,None),[0,2],3)
    test(testn,"Sub view and PowQ - Single list slice - full",idx, 'full')

    testn += 1
    idx = (slice(None,None,None),0,[0,2,3],3)
    test(testn,"Sub view and PowQ - Single list slice - full",idx, 'full')

    ###################################################################
    # Double list slice
    #
    testn += 1
    idx = ([0,1],slice(None,None,None),[0,2],3)
    test(testn,"Sub view and PowQ - Double list slice - full",idx, 'full')

    testn += 1
    idx = (slice(None,None,None),0,slice(None,None,None),[0,2,3])
    test(testn,"Sub view and PowQ - Double slice list - full",idx, 'full')

    ###################################################################
    # Lists slice
    #
    testn += 1
    idx = ([0,1],[0,2],slice(None,None,None),[1,3])
    test(testn,"Sub view and PowQ - Lists slice - full",idx, 'full')

    ###################################################################
    # Fix indices
    #
    testn += 1
    fix_idxs = [0,2]
    fix_dims = [0,2]
    Atmp = Atest_ext_full.copy()
    Atest_ext_full = Atest_ext_full[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Sub view and PowQ - Fix indices - full", idx, 'full')

    ###################################################################
    # Release indices
    #
    testn += 1
    Atest_ext_full = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Sub view and PowQ - Release indices - full",idx, 'full')

    ###################################################################
    # Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = Atest_ext_full.copy()
    Atest_ext_full = Atest_ext_full[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Sub view and PowQ - Fix indices - second test - full",idx, 'full')

    ###################################################################
    # Release indices
    #
    testn += 1
    Atest_ext_full = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Sub view and PowQ - Release indices - full",idx, 'full')


    ##########
    # Coarse
    TW.set_active_view('coarse')
    TW.data = {}
    
    ###################################################################
    # 01: Single address access
    #
    idx = (1,2,3,4)
    feval = 0
    out = TW[idx]
    testn += 1
    if not np.isclose(Atest_ext_coarse[idx], out, rtol=1e-10, atol=1e-12):
        print_fail(testn,"Sub view and PowQ - Single address access - coarse",msg='Different output')
        nfail += 1
    elif feval != 1:
        print_fail(testn,"Sub view and PowQ - Single address access - coarse",msg='Wrong number of function evaluations')
        nfail += 1
    else:
        print_ok(testn,"Sub view and PowQ - Single address access - coarse")
        nsucc += 1

    ###################################################################
    # Storage
    testn += 1
    TW.data = {}
    TW.store_location = store_location
    TW[:,:,:,0]
    TW.store(force=True)
    print_ok(testn, "Sub view and PowQ - Storage - coarse")
    nsucc += 1
    # Reload
    testn += 1
    TW = TT.load(store_location)
    TW.set_f(f,False)
    idx = tuple( [slice(None,None,None)]*d )
    test(testn,"Sub view and PowQ - Reload - coarse",idx, 'coarse')
    if os.path.isfile(store_location + ".pkl"):
        os.remove(store_location + ".pkl")
    if os.path.isfile(store_location + ".h5"):
        os.remove(store_location + ".h5")
    if os.path.isfile(store_location + ".pkl.old"):
        os.remove(store_location + ".pkl.old")
    if os.path.isfile(store_location + ".h5.old"):
        os.remove(store_location + ".h5.old")

    ###################################################################
    # Single slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,4)
    test(testn,"Sub view and PowQ - Single slice - coarse",idx, 'coarse')

    ###################################################################
    # Partial slice
    #
    testn += 1
    idx = (1,2,slice(1,3,1),4)
    test(testn,"Sub view and PowQ - Partial slice - coarse",idx, 'coarse')

    ###################################################################
    # Partial stepping slice
    #
    testn += 1
    idx = (1,2,slice(0,4,2),4)
    test(testn,"Sub view and PowQ - Partial stepping slice - coarse",idx, 'coarse')

    ###################################################################
    # Multiple slice
    #
    testn += 1
    idx = (1,slice(None,None,None),3,slice(0,4,2))
    test(testn,"Sub view and PowQ - Multiple slice - coarse",idx, 'coarse')

    ###################################################################
    # Coarse slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(shape))
    test(testn,"Sub view and PowQ - Coarse slice - coarse",idx, 'coarse')

    ###################################################################
    # List 
    #
    testn += 1
    idx = ([0,1],[1,2],[1,3],[0,4])
    test(testn,"Sub view and PowQ - Lists - coarse",idx, 'coarse')

    ###################################################################
    # Single list 
    #
    testn += 1
    idx = (0,1,[1,3],3)
    test(testn,"Sub view and PowQ - Single list - coarse",idx, 'coarse')

    ###################################################################
    # Double list 
    #
    testn += 1
    idx = (0,[0,2],[1,3],3)
    test(testn,"Sub view and PowQ - Double list - coarse",idx, 'coarse')

    ###################################################################
    # Single list slice
    #
    testn += 1
    idx = (0,[0,2],slice(None,None,None),3)
    test(testn,"Sub view and PowQ - Single list slice - coarse",idx, 'coarse')

    testn += 1
    idx = (0,slice(None,None,None),[0,2],3)
    test(testn,"Sub view and PowQ - Single list slice - coarse",idx, 'coarse')

    testn += 1
    idx = (slice(None,None,None),0,[0,2,3],3)
    test(testn,"Sub view and PowQ - Single list slice - coarse",idx, 'coarse')

    ###################################################################
    # Double list slice
    #
    testn += 1
    idx = ([0,1],slice(None,None,None),[0,2],3)
    test(testn,"Sub view and PowQ - Double list slice - coarse",idx, 'coarse')

    testn += 1
    idx = (slice(None,None,None),0,slice(None,None,None),[0,2,3])
    test(testn,"Sub view and PowQ - Double slice list - coarse",idx, 'coarse')

    ###################################################################
    # Lists slice
    #
    testn += 1
    idx = ([0,1],[0,2],slice(None,None,None),[1,3])
    test(testn,"Sub view and PowQ - Lists slice - coarse",idx, 'coarse')

    ###################################################################
    # Fix indices
    #
    testn += 1
    fix_idxs = [0,2]
    fix_dims = [0,2]
    Atmp = Atest_ext_coarse.copy()
    Atest_ext_coarse = Atest_ext_coarse[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Sub view and PowQ - Fix indices - coarse", idx, 'coarse')

    ###################################################################
    # Release indices
    #
    testn += 1
    Atest_ext_coarse = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Sub view and PowQ - Release indices - coarse",idx, 'coarse')

    ###################################################################
    # Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = Atest_ext_coarse.copy()
    Atest_ext_coarse = Atest_ext_coarse[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Sub view and PowQ - Fix indices - second test - coarse",idx, 'coarse')

    ###################################################################
    # Release indices
    #
    testn += 1
    Atest_ext_coarse = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Sub view and PowQ - Release indices - coarse",idx, 'coarse')

    ######################################################
    # Sub view and Reshaped PowQ
    ######################################################
    TW.data = {}
    newshape_full = [Q]*int(math.log(np.prod(ext_full_shape),Q))
    newshape_coarse = [Q]*int(math.log(np.prod(ext_coarse_shape),Q))
    Atest_ext_full_shape = Atest_ext_full.copy()
    Atest_ext_full = np.reshape( Atest_ext_full, newshape_full )
    TW.set_active_view('full')
    TW.reshape( newshape_full )
    Atest_ext_coarse_shape = Atest_ext_coarse.copy()
    Atest_ext_coarse = np.reshape( Atest_ext_coarse, newshape_coarse )
    TW.set_active_view('coarse')
    TW.reshape( newshape_coarse )

    ########## 
    # Full
    TW.set_active_view('full')
    TW.data = {}

    ###################################################################
    # Reshaped PowQ - Single slice
    #
    testn += 1
    idx = (0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-2) ])
    test(testn,"Reshaped PowQ - Single slice - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Partial slice
    #
    testn += 1
    idx = (0,slice(0,1,1)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-2) ])
    test(testn,"Reshaped PowQ - Partial slice - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Multiple slice
    #
    testn += 1
    idx = (slice(0,1,1),0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-3) ])
    test(testn,"Reshaped PowQ - Multiple slice - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Full slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(Atest_ext_full.shape))
    test(testn,"Reshaped PowQ - Full slice - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - List 
    #
    testn += 1
    nn = 2
    idx = tuple( [ [random.randint(0,Q-1) for j in range(nn)] for i in range(len(newshape_full)) ] )
    test(testn,"Reshaped PowQ - Lists - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Single list 
    #
    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-2) ])
    test(testn,"Reshaped PowQ - Single list - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Double list 
    #
    testn += 1
    nn = 2
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-3) ])
    test(testn,"Reshaped PowQ - Double list - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Single list slice
    #
    testn += 1
    nn = 2
    idx = (1,slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-3) ])
    test(testn,"Reshaped PowQ - Single list slice - full",idx,'full')

    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-3) ])
    test(testn,"Reshaped PowQ - Single list slice - full",idx,'full')

    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-4) ])
    test(testn,"Reshaped PowQ - Single list slice - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Double list slice
    #
    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-4) ])
    test(testn,"Reshaped PowQ - Double list slice - full",idx,'full')

    testn += 1
    nn = 3
    idx = (slice(None,None,None),0,slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_full)-4) ])
    test(testn,"Reshaped PowQ - Double slice list - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Lists slice
    #
    testn += 1
    nn = 3
    idx = ([random.randint(0,Q-1) for j in range(nn)],[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None)) + tuple( [ [random.randint(0,Q-1) for j in range(nn)] for i in range(len(newshape_full)-3) ] )
    test(testn,"Reshaped PowQ - Lists slice - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Fix indices
    #
    testn += 1
    fix_idxs = [0,0,1]
    fix_dims = [0,3,2]
    Atmp = Atest_ext_full.copy()
    Atest_ext_full = Atest_ext_full[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Fix indices - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Release indices
    #
    testn += 1
    Atest_ext_full = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Release indices - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = Atest_ext_full.copy()
    Atest_ext_full = Atest_ext_full[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Reshaped PowQ - Fix indices - second test - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Release indices
    #
    testn += 1
    Atest_ext_full = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Release indices - full",idx,'full')

    ###################################################################
    # Reshaped PowQ - Restore original shape
    #
    testn += 1
    Atest_ext_full = Atest_ext_full_shape.copy()
    TW.reset_ghost_shape()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Restore original shape - full",idx,'full')

    ########## 
    # Coarse
    TW.set_active_view('coarse')
    TW.data = {}

    ###################################################################
    # Reshaped PowQ - Single slice
    #
    testn += 1
    idx = (0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-2) ])
    test(testn,"Reshaped PowQ - Single slice - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Partial slice
    #
    testn += 1
    idx = (0,slice(0,1,1)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-2) ])
    test(testn,"Reshaped PowQ - Partial slice - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Multiple slice
    #
    testn += 1
    idx = (slice(0,1,1),0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-3) ])
    test(testn,"Reshaped PowQ - Multiple slice - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Coarse slice
    #
    testn += 1
    idx = tuple([slice(None,None,None)] * len(Atest_ext_coarse.shape))
    test(testn,"Reshaped PowQ - Coarse slice - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - List 
    #
    testn += 1
    nn = 2
    idx = tuple( [ [random.randint(0,Q-1) for j in range(nn)] for i in range(len(newshape_coarse)) ] )
    test(testn,"Reshaped PowQ - Lists - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Single list 
    #
    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-2) ])
    test(testn,"Reshaped PowQ - Single list - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Double list 
    #
    testn += 1
    nn = 2
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-3) ])
    test(testn,"Reshaped PowQ - Double list - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Single list slice
    #
    testn += 1
    nn = 2
    idx = (1,slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-3) ])
    test(testn,"Reshaped PowQ - Single list slice - coarse",idx,'coarse')

    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-3) ])
    test(testn,"Reshaped PowQ - Single list slice - coarse",idx,'coarse')

    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],0,slice(None,None,None)) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-4) ])
    test(testn,"Reshaped PowQ - Single list slice - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Double list slice
    #
    testn += 1
    nn = 3
    idx = (1,[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-4) ])
    test(testn,"Reshaped PowQ - Double list slice - coarse",idx,'coarse')

    testn += 1
    nn = 3
    idx = (slice(None,None,None),0,slice(None,None,None),[random.randint(0,Q-1) for j in range(nn)]) + tuple([ random.randint(0,Q-1) for i in range(len(newshape_coarse)-4) ])
    test(testn,"Reshaped PowQ - Double slice list - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Lists slice
    #
    testn += 1
    nn = 3
    idx = ([random.randint(0,Q-1) for j in range(nn)],[random.randint(0,Q-1) for j in range(nn)],slice(None,None,None)) + tuple( [ [random.randint(0,Q-1) for j in range(nn)] for i in range(len(newshape_coarse)-3) ] )
    test(testn,"Reshaped PowQ - Lists slice - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Fix indices
    #
    testn += 1
    fix_idxs = [0,0,1]
    fix_dims = [0,3,2]
    Atmp = Atest_ext_coarse.copy()
    Atest_ext_coarse = Atest_ext_coarse[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Fix indices - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Release indices
    #
    testn += 1
    Atest_ext_coarse = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Release indices - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Fix indices 2
    #
    testn += 1
    fix_idxs = [0]
    fix_dims = [0]
    Atmp = Atest_ext_coarse.copy()
    Atest_ext_coarse = Atest_ext_coarse[ tuple([ fix_idxs[fix_dims.index(i)] if (i in fix_dims) else slice(None,None,None) for i in range(d) ]) ]
    TW.fix_indices(fix_idxs, fix_dims)
    idx = [ slice(None,None,None) ] + [(0,1)] * (TW.ndim-1)
    test(testn, "Reshaped PowQ - Fix indices - second test - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Release indices
    #
    testn += 1
    Atest_ext_coarse = Atmp.copy()
    TW.release_indices()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Release indices - coarse",idx,'coarse')

    ###################################################################
    # Reshaped PowQ - Restore original shape
    #
    testn += 1
    Atest_ext_coarse = Atest_ext_coarse_shape.copy()
    TW.reset_ghost_shape()
    idx = [ slice(None,None,None) ] * TW.ndim
    test(testn, "Reshaped PowQ - Restore original shape - coarse",idx,'coarse')


    #############################################################################
    #############################################################################

    print_summary("TensorWrapper", nsucc, nfail)

    return (nsucc,nfail)

if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestTensorWrapper.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs,loglev=logging.INFO)
