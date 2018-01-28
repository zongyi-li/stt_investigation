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

__all__ = ['Candecomp']

import logging
import numpy as np
import numpy.linalg as npla
from scipy import linalg as scla
from scipy import sparse as scsp
from scipy.sparse import linalg as spla

class Candecomp:
    
    def __init__(self,A):

        #############################
        # List of attributes
        self.CP = None
        self.rank = None
        # End list of attributes
        ############################

        # Initialize the tensor with the input tensor in Candecomp ([]np.ndarray) or tensor (np.ndarray) form
        # TODO: add dimension consistency checks
        if isinstance(A,list):
            self.rank = A[0].shape[0]
            self.CP = A
        elif isinstance(A,np.ndarray):
            raise NameError("DABITensor.Candecomp.__init__: TODO TENSOR to CANDECOMP")
        else:
            raise NameError("DABITensor.Candecomp.__init__: Input type not allowed")

    def __getitem__(self, idxs):
        if len(idxs) != len(self.CP):
            raise NameError("DABITensor.TT.__init__: len(idxs)!=len(CP)")
        vals = np.empty((self.rank,len(self.CP)),dtype=np.float64)
        for i in range(len(self.CP)):
            vals[:,i] = self.CP[i][:,idxs[i]]
        # Multiply over cols and sum over rows
        return np.sum(np.prod(vals,axis=1),axis=0)
    
    def size(self):
        tot = 0
        for CPi in self.CP: tot += np.prod(CPi.shape)
        return tot
    
    def to_TT(self):
        TT = [] # Contains the cores
        # First core
        TTi = np.empty((1,self.CP[0].shape[1],self.rank),dtype=np.float64)
        for l in range(self.CP[0].shape[1]): TTi[:,l,:] = self.CP[0][:,l].reshape((1,self.rank))
        TT.append(TTi)
        # Central cores
        for i in range(1,len(self.CP)-1):
            # TTi = np.empty((self.rank, self.CP[i].shape[1], self.rank),dtype=np.float64)
            # for l in range(self.CP[i].shape[1]): TTi[:,l,:] = np.diag(self.CP[i][:,l])

            TTi = np.zeros((self.rank, self.CP[i].shape[1], self.rank),dtype=np.float64)
            for l in range(self.CP[i].shape[1]): TTi[range(self.rank),l,range(self.rank)] = self.CP[i][:,l]

            TT.append(TTi)

        if len(self.CP) > 1:
            # Last core
            TTi = np.empty((self.rank,self.CP[-1].shape[1],1),dtype=np.float64)
            for l in range(self.CP[-1].shape[1]): TTi[:,l,:] = self.CP[-1][:,l].reshape((self.rank,1))
            TT.append(TTi)
        return TT

    def to_TT_round(self):
        # Perform rounding on the fly. Increase the computational time, but save in memory allocation during QR decomposition. Idea:
        # Iterate over Left-right svd k in 1:d-1:
        #   Peform Right-left qr, without storing values of Q a part for the k-th term
        #   Perform SVD
        # TODO
        pass
