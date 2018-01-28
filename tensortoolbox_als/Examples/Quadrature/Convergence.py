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
# Copyright (C) 2014-2015 The Technical University of Denmark
# Scientific Computing Section
# Department of Applied Mathematics and Computer Science
#
# Author: Daniele Bigoni
#

import numpy as np

from QTTdmrgQuadrature import runner as TTr
from UQQuadrature import runner as UQr

import matplotlib.pyplot as plt

# Proceed by powers of 2
P = 10

TTfevals = np.zeros( (1,P) )
TTerrs = np.zeros( (1,P) )
for i in range(1,P+1):
    print("RUNNING TT P=" + str(i))
    N = 2**i
    
    # QTT approx
    (feval, err) = TTr( N-1 )
    TTfevals[0,i-1] = feval
    TTerrs[0,i-1] = err

U = 18
UQfevals = np.zeros( (2,U) )
UQerrs = np.zeros( (2,U) )
for i in range(1,U+1):
    N = 2**i
    print("RUNNING UQ U=" + str(i))
    print("RUNNING UQ N=" + str(N))

    # UQ approx MC
    (feval, err) = UQr( N, 'mc' )
    UQfevals[0,i-1] = feval
    UQerrs[0,i-1] = err

    # UQ approx MC
    (feval, err) = UQr( N, 'lhc' )
    UQfevals[1,i-1] = feval
    UQerrs[1,i-1] = err

plt.figure()
plt.loglog(TTfevals[0,:],TTerrs[0,:],'o-',label='QTT')
plt.loglog(UQfevals[0,:],UQerrs[0,:],'o-',label='MC')
plt.loglog(UQfevals[1,:],UQerrs[1,:],'o-',label='LHC')
plt.xlabel('N. func. eval')
plt.ylabel('Error')
plt.legend()
plt.show(block=False)
