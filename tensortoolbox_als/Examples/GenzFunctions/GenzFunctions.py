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

import sys

import numpy as np
import numpy.random as npr
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import axes3d

def test(FNUM, GenzNormalized=False, GenzPatrick=False):
    #########################################
    # Genz functions
    #########################################
    if not GenzPatrick:

        d = 2
        if FNUM == 0:
            FUNC = 0
        elif FNUM == 1:
            FUNC = 1
        elif FNUM == 2:
            FUNC = 2
        elif FNUM == 3:
            FUNC = 3
        elif FNUM == 4:
            FUNC = 4
        elif FNUM == 5:
            FUNC = 5

    if GenzPatrick:
        file_name_pat = "Patrick"
        GenzNormalized = True

        d = 2 
        if FNUM == 0:
            FUNC = 0
        elif FNUM == 1:
            FUNC = 1
        elif FNUM == 2:
            FUNC = 2
        elif FNUM == 3:
            FUNC = 3

    print "Function: " + str(FUNC) + " Norm: " + str(GenzNormalized) + " Dims: " + str(d)

    xspan = [0.,1.]
    
    expnts = np.array([ 1.5, 2.0, 2.0, 1.0, 2.0, 2.0 ])
    dfclt = np.array([284.6, 725.0, 185.0, 70.3, 2040., 430.])
    # dfclt = np.array([110., 600., 600., 100., 150., 100.])
    if not GenzPatrick:
        csSum = dfclt / (float(d)**expnts)
    else:
        csSum = np.array([1.5, float(d), 1.85, 7.03, 20.4, 4.3])
    # csSum = np.array([9., 7.25, 1.85, 7.03, 20.4, 4.3])

    if FUNC != 5:
        ws = npr.random(d)
    elif FUNC == 5:
        # For function 5 let the discontinuity be cutting the space in two equiprobable regions
        beta = 1.
        alpha = np.exp(np.log(1./2.)/d) / (1 - np.exp(np.log(1./2.)/d)) * beta
        dd = stats.beta(alpha,beta)
        ws = dd.rvs(d)

    cs = npr.random(d)
    if GenzNormalized:
        cs *= csSum[FUNC] / np.sum(cs)
    params = {'ws':ws,'cs':cs}

    if FUNC == 0:
        # Oscillatory
        def f(X,params): 
            if X.ndim == 1:
                return np.cos(2.*np.pi*params['ws'][0] + np.sum( params['cs'] * X ))
            else:
                return np.cos(2.*np.pi*params['ws'][0] + np.sum( np.tile(params['cs'],(X.shape[0],1)) * X, 1))

    elif FUNC == 1:
        # Product peak
        def f(X,params): 
            if X.ndim == 1: 
                return np.prod( ( params['cs']**-2. + (X - params['ws'])**2. )**-1. )
            else:
                return np.prod( ( np.tile(params['cs'], (X.shape[0],1))**-2. + (X - np.tile(params['ws'],(X.shape[0],1))) ** 2. ) ** -1. , 1)

    elif FUNC == 2:
        # Corner peak
        def f(X,params): 
            if X.ndim == 1:
                return (1.+ np.sum(params['cs'] * X))**(-(d+1.))
            else:
                return (1. + np.sum( np.tile(params['cs'],(X.shape[0],1)) * X, 1 ) ) ** (-(d+1))

    elif FUNC == 3:
        # Gaussian
        def f(X,params): 
            if X.ndim == 1:
                return np.exp( - np.sum( params['cs']**2. * (X - params['ws'])**2. ) )
            else:
                return np.exp( - np.sum( np.tile(params['cs'],(X.shape[0],1))**2. * (X - np.tile(params['ws'],(X.shape[0],1)))**2., 1 ) )

    elif FUNC == 4:
        # Continuous
        def f(X,params):
            if X.ndim == 1:
                return np.exp( - np.sum( params['cs'] * np.abs(X - params['ws']) ) );
            else:
                return np.exp( - np.sum( np.tile(params['cs'],(X.shape[0],1)) * np.abs(X - np.tile(params['ws'],(X.shape[0],1))), 1 ) );

    elif FUNC == 5:
        # Discontinuous (not C^0)
        def f(X,params):
            ws = params['ws']/2. + 0.25
            # ws = params['ws']
            if X.ndim == 1:
                if np.any(X > ws): return 0.
                else: return np.exp( np.sum( params['cs'] * X ) )
            else:
                out = np.zeros(X.shape[0])
                idxs = np.where( np.logical_not( np.any(X > np.tile(ws,(X.shape[0],1)), axis=1) ) )[0]
                if len(idxs) == 1:
                    out[ idxs ] = np.exp( np.sum( params['cs'] * X[idxs,:] ) )
                elif len(idxs) > 1:
                    out[ idxs ] = np.exp( np.sum( np.tile(params['cs'],(len(idxs),1)) * X[idxs,:] , 1) )
                return out

    # Plot function
    pN = 40
    px = np.linspace(0.,1.,pN)
    pX,pY = np.meshgrid(px,px)
    pZ = np.zeros(pX.shape)
    for i in range(pN):
        for j in range(pN):
            pZ[i,j] = f(np.array([pX[i,j],pY[i,j]]),params)

    fig = plt.figure(figsize=(6,5))
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(pX,pY,pZ)
    plt.tight_layout()
    plt.show(block=False)

if __name__ == "__main__":
    FNUM = int(sys.argv[1])
    GenzNorm = (sys.argv[2] == 'True')
    if len(sys.argv) < 4:
        GenzPatrick = False
    else:
        GenzPatrick = (sys.argv[3] == 'True')
    test(FNUM,GenzNorm,GenzPatrick)
