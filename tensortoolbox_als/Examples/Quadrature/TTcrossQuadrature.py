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

import TensorToolbox as DT
import TensorToolbox.multilinalg as mla

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from UQToolbox import UncertaintyQuantification as UQ

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

# Dimension of the function. The script works for any dimension, but for d=2,3 
# some plotting is provided and the error estimation can be done against the
# analytical value of the integral
d = 3

# Parameters defining the node's grid. 
# For closed and bounded intervals, use S1D.JACOBI with parameters [0.,0.].
# As quadrature nodes you can pick S1D.GAUSS, S1D.GAUSSLOBATTO, S1D.GAUSSRADAU
# For the whole real line, use S1D.HERMITEP_PROB with parameters None.
# As quadrature nodes you can pick only S1D.GAUSS.
# For the half real line, use S1D.LAGUERREP with parameter alpha (select it according to your measure)
# As quadrature nodes you can pick only S1D.GAUSS and S1D.GAUSSRADAU
# For more complex measures, the additional package PyORTHPOL can be used to construct proper quadrature rules.
# Additionally nested rules (not useful right now without order adaptation) can be used by the functions:
# S1D.cc(N) and S1D.fej(N) for Clenshaw-Curtis and Fejer's rules respectively.
PolyType = S1D.JACOBI
PolyParams = [0.,0.]
QuadType = S1D.GAUSS

# Polynomial order for each dimension. The actual implementation does not allow to have
# anisotropic order adaptivity. This will be made available soon.
# The order is set fairly high in order to be able to catch the spike in the integrand function,
# In spite of this, the total number of function evaluations will be relatively low.
PolyOrd = [200] * d

# Parameters for the tensor-train construction. I suggest to keep them as they are.
eps = 1e-10
maxvoleps = 1e-10
delta = 1e-10
tt_maxit = 50

# Problem specific definitions
center = np.array([0.2]*d)      # center of the Gaussian function
l = 0.01                        # Correlation length
params = {'center': center,
          'l': l}

plot_span = [0.,1.]

# Definition of the integrand
def f(X,params):
    if X.ndim == 1:
        return 1./(params['l'] * np.sqrt(2*np.pi)) * np.exp( - np.sum( (X - params['center'])**2. )/(2*params['l']**2.) )
    else:
        return 1./(params['l'] * np.sqrt(2*np.pi)) * np.exp( - np.sum( (X - np.tile(params['l'],(X.shape[0],1)))**2., 1 ) / (2. * np.tile(params['l'],(X.shape[0],1))**2.) )

if d == 2:
    # Plot function
    pN = 100
    px = np.linspace(plot_span[0],plot_span[1],pN)
    pX,pY = np.meshgrid(px,px)
    pZ = np.zeros(pX.shape)
    for i in range(pN):
        for j in range(pN):
            pZ[i,j] = f(np.array([pX[i,j],pY[i,j]]),params)

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(pX,pY,pZ)
    plt.title('Original')
    plt.show(block=False)

# Quadrature rule construction (No need to change here a part from the comments)
P = S1D.Poly1D(PolyType,PolyParams)
X = []
W = []
for i in range(d):
    [x,w] = P.Quadrature(PolyOrd[i],QuadType,normed=False)
    x = (x+1.)/2.               # The points are rescaled from [-1,1] to [0,1]
    X.append(x)
    w /= 2.                     # The weights are rescaled from |w|_1 = 2 to |w|_1 = 1
    W.append(w)

# The TensorWrapper is an object that wraps the user defined function and stores the computed values,
# in case they need to be reused. This object has also some features allowing to see the indices of
# the computed values and total number of function evaluations.
TW = DT.TensorWrapper(f,X,params)

# The TTapprox is the tensor-train approximation of the integrand. This is the part where the algorithm
# decide where to evaluate the function and construct the approximation. Optionally you can define a file
# where to store intermediate stages of the approximation, in order to make easier a restart of the
# algorithm.
TTapprox = None
TTapprox = DT.TTvec(TW, method='ttcross',eps=eps,lr_delta=delta,mv_eps=maxvoleps,lr_maxit=tt_maxit,lr_store_location=None)

print 'Ranks: ' + str(TTapprox.ranks())
print '# function evaluations: ' + str(TW.get_fill_level())
print 'Fill level: %.2f%%' % (float(TW.get_fill_level())/float(TW.get_size()) * 100.)

print "Plotting..."
if d == 2:
    # Plot TT approx
    ttX, ttY = np.meshgrid(X[0],X[1])
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.plot_surface(ttX,ttY,TTapprox.to_tensor())
    plt.title('TT approx.')
    plt.show(block=False)

# Plot quadrature points
if d == 2:
    # Get filled idxs
    fill_idxs = np.array(TW.get_fill_idxs())
    fig = plt.figure()
    plt.plot(fill_idxs[:,0],fill_idxs[:,1],'o')

    # Get last used idxs
    Is = TTapprox.ttcross_Is
    Js = TTapprox.ttcross_Js
    ndim = len(X)
    dims = [len(Xi) for Xi in X]
    idxs = []
    for k in range(len(Is)-1,-1,-1):
        for i in range(len(Is[k])):
            for j in range(len(Js[k])):
                for kk in range(dims[k]):
                    idxs.append( Is[k][i] + (kk,) + Js[k][j] )

    last_idxs = np.array(idxs)
    plt.plot(last_idxs[:,0],last_idxs[:,1],'or')

if d == 3:
    # Get filled idxs
    fill_idxs = np.array(TW.get_fill_idxs())
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')

    # Get last used idxs
    Is = TTapprox.ttcross_Is
    Js = TTapprox.ttcross_Js
    ndim = len(X)
    dims = [len(Xi) for Xi in X]
    idxs = []
    for k in range(len(Is)-1,-1,-1):
        for i in range(len(Is[k])):
            for j in range(len(Js[k])):
                for kk in range(dims[k]):
                    idxs.append( Is[k][i] + (kk,) + Js[k][j] )

    last_idxs = np.array(idxs)

    overlap = [np.any([np.all(i == x) for x in list(last_idxs)]) for i in list(fill_idxs)]
    notover = fill_idxs[np.logical_not(overlap),:]

    ax.scatter(notover[:,0],notover[:,1],notover[:,2],c='b',marker='o')
    ax.scatter(last_idxs[:,0],last_idxs[:,1],last_idxs[:,2],c='r',marker='o')

plt.show(block=False)

# Analytic integral
import scipy.special as scspec
if d == 2:
    Exact_int = 1./2. * l * np.sqrt(np.pi/2) * \
        (scspec.erf((-1+center[0])/(np.sqrt(2.) * l)) - scspec.erf(center[0]/(np.sqrt(2.) * l))) * \
        (scspec.erf((-1+center[1])/(np.sqrt(2.) * l)) - scspec.erf(center[1]/(np.sqrt(2) * l)))
elif d == 3:
    Exact_int = - 1./4. * l**2. * np.pi * \
        (scspec.erf((-1+center[0])/(np.sqrt(2.) * l)) - scspec.erf(center[0]/(np.sqrt(2.) * l))) * \
        (scspec.erf((-1+center[1])/(np.sqrt(2.) * l)) - scspec.erf(center[1]/(np.sqrt(2) * l))) * \
        (scspec.erf((-1+center[2])/(np.sqrt(2.) * l)) - scspec.erf(center[2]/(np.sqrt(2) * l)))

# Compute integral on the tensor-train approximation through the tensor-train contraction.
TT_int = mla.contraction(TTapprox,W)
print "Exact            TT              Error"
print "%.6e     %.6e    %.6e" % (Exact_int,TT_int, np.abs(Exact_int-TT_int))
