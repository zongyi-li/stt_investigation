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

from scipy import stats

from UQToolbox import RandomSampling as RS

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

def runner(N, sampler, PLOTTING=False):
    # Dimension of the function. The script works for any dimension, but for d=2,3 
    # some plotting is provided and the error estimation can be done against the
    # analytical value of the integral
    d = 3

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
    
    # Run experiments
    experiments = RS.Experiments( f, params, [stats.uniform()] * d)
    experiments.sample(N, sampler)
    experiments.run()

    # Compute integral
    UQ_int = np.mean( experiments.get_results() )
    err = np.abs(Exact_int-UQ_int)
    print("Exact            UQ              Error")
    print("%.6e     %.6e    %.6e" % (Exact_int,UQ_int,err))
    
    return (N, err)

if __name__ == "__main__":
    (feval, err) = runner(10000, 'mc', True)
