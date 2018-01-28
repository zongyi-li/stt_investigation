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
import sys
import time as systime
import numpy as np
import numpy.linalg as npla
import scipy.stats as stats
from SpectralToolbox import Spectral1D as S1D
import TensorToolbox as TT
import os
import os.path
from TensorToolbox.unittests.auxiliary import print_ok, print_fail, print_summary

# For this example see also the documentation of the class STT

def run(maxprocs, loglev=logging.WARNING):

    logging.basicConfig(level=loglev)

    store_file = '0d'           # Storage file (used for storage and restarting purposes

    if os.path.exists(store_file + ".pkl"): os.remove(store_file + '.pkl')
    if os.path.exists(store_file + ".pkl.old"): os.remove(store_file + ".pkl.old")
    if os.path.exists(store_file + ".h5"): os.remove(store_file + '.h5')
    if os.path.exists(store_file + ".h5.old"): os.remove(store_file + ".h5.old")

    nsucc = 0
    nfail = 0

    orders = range(2,15)

    for order in orders:
        # Function definition.
        # For MPI purposes, that cannot be imported from the system path (like user defined functions
        # used by f) MUST be defined inside the following definition. Whatever can be imported from
        # the system path (like numpy), can be imported from inside it.
        def f(p,params):
            import numpy as np
            return 1./(np.sum(p) + 1.)

        params = None               # Parameters to be passed to function f. In this case None.
                                    # They must be Pickable

        eps = 1e-8
        kickrank = None             # Improves convergence for high rank functions
        store_freq = 10             # Seconds between every storage
        surr_type = TT.PROJECTION   # Alternatives are: TT.PROJECTION, TT.LINEAR_INTERPOLATION
                                    #                   and TT.LAGRANGE_INTERPOLATION

        d = 4                       # Number of dimensions (The analytical integrals work for d=4)
        orders = [order] * d        # orders of the approximation
        X = []                      # Approximation points
        for i in range(d):
            if surr_type == TT.PROJECTION:
                # For each dimension, define: the polynomial type, the type of points, the polynomial parameters and the span over which the points will be rescaled
                X.append( (S1D.JACOBI, S1D.GAUSSLOBATTO, (0.,0.), [0.,1.]) )
            else:
                # In the LINEAR_INTERPOLATION and LAGRANGE_INTERPOLATION case any list of points in the span will work (Include the endpoints!).
                X.append( np.linspace(0.,1.,40) )

        if os.path.isfile(store_file + '.pkl'): # If file already exists, then restart the construction from
                                       # already computed values
            STTapprox = TT.load(store_file,load_data=True)

            STTapprox.set_f(f)
            STTapprox.store_freq = store_freq
            STTapprox.build(maxprocs)

        else:                       # Otherwise start a new approximation
            STTapprox = TT.SQTT(f, X, params, eps=eps, range_dim=0, orders=orders, method='ttdmrg', kickrank=kickrank, surrogateONOFF=True, surrogate_type=surr_type, store_location=store_file, store_overwrite=True, store_freq=store_freq) # See documentation of STT
            STTapprox.build(maxprocs)

        logging.info("Fill: %.3f%% - N: %e" % (float(STTapprox.TW.get_fill_level())/float(STTapprox.TW.get_global_size()) * 100., float(STTapprox.TW.get_fill_level())))

        def eval_point(STTapprox,params):
            # Evaluates a point x with the STT approximation and the analytical function
            # Compare times and error
            N = 1000
            xs = stats.uniform().rvs(N*d).reshape(N,d)

            # Evaluate a point
            val = np.zeros(N)
            start_eval = systime.clock()
            for i in range(N):
                val[i] = STTapprox(xs[i,:])      # Point evaluation in TT format
            end_eval = systime.clock()
            logging.info("Evaluation time: " + str((end_eval-start_eval)/N))

            exact = np.zeros(N)
            start_eval = systime.clock()
            for i in range(N):
                exact[i] = f(xs[i,:],params)
            end_eval = systime.clock()
            logging.info("Exact evaluation time: " + str((end_eval-start_eval)/N))

            err = np.sqrt(np.mean(val-exact)**2.)
            return err

        # Point evaluation
        point_err = eval_point(STTapprox,params)

        # Computing the mean
        mean = STTapprox.integrate()
        exact_mean = -272. * np.log(2)/3. + 27. * np.log(3) + 125. * np.log(5) / 6.
        mean_err = np.abs(mean-exact_mean)

        # Computing the variance
        var = (STTapprox**2).integrate() - mean**2. # Mind that in the expression STTapprox**2,
                                                    # 2 is an integer!

        exact_var = np.log(4722366482869645213696./(1861718135983154296875. * np.sqrt(5))) - exact_mean**2.
        var_err = np.abs(var-exact_var)

        print_ok("Spectral Quantics Tensor Train DMRG - 0D - Ord: %d - Point err: %e - Mean err: %e - Var err: %e" % (order,point_err, mean_err, var_err))
        nsucc += 1

        if os.path.exists(store_file + '.pkl'): os.remove(store_file + '.pkl')
        if os.path.exists(store_file + ".pkl.old"): os.remove(store_file + ".pkl.old")
        if os.path.exists(store_file + '.h5'): os.remove(store_file + '.h5')
        if os.path.exists(store_file + ".h5.old"): os.remove(store_file + ".h5.old")
    
    print_summary("SQTTdmrg", nsucc, nfail)
    
    return (nsucc,nfail)

if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestSQTTdmrg_0D.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs,loglev=logging.INFO)

