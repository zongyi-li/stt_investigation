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
from scipy import stats
import os.path

from SpectralToolbox import Spectral1D as S1D
from UQToolbox import RandomSampling as RS
import TensorToolbox as TT

from TensorToolbox.unittests.auxiliary import print_ok, print_fail, print_summary

def run(maxprocs, PLOTTING=False, loglev=logging.WARNING):

    logging.basicConfig(level=loglev)

    store_file = '2d'           # Storage file (used for storage and restarting purposes

    if os.path.exists(store_file + ".pkl"): os.remove(store_file + '.pkl')
    if os.path.exists(store_file + ".pkl.old"): os.remove(store_file + ".pkl.old")
    if os.path.exists(store_file + ".h5"): os.remove(store_file + '.h5')
    if os.path.exists(store_file + ".h5.old"): os.remove(store_file + ".h5.old")
    
    nsucc = 0
    nfail = 0

    def f(p,params):
        import numpy as np
        XX = params['XX']
        YY = params['YY']
        return np.sin(np.pi *(XX+YY)) * 1./( (1. + np.sin(2*np.pi*(XX+YY))) * np.sum(p) + 1.)

    x = np.linspace(0,1,11)
    y = np.linspace(0,1,11)
    XX,YY = np.meshgrid(x,y)
    params = {'XX': XX, 'YY': YY}

    d = 8
    ord = 10
    store_freq = 20

    orders = [20] * d
    X = [x, y]
    for i in range(d):
        X.append( (S1D.JACOBI, S1D.GAUSS, (0.,0.), [0.,1.]) )

    if os.path.isfile(store_file + '.pkl'):
        STTapprox = TT.load(store_file,load_data=True)

        STTapprox.set_f(f)
        STTapprox.store_freq = store_freq
        STTapprox.build(maxprocs)
    else:
        STTapprox = TT.STT(f, X, params, range_dim=2, orders=orders, method='ttcross', surrogateONOFF=True, surrogate_type=TT.PROJECTION, store_location=store_file, store_overwrite=True, store_freq=store_freq)
        STTapprox.build(maxprocs)

    def eval_point(STTapprox,x,params,plotting=False):
        XX = params['XX']
        YY = params['YY']

        # Evaluate a point
        start_eval = systime.clock()
        val = STTapprox(x)
        end_eval = systime.clock()
        logging.info("TestSTTcross_2D: Evaluation time: " + str(end_eval-start_eval))

        start_eval = systime.clock()
        exact = f(x,params)
        end_eval = systime.clock()
        logging.info("TestSTTcross_2D: Exact evaluation time: " + str(end_eval-start_eval))

        logging.info("TestSTTcross_2D: Pointwise L2err: " + str(npla.norm( (val-exact).flatten(),2 )))

        if plotting:
            import matplotlib.pyplot as plt
            from mpl_toolkits.mplot3d import Axes3D
            from matplotlib import cm

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(XX,YY,val, rstride=1, cstride=1, cmap=cm.coolwarm, \
                                   linewidth=0, antialiased=False)
            plt.title("Surrogate")
            plt.show(block=False)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(XX,YY,exact, rstride=1, cstride=1, cmap=cm.coolwarm, \
                                   linewidth=0, antialiased=False)
            plt.title("Exact")
            plt.show(block=False)

            fig = plt.figure()
            ax = fig.add_subplot(111, projection='3d')
            surf = ax.plot_surface(XX,YY,np.abs(exact-val), rstride=1, cstride=1, cmap=cm.coolwarm, \
                                   linewidth=0, antialiased=False)
            plt.title("Error")
            plt.show(block=False)

    eval_point(STTapprox,np.array([0.2]*d),params,plotting=PLOTTING)

    # Estimate mean error:
    DIST = RS.MultiDimDistribution([ stats.uniform() ] * d)
    exact = []
    MCstep = 100
    # Sampling
    xx = np.asarray( DIST.rvs(MCstep) )
    # STT evaluation
    STTvals = STTapprox(xx)
    # Exact evaluation
    exact = np.asarray( [ f(xx[i,:],params) for i in range(MCstep) ] )
    # Mean abs error
    abs_err = np.abs(STTvals - exact)
    mean_abs_err = np.mean(abs_err, axis=0)
    var_abs_err = np.var(abs_err, axis=0)
    if PLOTTING:
        import matplotlib.pyplot as plt
        from mpl_toolkits.mplot3d import Axes3D
        from matplotlib import cm

        XX = params['XX']
        YY = params['YY']
        
        fig = plt.figure()
        ax = fig.add_subplot(121, projection='3d')
        surf = ax.plot_surface(XX,YY,mean_abs_err,rstride=1, cstride=1, cmap=cm.coolwarm, \
                               linewidth=0, antialiased=False)
        plt.title("Mean abs error")
        ax = fig.add_subplot(122, projection='3d')
        surf = ax.plot_surface(XX,YY,var_abs_err,rstride=1, cstride=1, cmap=cm.coolwarm, \
                               linewidth=0, antialiased=False)
        plt.title("Variance abs error")
        plt.show(block=False)
    
    # Mean L2 error
    L2_err = npla.norm( STTvals - exact, ord=2, axis=(1,2) )
    mean_L2_err = np.mean(L2_err)
    var_L2_err = np.var(L2_err)
    logging.info("TestSTTcross_2D: Mean L2 Err = %e , Variance L2 Err = %e" % (mean_L2_err,var_L2_err))

    os.remove(store_file + '.pkl')
    if os.path.exists(store_file + ".pkl.old"): os.remove(store_file + ".pkl.old")
    os.remove(store_file + '.h5')
    if os.path.exists(store_file + ".h5.old"): os.remove(store_file + ".h5.old")

    print_ok("STTcross 2D")
    nsucc += 1

    print_summary("STTcross 2D", nsucc, nfail)
    
    return (nsucc,nfail)

if __name__ == "__main__":
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs, PLOTTING=True, loglev=logging.INFO)

