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
import gc
import logging
import os.path

import operator
import time

import cPickle as pkl

import random

import numpy as np
import numpy.linalg as npla
import numpy.random as npr

from scipy import stats
from scipy import sparse as scsp

import TensorToolbox as DT
import TensorToolbox.multilinalg as mla

from SpectralToolbox import Spectral1D as S1D
from SpectralToolbox import SpectralND as SND
from UQToolbox import RandomSampling as RS

import matplotlib.pyplot as plt
from matplotlib import cm
from mpl_toolkits.mplot3d import Axes3D

gc.enable()

npr.seed(3)

IS_PLOTTING = True
fsize = (6,4.5)
STORE_DATA = False
IS_STORING = False
DAT_FOLDER = "./Data-qttdmrg"
FIG_FOLDER = "./Fig-tmp-qttdmrg"
SQTT = 'SQTT'
STT = 'STT'

def test(FNUM, GenzNormalized=False, SpectralType='Projection', GenzPatrick=False):

    logging.basicConfig(filename='QTTdmrgGenz-%d-%s-%s-%s.log' % (FNUM, str(GenzNormalized), SpectralType, str(GenzPatrick)),
                        level=logging.INFO)
    # logging.basicConfig(level=logging.DEBUG)

    # SpectralType = "Projection", "LinInterp", "PolyInterp"
    if SpectralType == "Projection":
        QuadType = S1D.GAUSS
    elif SpectralType == "PolyInterp":
        QuadType = S1D.GAUSSLOBATTO

    #########################################
    # Genz functions
    #########################################
    if not GenzPatrick:
        file_name_pat = ""
        if SpectralType == "Projection": 
            sizes = np.arange(1,16)
            # sizes = [1,2,3,5,7,8]
            PolyType = S1D.JACOBI
            PolyParams = [0.,0.]
        elif SpectralType == "PolyInterp":
            sizes = np.arange(1,16)
            PolyType = S1D.JACOBI
            PolyParams = [0.,0.]
        elif SpectralType == "LinInterp":
            sizes = 2**np.arange(1,7)

        if FNUM == 0:
            FUNC = 0
            # ds = [10,50,100]
            # stt_types = [SQTT,SQTT,SQTT]
            ds = [10]
            stt_types = [SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 1:
            FUNC = 1
            ds = [10,15,20]
            stt_types = [SQTT,SQTT,SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 2:
            FUNC = 2
            ds = [5,10,15]
            stt_types = [SQTT,SQTT,SQTT]
            # ds = [10]
            # stt_types = [SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 3:
            FUNC = 3
            ds = [10,50,100]
            stt_types = [SQTT,SQTT,SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 4:
            FUNC = 4
            ds = [10,50,100]
            stt_types = [SQTT,SQTT,SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 5:
            FUNC = 5
            ds = [10,15,20]
            stt_types = [SQTT,SQTT,SQTT]
            kickrank = None
            eps = 1e-10

    if GenzPatrick:
        file_name_pat = "Patrick"
        GenzNormalized = True

        if SpectralType == "Projection": 
            sizes = np.arange(1,16)
            # sizes = [1,2,3,7,8,15,26,31]
            PolyType = S1D.JACOBI
            PolyParams = [0.,0.]
        elif SpectralType == "PolyInterp":
            sizes = np.arange(1,16)
            PolyType = S1D.JACOBI
            PolyParams = [0.,0.]
        elif SpectralType == "LinInterp":
            sizes = 2**np.arange(1,7)

        if FNUM == 0:
            FUNC = 0
            ds = [5]
            stt_types = [SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 1:
            FUNC = 1
            ds = [5]
            stt_types = [SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 2:
            FUNC = 2
            ds = [5]
            stt_types = [SQTT]
            kickrank = None
            eps = 1e-10
        elif FNUM == 3:
            FUNC = 3
            ds = [5]
            stt_types = [SQTT]
            kickrank = None
            eps = 1e-10

    print "Function: " + str(FUNC) + " Norm: " + str(GenzNormalized) + " Dims: " + str(ds) + " Type: " + SpectralType
    
    colmap = plt.get_cmap('jet')
    cols = [colmap(i) for i in np.linspace(0, 1.0, len(ds))]
    N_EXP = 30
    xspan = [0.,1.]

    maxvoleps = 1e-10
    tt_maxit=100

    if not GenzNormalized:
        file_name_ext = "New"
    else: 
        file_name_ext = ""

    MCestVarLimit = 1e-1
    MCestMinIter = 100
    MCestMaxIter = 1e6
    MCstep = 10000

    names = ["Oscillatory","Product Peak","Corner Peak", "Gaussian", "Continuous", "Discontinuous"]
    file_names = ["Oscillatory","ProductPeak","CornerPeak", "Gaussian", "Continuous", "Discontinuous"]

    # Try to load data if already there
    if GenzPatrick:
        path = DAT_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "data-Patrick.pkl"
    else:
        path = DAT_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "data.pkl"
        
    if os.path.isfile(path):
        infile = open(path,'rb')
        data = pkl.load(infile)
        infile.close()
        
        L2err = data['L2err']
        feval = data['feval']
    else:
        L2err = np.empty((len(ds),N_EXP,len(sizes)),dtype=object)
        feval = np.empty((len(ds),N_EXP,len(sizes)),dtype=object)

    for i_d, (d,stt_type) in enumerate(zip(ds,stt_types)):
        for n_exp in range(N_EXP):
            vol = (xspan[1]-xspan[0])**d
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

            if d == 2:
                # Plot function
                pN = 40
                px = np.linspace(0.,1.,pN)
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

            for i_size, size in enumerate(sizes):
                if L2err[i_d,n_exp,i_size] == None or np.isnan(L2err[i_d,n_exp,i_size]) :
                    size1D = size
                    size = [size1D for i in range(d)]

                    if SpectralType == "Projection":
                        surr_type = DT.PROJECTION
                    elif SpectralType == "LinInterp":
                        surr_type = DT.LINEAR_INTERPOLATION
                    elif SpectralType == "PolyInterp":
                        surr_type = DT.LAGRANGE_INTERPOLATION

                    X = []
                    for i in range(d):
                        if SpectralType == "Projection":
                            X.append( (PolyType, QuadType, PolyParams, xspan) )
                        elif SpectralType == "LinInterp":
                            X.append( np.linspace(xspan[0],xspan[1],size[i]) )
                        elif SpectralType == "PolyInterp":
                            X.append( (PolyType, QuadType, PolyParams, xspan) )

                    dims = size

                    print "---------------------------------------------------"
                    print "Run test: d=%d, n_exp=%d, ord=%d" % ( d, n_exp, size1D )

                    if stt_type == STT:
                        STTapprox = DT.STT(f, X, params, orders=size,
                                           method='ttdmrg', eps=eps, mv_eps=maxvoleps,maxit=tt_maxit,
                                           kickrank=None,
                                           surrogateONOFF=True, surrogate_type=surr_type)
                    elif stt_type == SQTT:
                        STTapprox = DT.SQTT(f, X, params, orders=size,
                                            method='ttdmrg', eps=eps, mv_eps=maxvoleps,maxit=tt_maxit,
                                            kickrank=None,
                                            surrogateONOFF=True, surrogate_type=surr_type)

                    STTapprox.build()
                    print STTapprox.TTapprox[0].ranks()

                    if d == 2 and IS_PLOTTING:
                        # Plot TT approx
                        ttX, ttY = np.meshgrid(X[0],X[1])
                        fig = plt.figure()
                        ax = fig.add_subplot(111, projection='3d')
                        ax.plot_surface(ttX,ttY,STTapprox.TTapprox[0].to_tensor())
                        plt.title('TT approx.')
                        plt.show(block=False)

                    feval[i_d,n_exp,i_size] = STTapprox.TW.get_fill_level()

                    # Estimate L2error using Monte Carlo method
                    VarI = 1.
                    dist = stats.uniform()
                    DIST = RS.MultiDimDistribution([dist] * d)
                    intf = []
                    values = []
                    multi = 1
                    while (len(values) < MCestMinIter or VarI > MCestVarLimit) and len(values) < MCestMaxIter:
                        MCstep_loop = multi * MCstep - len(values)
                        multi *= 2
                        # Monte Carlo
                        xx = np.asarray( DIST.rvs(MCstep_loop) )

                        TTvals = STTapprox(xx)

                        fval = f(xx,params)

                        intf.extend( list(fval**2.) )
                        values.extend( list( (np.asarray(fval)-np.asarray(TTvals))**2. ) )


                        EstI = vol * np.mean( np.asarray(values) )
                        VarI = (vol**2. * np.var( np.asarray(values) ) / float(len(values))) # / EstI**2.

                        # EstI = vol * np.mean( np.asarray(values) / np.asarray(intf) )
                        # VarI = (vol**2. * np.var( np.asarray(values) / np.asarray(intf) ) / float(len(values)) )

                        # mean = vol * np.mean(values)
                        # var = vol**2. * np.var(values) / len(values)

                        sys.stdout.write("L2err estim. iter: %d Var: %e VarLim: %e L2err: %e Feval: %d \r" % (len(values) , VarI,  MCestVarLimit, np.sqrt(EstI)/np.sqrt(np.mean(intf)), feval[i_d,n_exp,i_size] ))
                        sys.stdout.flush()

                    sys.stdout.write("\n")
                    sys.stdout.flush()
                    if GenzPatrick:
                        L2err[i_d,n_exp,i_size] = np.sqrt( EstI )
                    else:
                        L2err[i_d,n_exp,i_size] = np.sqrt( EstI )/np.sqrt(np.mean(intf))

                    # Free memory
                    STTapprox = None
                    gc.collect()

                    data = {'feval': feval,
                            'L2err': L2err}
                    if GenzPatrick:
                        path = DAT_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "data-Patrick.pkl"
                    else:
                        path = DAT_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "data.pkl"
                    
                    if STORE_DATA:
                        outfile = open(path,'wb')
                        pkl.dump(data,outfile)
                        outfile.close()


    if IS_PLOTTING:
        plt.figure(figsize=fsize)
        for i_d,d in enumerate(ds):
            for n_exp in range(N_EXP):
                plt.semilogy(sizes,L2err[i_d,n_exp,:],'.',color=cols[i_d])
            plt.semilogy(sizes,np.mean(L2err[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
        plt.xlabel('Order')
        plt.ylabel('L2err')
        plt.subplots_adjust(bottom=0.15)
        plt.grid(True)
        plt.title(names[FUNC])
        plt.legend(loc='best')
        if IS_STORING:
            path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "OrdVsL2err"
            plt.savefig(path + ".png", format="png")
            plt.savefig(path + ".pdf", format="pdf")
            plt.savefig(path + ".ps", format="ps")

        plt.figure(figsize=fsize)
        for i_d,d in enumerate(ds):
            for n_exp in range(N_EXP):
                plt.semilogy(feval[i_d,n_exp,:],L2err[i_d,n_exp,:],'.',color=cols[i_d])
            plt.semilogy(np.mean(feval[i_d,:,:],axis=0),np.mean(L2err[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
        plt.xlabel('# func. eval')
        plt.ylabel('L2err')
        plt.subplots_adjust(bottom=0.15)
        plt.grid(True)
        plt.title(names[FUNC])
        plt.legend(loc='best')

        plt.figure(figsize=fsize)
        for i_d,d in enumerate(ds):
            for n_exp in range(N_EXP):
                plt.loglog(feval[i_d,n_exp,:],L2err[i_d,n_exp,:],'.',color=cols[i_d])
            plt.loglog(np.mean(feval[i_d,:,:],axis=0),np.mean(L2err[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
        plt.xlabel('# func. eval')
        plt.ylabel('L2err')
        plt.subplots_adjust(bottom=0.15)
        plt.grid(True)
        plt.title(names[FUNC])
        plt.legend(loc='best')
        if IS_STORING:
            path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "convergence"
            plt.savefig(path + ".png", format="png")
            plt.savefig(path + ".pdf", format="pdf")
            plt.savefig(path + ".ps", format="ps")

        plt.figure(figsize=fsize)
        for i_d,d in enumerate(ds):
            for n_exp in range(N_EXP):
                plt.semilogy(sizes,feval[i_d,n_exp,:],'.',color=cols[i_d])
            plt.semilogy(sizes,np.mean(feval[i_d,:,:],axis=0),'o-',color=cols[i_d],label='d=%d' % d)
        plt.xlabel('Order')
        plt.ylabel('# func. eval')
        plt.subplots_adjust(bottom=0.15)
        plt.grid(True)
        plt.title(names[FUNC])
        plt.legend(loc='best')
        if IS_STORING:
            path = FIG_FOLDER + "/" + SpectralType + "-" + file_names[FUNC] + file_name_ext + "-" + "OrdVsFeval"
            plt.savefig(path + ".png", format="png")
            plt.savefig(path + ".pdf", format="pdf")
            plt.savefig(path + ".ps", format="ps")

        plt.show(block=False)

if __name__ == "__main__":
    FNUM = int(sys.argv[1])
    GenzNorm = (sys.argv[2] == 'True')
    Type = sys.argv[3]
    if len(sys.argv) < 5:
        GenzPatrick = False
    else:
        GenzPatrick = (sys.argv[4] == 'True')
    test(FNUM,GenzNorm,Type,GenzPatrick)
