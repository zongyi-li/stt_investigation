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

from TensorToolbox.unittests.auxiliary import bcolors, print_ok, print_fail, print_summary

__all__ = ['RunTestWTT0','RunTestWTT2','RunTestWTT4','RunTestWTT5','RunTestWTT6','RunTestWTT7']

def RunTestWTT0(maxprocs=None, PLOTTING=False, loglev=logging.WARNING):
    from TensorToolbox.unittests import TestWTT_0
    return TestWTT_0.run(maxprocs,PLOTTING,loglev)

def RunTestWTT2(maxprocs=None, PLOTTING=False, loglev=logging.WARNING):
    from TensorToolbox.unittests import TestWTT_2
    return TestWTT_2.run(maxprocs,PLOTTING,loglev)

def RunTestWTT4(maxprocs=None, PLOTTING=False, loglev=logging.WARNING):
    from TensorToolbox.unittests import TestWTT_4
    return TestWTT_4.run(maxprocs,PLOTTING,loglev)

def RunTestWTT5(maxprocs=None, PLOTTING=False, loglev=logging.WARNING):
    from TensorToolbox.unittests import TestWTT_5
    return TestWTT_5.run(maxprocs,PLOTTING,loglev)

def RunTestWTT6(maxprocs=None, PLOTTING=False, loglev=logging.WARNING):
    from TensorToolbox.unittests import TestWTT_6
    return TestWTT_6.run(maxprocs,PLOTTING,loglev)

def RunTestWTT7(maxprocs=None, PLOTTING=False, loglev=logging.WARNING):
    from TensorToolbox.unittests import TestWTT_7
    return TestWTT_7.run(maxprocs,PLOTTING,loglev)

def run(maxprocs, PLOTTING=False, loglev=logging.WARNING):

    logging.basicConfig(level=loglev)

    import numpy as np
    import numpy.linalg as npla
    import itertools
    import time

    import TensorToolbox as DT
    import TensorToolbox.multilinalg as mla

    RUNTESTS = [0,2]

    if PLOTTING:
        from matplotlib import pyplot as plt

    nsucc = 0
    nfail = 0

    if (0 in RUNTESTS):
        (ns,nf) = RunTestWTT0(maxprocs,PLOTTING,loglev)
        nsucc += ns
        nfail += nf

    if (2 in RUNTESTS):
        (ns,nf) = RunTestWTT2(maxprocs,PLOTTING,loglev)
        nsucc += ns
        nfail += nf

    if (4 in RUNTESTS):
        (ns,nf) = RunTestWTT4(maxprocs,PLOTTING,loglev)
        nsucc += ns
        nfail += nf

    if (5 in RUNTESTS):
        (ns,nf) = RunTestWTT5(maxprocs,PLOTTING,loglev)
        nsucc += ns
        nfail += nf

    if (6 in RUNTESTS):
        (ns,nf) = RunTestWTT6(maxprocs,PLOTTING,loglev)
        nsucc += ns
        nfail += nf

    if (7 in RUNTESTS):
        (ns,nf) = RunTestWTT7(maxprocs,PLOTTING,loglev)
        nsucc += ns
        nfail += nf

    print_summary("WTT General", nsucc, nfail)
    
    return (nsucc,nfail)

if __name__ == "__main__":
    # Number of processors to be used, defined as an additional arguement 
    # $ python TestWTT.py N
    # Mind that the program in this case will run slower than the non-parallel case
    # due to the overhead for the creation and deletion of threads.
    if len(sys.argv) == 2:
        maxprocs = int(sys.argv[1])
    else:
        maxprocs = None

    run(maxprocs,PLOTTING=True, loglev=logging.INFO)
