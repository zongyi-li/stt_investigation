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

__all__ = ['RunUnitTests','RunTestTT','RunTestWTT','RunTestTensorWrapper','RunTestWeightedTensorWrapper','RunTestTTcross','RunTestTTdmrg','RunTestWTTdmrg','RunTestTTdmrgcross','RunTestQTTdmrg','RunTestQTT','RunTestSTTcross_0D', 'RunTestSTTcross_2D','RunTestSTTdmrg_0D', 'RunTestSTTdmrg_2D','RunTestSTTdmrgcross_0D','RunTestSQTTdmrg_0D']

def RunTestTT(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestTT
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    :returns: tuple (n success,n fail)
    """
    from TensorToolbox.unittests import TestTT
    return TestTT.run(maxprocs, loglev=loglev)

def RunTestWTT(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestWTT
    
    :params int maxprocs: If MPI support is enabled, defines how many processors to use.
    :returns: tuple (n success,n fail)
    """
    from TensorToolbox.unittests import TestWTT
    return TestWTT.run(maxprocs, loglev=loglev)

def RunTestTensorWrapper(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestTensorWrapper
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    :returns: tuple (n success,n fail)
    """
    from TensorToolbox.unittests import TestTensorWrapper
    return TestTensorWrapper.run(maxprocs,loglev=loglev)

def RunTestWeightedTensorWrapper(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestWeightedTensorWrapper
    
    :params int maxprocs: If MPI support is enabled, defines how many processors to use.
    :returns: tuple (n success,n fail)
    """
    from TensorToolbox.unittests import TestWeightedTensorWrapper
    return TestWeightedTensorWrapper.run(maxprocs,loglev=loglev)

def RunTestTTcross(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestTTcross
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestTTcross
    return TestTTcross.run(maxprocs, loglev=loglev)
   
def RunTestTTdmrg(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestTTdmrg
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestTTdmrg
    return TestTTdmrg.run(maxprocs, loglev=loglev)

def RunTestWTTdmrg(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestWTTdmrg
    
    :params int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestWTTdmrg
    return TestWTTdmrg.run(maxprocs, loglev=loglev)

def RunTestTTdmrgcross(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestTTdmrg
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestTTdmrgcross
    return TestTTdmrgcross.run(maxprocs, loglev=loglev)

def RunTestQTTdmrg(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestQTTdmrg
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestQTTdmrg
    return TestQTTdmrg.run(maxprocs, loglev=loglev)

def RunTestQTT(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestQTT
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestQTT
    return TestQTT.run(maxprocs, loglev=loglev)

def RunTestSTTcross_0D(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestSTTcross_0D
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestSTTcross_0D
    return TestSTTcross_0D.run(maxprocs, loglev=loglev)

def RunTestSTTcross_2D(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestSTTcross_2D
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestSTTcross_2D
    return TestSTTcross_2D.run(maxprocs, loglev=loglev)

def RunTestSTTdmrg_0D(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestSTTdmrg_0D
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestSTTdmrg_0D
    return TestSTTdmrg_0D.run(maxprocs, loglev=loglev)

def RunTestSTTdmrg_2D(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestSTTdmrg_2D
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestSTTdmrg_2D
    return TestSTTdmrg_2D.run(maxprocs, loglev=loglev)

def RunTestSTTdmrgcross_0D(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestSTTdmrgcross_0D
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestSTTdmrgcross_0D
    return TestSTTdmrgcross_0D.run(maxprocs, loglev=loglev)

def RunTestSQTTdmrg_0D(maxprocs=None, loglev=logging.WARNING):
    """ Runs the TestSTTdmrg_0D
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    from TensorToolbox.unittests import TestSQTTdmrg_0D
    return TestSQTTdmrg_0D.run(maxprocs, loglev=loglev)

def RunUnitTests(maxprocs=None,loglev=logging.WARNING):
    """ Runs all the unit tests.
    
    :param int maxprocs: If MPI support is enabled, defines how many processors to use.
    """
    
    from TensorToolbox.unittests.auxiliary import print_summary

    nsucc = 0
    nfail = 0

    (ns,nf) = RunTestTensorWrapper(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestWeightedTensorWrapper(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestTT(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestWTT(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestQTT(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestTTcross(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestTTdmrg(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestWTTdmrg(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestTTdmrgcross(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestQTTdmrg(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestSTTcross_0D(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestSTTcross_2D(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestSTTdmrg_0D(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    # (ns,nf) = RunTestSTTdmrg_2D(maxprocs,loglev=loglev)   # Need to fix restarting

    (ns,nf) = RunTestSTTdmrgcross_0D(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    (ns,nf) = RunTestSQTTdmrg_0D(maxprocs,loglev=loglev)
    nsucc += ns
    nfail += nf

    print_summary("TT ALL", nsucc, nfail)

    if nfail > 0:
        sys.exit(1)
    
    return (nsucc,nfail)
