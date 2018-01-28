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

__all__ = ['WQTTvec']

import sys
import logging 

from TensorToolbox.core import TTvec, QTTvec, WTTvec

class WQTTvec(QTTvec, WTTvec):
    """ Constructor of multidimensional tensor in Weighted Tensor Train format
    
    :param Candecomp,ndarray,TT,TensorWrapper A: Available input formats are Candecomp, full tensor in numpy.ndarray, Tensor Train structure (list of cores), or a Tensor Wrapper.
    :param list W: list of 1-dimensional ndarray containing the weights for each dimension.
    :param int base: base selected to do the folding
    :param string store_location: Store computed values during construction on the specified file path. The stored values are ttcross_Jinit and the values used in the TensorWrapper. This permits a restart from already computed values. If empty string nothing is done. (method=='ttcross')
    :param string store_object: Object to be stored (default are the tensor wrapper and ttcross_Jinit)
    :param int store_freq: storage frequency. ``store_freq==1`` stores intermediate values at every iteration. The program stores data every ``store_freq`` internal iterations. If ``store_object`` is a SpectralTensorTrain, then ``store_freq`` determines the number of seconds every which to store values.
    :param int multidim_point: If the object A returns a multidimensional array, then this can be used to define which point to apply ttcross to.
    """

    logger = logging.getLogger(__name__)
    logger.propagate = False
    ch = logging.StreamHandler(sys.stdout)
    formatter = logging.Formatter("%(asctime)s %(levelname)s:%(name)s: %(message)s",
                                  "%Y-%m-%d %H:%M:%S")
    ch.setFormatter(formatter)
    logger.addHandler(ch)

    def __init__(self,A, W, base=2,
                 store_location="",store_object=None,store_freq=1, store_overwrite=False, 
                 multidim_point=None):
        TTvec.__init__(self, A, store_location=store_location, store_object=store_object,
                       store_freq=store_freq, store_overwrite=store_overwrite,
                       multidim_point=multidim_point)
        WTTvec._init(self, W)
        QTTvec._init(self, base)

    def __getstate__(self):
        return TTvec.__getstate__(self)

    def __setstate__(self, state):
        TTvec.__setstate__(state)

    def build(self, eps=1e-10, method='svd', rs=None, fix_rank=False, Jinit=None,
              delta=1e-4, maxit=100, mv_eps=1e-6, mv_maxit=100, max_ranks=None,
              kickrank=None):
        """ Common interface for the construction of the approximation.

        :param float eps: [default == 1e-10] For method=='svd': precision with which to approximate the input tensor. For method=='ttcross': TT-rounding tolerance for rank-check.
        :param string method: 'svd' use singular value decomposition to construct the TT representation :cite:`Oseledets2011`, 'ttcross' use low rank skeleton approximation to construct the TT representation :cite:`Oseledets2010`, 'ttdmrg' uses Tensor Train Renormalization Cross to construct the TT representation :cite:`Savostyanov2011,Savostyanov2013`, 'ttdmrgcross' uses 'ttdmrg' with 'ttcross' approximation of supercores
        :param list rs: list of integer ranks of different cores. If ``None`` then the incremental TTcross approach will be used. (method=='ttcross')
        :param bool fix_rank: determines whether the rank is allowed to be increased (method=='ttcross')
        :param list Jinit: list of list of integers containing the r starting columns in the lowrankapprox routine for each core. If ``None`` then pick them randomly. (method=='ttcross')
        :param float delta: accuracy parameter in the TT-cross routine (method=='ttcross'). It is the relative error in Frobenious norm between two successive iterations.
        :param int maxit: maximum number of iterations in the lowrankapprox routine (method=='ttcross')
        :param float mv_eps: accuracy parameter for each usage of the maxvol algorithm (method=='ttcross')
        :param int mv_maxit: maximum number of iterations in the maxvol routine (method=='ttcross')
        :param bool fix_rank: Whether the rank is allowed to increase
        :param list max_ranks: Maximum ranks to be used to limit the trunaction rank due to ``eps``. The first and last elements of the list must be ``1``, e.g. ``[1,...,1]``. Default: ``None``.
        :param int kickrank: rank overshooting for 'ttdmrg'

        .. note:: Weights are not removed after computation, because cannot be trivially
           removed from the folded qauntics approximation! The weights need to be
           removed manually. For example:

           >>> wqtt.build()
           >>> wtt = wqtt.to_TTvec()
           >>> wtt.remove_weights()
        """
        WTTvec._build_preprocess(self)
        TTvec.build(self, eps=eps, method=method, rs=rs, fix_rank=fix_rank,
                    Jinit=Jinit, delta=delta, maxit=maxit,
                    mv_eps=mv_eps, mv_maxit=mv_maxit,
                    max_ranks=max_ranks, kickrank=kickrank)
        QTTvec._build_postprocess(self)
        WTTvec._build_postprocess(self)
        return self

    def to_TTvec(self):
        ttvec = QTTvec.to_TTvec(self)
        return WTTvec(ttvec.TT, self.W).build()