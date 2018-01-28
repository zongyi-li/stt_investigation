# -*- coding: utf-8 -*-

#!/usr/bin/env python

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

from TensorToolbox.core import auxiliary
from TensorToolbox.core.auxiliary import *
from TensorToolbox.core import storage
from TensorToolbox.core.storage import *
from TensorToolbox.core import tensor_wrapper
from TensorToolbox.core.tensor_wrapper import *
from TensorToolbox.core import candecomp
from TensorToolbox.core.candecomp import *
from TensorToolbox.core import TensorTrainVec
from TensorToolbox.core.TensorTrainVec import *
from TensorToolbox.core import WeightedTensorTrainVec
from TensorToolbox.core.WeightedTensorTrainVec import *
from TensorToolbox.core import TensorTrainMat
from TensorToolbox.core.TensorTrainMat import *
from TensorToolbox.core import QuanticsTensorTrainVec
from TensorToolbox.core.QuanticsTensorTrainVec import *
from TensorToolbox.core import QuanticsTensorTrainMat
from TensorToolbox.core.QuanticsTensorTrainMat import *
from TensorToolbox.core import WeightedQuanticsTensorTrainVec
from TensorToolbox.core.WeightedQuanticsTensorTrainVec import *
from TensorToolbox.core import SpectralTensorTrain 
from TensorToolbox.core.SpectralTensorTrain import *

__all__ = []
__all__ += auxiliary.__all__
__all__ += storage.__all__
__all__ += tensor_wrapper.__all__
__all__ += candecomp.__all__
__all__ += TensorTrainVec.__all__
__all__ += WeightedTensorTrainVec.__all__
__all__ += TensorTrainMat.__all__
__all__ += QuanticsTensorTrainVec.__all__
__all__ += QuanticsTensorTrainMat.__all__
__all__ += WeightedQuanticsTensorTrainVec.__all__
__all__ += SpectralTensorTrain.__all__

__author__ = "Daniele Bigoni"
__copyright__ = """Copyright 2014, The Technical University of Denmark"""
__credits__ = ["Daniele Bigoni"]
__maintainer__ = "Daniele Bigoni"
__email__ = "dabi@imm.dtu.dk"
__status__ = "Production"
