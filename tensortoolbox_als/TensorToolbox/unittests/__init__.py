# -*- coding: utf-8 -*-

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

from TensorToolbox.unittests import run_unit_tests
from TensorToolbox.unittests.run_unit_tests import *
from TensorToolbox.unittests import TestTT
from TensorToolbox.unittests.TestTT import *
from TensorToolbox.unittests import TestWTT
from TensorToolbox.unittests.TestWTT import *


__all__ = []
__all__ += run_unit_tests.__all__
__all__ += TestTT.__all__
__all__ += TestWTT.__all__

__author__ = "Daniele Bigoni"
__copyright__ = """Copyright 2014, The Technical University of Denmark"""
__credits__ = ["Daniele Bigoni"]
__maintainer__ = "Daniele Bigoni"
__email__ = "dabi@imm.dtu.dk"
__status__ = "Production"
