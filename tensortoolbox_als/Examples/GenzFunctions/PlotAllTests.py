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

# from TTcrossGenzPlotData import plot_test
# from TTdmrgGenzPlotData import plot_test
from QTTdmrgGenzPlotData import plot_test

def main(argv):
    Ns_FUNC = range(6)
    Types = ['Projection','LinInterp','PolyInterp']
    Norm = [ True, False ]
    
    # Submit Genz functions normalized
    for GenzNormalized in Norm:
        for Type in Types:
            for FNUM in Ns_FUNC:
                plot_test(FNUM,GenzNormalized,Type)

if __name__ == "__main__":
    main(sys.argv[1:])
