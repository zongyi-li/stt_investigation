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
import os
import errno

def main():
    QUEUE = 'himem'
    RUN_PATH = './runners'
    
    Ns_FUNC = range(6)
    Types = ['Projection','LinInterp','PolyInterp']
    Norm = [ False, True ]
    
    # Submit Genz functions normalized
    for GenzNormalized in Norm:
        for Type in Types:
            for FNUM in Ns_FUNC:
                f_name = "f" + str(FNUM) + "-T" + Type + "-N" + str(GenzNormalized)
                run_path = RUN_PATH + "/" + f_name
                f = open( run_path, 'w' )
                f.write('#!/bin/bash\n')
                f.write('# -- Daniele Bigoni ---\n')
                f.write('# -- request /bin/bash --\n')
                f.write('#$ -S /bin/bash\n')
                f.write('# -- run in the current working (submission) directory --\n')
                f.write('#$ -cwd\n')
                f.write('#$ -m bea\n')
                f.write('#$ -pe mpi 1\n')
                f.write("source /home/dabi/pythonEnvs/dtu-uq-hms1/bin/activate\n")
                f.write("python QTTdmrgGenz.py " + str(FNUM) + " " + str(GenzNormalized) + " " + Type)
                f.close()
                
                # Submit job to the cluster
                error = os.system('qsub -q %s %s' % (QUEUE,run_path))
                if error:
                    print >> sys.stderr, 'Error in the submission of job %d to the cluster' % i


if __name__ == "__main__":
    main()
