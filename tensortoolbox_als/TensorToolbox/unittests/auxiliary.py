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

class bcolors:
    HEADER = '\033[95m'
    OKBLUE = '\033[94m'
    OKGREEN = '\033[1;32m'
    WARNING = '\033[93m'
    FAIL = '\033[1;91m'
    ENDC = '\033[0m'

    def disable(self):
        self.HEADER = ''
        self.OKBLUE = ''
        self.OKGREEN = ''
        self.WARNING = ''
        self.FAIL = ''
        self.ENDC = ''


def print_ok(string):
    print(bcolors.OKGREEN + "[SUCCESS] " + string + bcolors.ENDC)

def print_fail(string,msg=''):
    print(bcolors.FAIL + "[FAILED] " + string + bcolors.ENDC)
    if msg != '':
        print(bcolors.FAIL + msg + bcolors.ENDC)

def print_summary(title,nsucc,nfail):
    if nfail > 0:
        print(bcolors.FAIL + "[SUMMARY " + title + "] Passed: " + str(nsucc) + "/" + str(nsucc+nfail) + bcolors.ENDC)
    else:
        print(bcolors.OKBLUE + "[SUMMARY " + title + "] Passed: " + str(nsucc) + "/" + str(nsucc+nfail) + bcolors.ENDC)
