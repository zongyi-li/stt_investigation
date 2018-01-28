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

__all__ = ['load','storable_object','ttcross_store','to_v_0_3_0']

import time
import itertools
import shutil
import os.path
try:
    import cPickle as pkl
except ImportError:
    import pickle as pkl
import h5py

from TensorToolbox import __version__ as TT_version

def to_v_0_3_0(filename):
    """ Used to upgrade the storage version from version <0.3.0 to version 0.3.0
    
    :param string filename: path to the filename. This must be the main filename with no extension.
    """
    pkl_location = filename + ".pkl"
    
    # Open the old version
    print("Opening %s" % pkl_location)
    ff = open(pkl_location, 'rb')
    obj = pkl.load(ff)
    ff.close()

    try:
        obj.VERSION
    except:
        # Copy old version for backup
        print("Backup copy %s" % pkl_location)
        shutil.copyfile(pkl_location, pkl_location + ".deprec")

        # Remove old version file
        print("Removing %s" % pkl_location)
        os.remove(pkl_location)

        # Update version of the object
        print("Updating objects")
        obj.to_v_0_3_0(filename)

        # Force storage
        print("Storing %s" % filename)
        obj.store(force=True)
    else:
        print("The version of the file is already >0.3.0")

def load(filename,load_data=True):
    """ Used to load TensorToolbox data.
    
    :param string filename: path to the filename. This must be the main filename with no extension.
    :param bool load_data: whether to load additional data from ".h5" files.
    """
    pkl_location = filename + ".pkl"
    h5_location = filename + ".h5"
    
    ff = open(pkl_location,'rb')
    obj = pkl.load(ff)
    ff.close()
    if load_data:
        obj.load(h5_location=h5_location)
    return obj

class storable_object( object ):
    """ Constructor of objects that can be stored
    
    :param string store_location: path to the storage file
    :param int store_freq: Number of seconds between each storage
    :param bool store_overwrite: whether to overwrite existing files
    :param object store_object: parent object to be stored in place of the current object

    Attributes:
        **serialize_list**: list of objects that must be serialized.
        **subserialize_list**: list of objects for which the serialization must be done separately.
    """
    
    def __init__(self, store_location='', store_freq=None, store_overwrite=False, store_object=None):

        #######################################
        # List of attributes
        self.VERSION = None
        
        self.store_freq = None
        self.store_location = ''

        self.serialize_list = ['VERSION', 'serialize_list', 'subserialize_list', 'store_location', 'store_freq']
        self.subserialize_list = []

        # Non serialized attributes which must be re-init on setstate
        self.last_store_time = None

        # End list
        #######################################

        self.VERSION = TT_version

        self.store_location = store_location
        self.store_freq = store_freq
        self.last_store_time = -float("inf")
        self.store_object = store_object
        if self.store_object == None  and os.path.isfile(self.store_location) and not store_overwrite:
            raise AttributeError("The file %s already exist." % self.store_location)
    
    def __getstate__(self):
        return dict( [ (tag, getattr( self, tag )) for tag in self.serialize_list ] )

    def __setstate__(self,state, store_object = None):
        for tag in state.keys():
            setattr(self, tag, state[tag])
        # Reset parameters
        self.reset_store_time()
        self.set_store_object( store_object )

    def set_store_location(self,store_location):
        """ Set a new store location for the object
        
        :param string store_location: new store location
        """
        self.store_location = store_location

    def set_store_freq(self,store_freq):
        """ Set a new store frequency for the object
        
        :param int store_freq: new store location
        """
        self.store_freq = store_freq

    def to_be_stored(self,force=False):
        # force = force or (self.last_store_time == None) or (self.store_freq == None)
        # if self.store_location not in ('',None) and \
        #    (force or time.time() > self.last_store_time + self.store_freq): 
        #     return True
        # else: return False

        # Ensure first storage when last_store_time is not set yet
        force = force or (self.store_freq != None and self.last_store_time == None) 
        if self.store_location not in ('',None) and \
           (force or \
                (self.store_freq != None and time.time() > self.last_store_time + self.store_freq)
            ): 
            return True
        else: return False
    
    def reset_store_time(self):
        self.last_store_time = time.time()
    
    def set_store_object(self, store_object):
        self.store_object = store_object
    
    def h5store(self, h5file):
        """ Used to store additional data in hdf5 format. To be redefined in subclasses.
        """
        pass

    def h5load(self, h5file):
        """ Used to load additional data in hdf5 format. To be redefined in subclasses.
        """
        pass
    
    def load(self,h5_location=None):
        """ Used to load additional data.
        """
        if self.store_object == None:
            try:
                self.VERSION
            except:
                # Old pickle serialization. Nothing extra to be loaded.
                pass
            else:
                # New storage of objects (versions > 0.3.0)
                # The input will be:
                #   - an h5 file containing the data
                
                # File name
                if h5_location == None:
                    h5_location = self.store_location + ".h5"
                
                # Call the data loading method in self
                h5file = h5py.File(h5_location, 'r')
                self.h5load(h5file)
                h5file.close()
        else:
            self.store_object.load()
    
    def store(self, force=False):
        """ Used to store any object in the library.

        :param bool force: force storage before time
        """
        if self.store_object == None:
            if self.to_be_stored(force):
                try:
                    self.VERSION
                except:
                    # Old pickle serialization of objects
                    if os.path.isfile(self.store_location):
                        # If the file already exists, make a safety copy
                        shutil.copyfile(self.store_location, self.store_location+".old")
                    ff = open(self.store_location,'wb')
                    pkl.dump(self,ff)
                    ff.close()
                else:
                    # New storage of objects (versions > 0.3.0)
                    # The output will be two files containing:
                    #   - a pickle file conatining the serialization of self
                    #   - an h5 file containing the data

                    # File names
                    pkl_location = self.store_location + ".pkl"
                    h5_location = self.store_location + ".h5"
                    
                    # Store old copy for safety
                    if os.path.isfile(pkl_location):
                        shutil.copyfile(pkl_location, pkl_location + ".old")
                    if os.path.isfile(h5_location):
                        shutil.copyfile(h5_location, h5_location + ".old")
                    
                    # Dump the serialized version of the object
                    ff = open(pkl_location,'wb')
                    pkl.dump(self,ff)
                    ff.close()
                    
                    # Call the data storage method in self
                    h5file = h5py.File(h5_location,'a') # Read/write if exists, create otherwise
                    self.h5store(h5file)
                    h5file.close()
                finally:
                    self.reset_store_time()
        else:
            self.store_object.store(force)
    
    def to_v_0_3_0(self, store_location):
        """ To be implemented for objects that need to be upgraded to v0.3.0.
        
        :param string filename: path to the filename. This must be the main filename with no extension.
        """
        self.VERSION = '0.3.0'
        self.store_location = store_location
        self.serialize_list.append('VERSION')


##############################################
# DEPRECATED
##############################################
def ttcross_store(path,TW,TTapp):
    """ Used to store the computed values of a TTcross approximation. Usually needed when the single function evaluation is demanding or when we need to restart TTcross later on.

    :param string path: path pointing to the location where to store the data
    :param TensorWrapper TW: Tensor wrapper used to build the ttcross approximation. TW.get_data(), TW.get_X() and TW.get_params() will be stored.
    :param TTvec TTapp: TTcross approximation. TTapp.ttcross.Jinit will be stored.

    .. deprecated:: 0.3.0
       Use the objects' methods :func:`store`.
    """
    dic = {'Jinit' : TTapp.ttcross_Jinit,
           'rs': TTapp.ttcross_rs,
           'Js': TTapp.ttcross_Js,
           'Is': TTapp.ttcross_Is,
           'Js_last': TTapp.ttcross_Js_last,
           'ltor_fiber_lists': TTapp.ltor_fiber_lists,
           'rtol_fiber_lists': TTapp.rtol_fiber_lists,
           'X' : TW.get_X(),
           'params' : TW.get_params(),
           'data' : TW.get_data()}
    if os.path.isfile(path):
        # If the file already exists, make a safety copy
        shutil.copyfile(path, path+".old")
    ff = open(path,'wb')
    pkl.dump(dic,ff)
    ff.close()
