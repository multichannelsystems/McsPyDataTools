"""
    McsCMOSMEA
    ~~~~~~~~~~

    Data classes to wrap and hide raw data handling of the CMOS-MEA HDF5 data files.
    It is based on the MCS-CMOS-MEA Rawdata and ProcessedData definitions for HDF5 
    of the given compatible versions.

    :copyright: (c) 2018 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

import h5py
from builtins import IndexError
import datetime
import math
import uuid
import collections
import numpy as np
import pandas as pd
from numpy import rec
import itertools
from numbers import Number
from inspect import signature
import re
from typing import Dict

from . import ureg, McsHdf5Types, McsHdf5Protocols
from .McsData import RawData
from pint import UndefinedUnitError

MCS_TICK = 1 * ureg.us
CLR_TICK = 100 * ureg.ns

# day -> number of clr ticks (100 ns)
DAY_TO_CLR_TIME_TICK = 24 * 60 * 60 * (10**7)

VERBOSE = False

def dprint(n, *args):
    if VERBOSE:
        print(n, args)

class DictProperty_for_Classes(object):
    """

    """

    class _proxy(object):

        def __init__(self, obj, fget, fset, fdel):
            self._obj = obj
            self._fget = fget
            self._fset = fset
            self._fdel = fdel

        def __getitem__(self, key):
            if self._fset is None:
                raise TypeError("Cannot read item.")
            return self._fget(self._obj, key)

        def __setitem__(self, key, value):
            if self._fset is None:
                raise TypeError("Cannot set item.")
            self._fset(self._obj, key, value)

        def __delitem__(self, key):
            if self._fdel is None:
                raise TypeError("Cannot delete item.")
            self._fdel(self._obj, key)

    def __init__(self, fget=None, fset=None, fdel=None):
        self._fget = fget
        self._fset = fset
        self._fdel = fdel

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self._proxy(obj, self._fget, self._fset, self._fdel)


class _property(object):


    class _proxy(object):

        def __init__(self, obj, fget, fset, fdel):
            self._obj  = obj 
            self._fget = fget
            self._fset = fset
            self._fdel = fdel

        def __getitem__(self,key):
            if self._fget is None:
                raise TypeError("Cannot read item.")#
            return self._fget(self._obj, key)

        def __setitem__(self,key,value):
            if self._fset is None:
                raise TypeError("Cannot set item.")
            self._fset(self._obj, key, value)

        def __delitem__(self, key):
            if self._fdel is None:
                raise TypeError("Cannot delete item.")
            self._fdel(self._obj, key)

    def __init__(self, fget=None, fset=None, fdel=None):
        self._fget = fget
        self._fset = fset
        self._fdel = fdel

    def __get__(self, obj, objtype=None):
        if obj is None:
            return self
        return self._proxy(obj, self._fget, self._fset, self._fdel)


class _list_property(object):
    """
    Creates helper class which is a list subclass. It is used to hand lists of streams to the McsPy user.

    :param list: list of streams
    """
    
    class McsProxy(collections.UserList):

        def __init__(self, initlist=None, obj=None, fget=None, fset=None, fdel=None):
            """
            ATTENTION! The collections.UserList documentation requires the init method of collections.UserList subclasses to accept zero or one argument!
            """
            super().__init__(initlist)
            self._obj = obj
            self._fget = fget
            self._fset = fset
            self._fdel = fdel

        def __getitem__(self,key):
            if self._fget is None:
                raise TypeError("Cannot read item.")
            if isinstance(key, int):
                return self._fget([self.data[key][1]])
            return self._fget([id_set.mcs_instanceid for id_set in selection])

        def __setitem__(self,key,value):
            if self._fset is None:
                raise TypeError("Cannot set item.")
            self._fset(self._obj, key, value)

        def __delitem__(self, key):
            if self._fdel is None:
                raise TypeError("Cannot delete item.")
            self._fdel(self._obj, key)

        def __str__(self):
            stream_types = dict()
            column_width = 35
            line = '-'*(column_width*3+4)+'\n'
            bold_line = '='*(column_width*3+4)+'\n'
            out = '|'+'{:^{}}'.format('Subtype', column_width)+'|'+'{:^{}}'.format('McsPy name', column_width)+'|'+'{:^{}}'.format('HDF5 name', column_width)+'|\n'
            out += bold_line
            for id_set in self.data:
                type =self._obj[id_set.h5py].attrs['ID.Type'].decode('UTF-8')
                subtype = self._obj[id_set.h5py].attrs['SubType'].decode('UTF-8')
                if not type in stream_types:
                    stream_types[type] = list()
                stream_types[type].append('|'+'{:^{}}'.format(subtype, column_width)+'|'+'{:^{}}'.format(id_set.mcspy, column_width)+'|'+'{:^{}}'.format(id_set.h5py, column_width)+'|\n')
            for type in stream_types:
                out += type +':\n'
                out += ''.join(stream_types[type])
                out += line
            return out

    def __init__(self, content, owner_instance, fget=None, fset=None, fdel=None):
        self._content = content
        self._owner_instance = owner_instance
        self._fget = fget
        self._fset = fset
        self._fdel = fdel

    def __get__(self, obj, objtype=None):
        #if obj is None:
        #    return self
        return self.McsProxy(self._content, obj=obj, fget=self._fget, fset=self._fset, fdel=self._fdel)

    def __str__(self):
        return self.McsProxy(self._content, obj=self._owner_instance).__str__()


class McsHDF5(object):
    """
    Container class that provides common structures for an Mcs HDF5 file
    """
    def __init__(self, hdf5_object):
        """
        Initializes the HDF5 container class from an HDF5 object
        """
        self._hdf5_attributes   = None
        self._h5py_object       = hdf5_object
        if hasattr(self._h5py_object,'attrs'):
            self._mcs_type          = hdf5_object.attrs['ID.Type'].decode('UTF-8')
            self._mcs_typeid        = hdf5_object.attrs['ID.TypeID'].decode('UTF-8')
            self._mcs_instance      = hdf5_object.attrs['ID.Instance'].decode('UTF-8')
            self._mcs_instanceid    = hdf5_object.attrs['ID.InstanceID'].decode('UTF-8')

    def _get_attributes(self):
        "Read and convert all attributes of the HDF5 group for easy access"
        if hasattr(self._h5py_object,'attrs'):
            hdf5_attributes = self._h5py_object.attrs.items()
            self._hdf5_attributes = {}
            for (name, value) in hdf5_attributes:
                if hasattr(value, "decode"):
                    try:
                        self._hdf5_attributes[name] = value.decode('utf-8').rstrip()
                    except:
                        self._hdf5_attributes[name] = value
                else:
                    self._hdf5_attributes[name] = value[0]
        else:
            raise AttributeError('No Attributes')

    def _get_mcspy_instance(self, h5py_object, mcspy_parent=None):
        """
        takes a h5py object and returns an appropriate mcspy object 

        :param hdf5_object. 
        """
        typeID = h5py_object.attrs['ID.TypeID'].decode('utf-8').rstrip()
        cls = McsHdf5Types.get_mcs_class_name(typeID)
        if cls is None:
            return h5py_object
        elif isinstance(h5py_object, h5py.Dataset):
            if isinstance(mcspy_parent, McsGroup) and 'mcspy_parent' in signature(cls.__init__).parameters and h5py_object.name.split('/')[-1] in mcspy_parent:
                return cls(h5py_object, mcspy_parent=mcspy_parent)
        return cls(h5py_object)

    @staticmethod
    def get_attributes(hdf5_object):
        "Read and convert all attributes of the HDF5 group for easy access"
        if hasattr(hdf5_object,'attrs'):
            hdf5_attributes         = hdf5_object.attrs.items()
            hdf5_attributes_decoded = {}
            for (name, value) in hdf5_attributes:
                if hasattr(value, "decode"):
                    hdf5_attributes_decoded[name] = value.decode('utf-8').rstrip()
                else:
                    hdf5_attributes_decoded[name] = value
            return hdf5_attributes_decoded
        else:
            raise AttributeError('No Attributes')

    def hdf5_to_mcspy(self, hdf5_names):
        """
        receives a hdf5_name as string in Mcs CMOS MEA file system style
        and converts it to python toolbox equivalent.
        """
        #weird_mcf_hdf5_file_name = ["channeldata", "sensordata", "high-pass"]
        if isinstance(hdf5_names,str):
            return hdf5_names.strip().replace(":","").replace("(","").replace(")","").replace(" ","_").replace('@','at').replace('.','_').replace(',','_')
        else:
            raise TypeError("Pass a 'str' object")

    @property
    def attributes(self):
        if self._hdf5_attributes == None:
            try:
                self._get_attributes()
            except AttributeError as err:
                print(err)
        return self._hdf5_attributes

    @property
    def h5py_object(self):
        return self._h5py_object

class McsGroup(h5py.Group, McsHDF5):
    """
    this class subclasses the h5py.Group object and extends it with McsPy toolbox functionality
    """

    IDSetGroup      = collections.namedtuple('IDSetGroup', ['h5py', 'mcs_instanceid', 'mcspy', 'mcs_typeid'])
    IDSetDataset    = collections.namedtuple('IDSetDataset', ['h5py', 'mcs_instanceid', 'mcspy', 'mcs_typeid'])

    def __init__(self, h5py_group_object):
        if isinstance(h5py_group_object, h5py.Group):
            h5py.Group.__init__(self, h5py_group_object.id)
            McsHDF5.__init__(self, h5py_group_object)

            self._child_storage     = dict()
            self._child_inventory   = list()
            for child in h5py_group_object:
                try: 
                    mcs_instanceid      = h5py_group_object[child].attrs['ID.InstanceID'].decode('UTF-8')
                    mcs_typeid          = h5py_group_object[child].attrs['ID.TypeID'].decode('UTF-8').rstrip()
                    mcspy_child_name    = self.hdf5_to_mcspy(child)
                    if isinstance(self._h5py_object[child], h5py.Dataset):
                        self._child_inventory.append(McsGroup.IDSetDataset(h5py=child,
                                                                        mcs_instanceid=mcs_instanceid,
                                                                        mcspy=mcspy_child_name,
                                                                        mcs_typeid=mcs_typeid)) # (h5py key/name, mcs instance id, mcs py key/name, mcs_typeid)
                    if isinstance(self._h5py_object[child], h5py.Group):
                        self._child_inventory.append(McsGroup.IDSetGroup(h5py=child,
                                                                        mcs_instanceid=mcs_instanceid,
                                                                        mcspy=mcspy_child_name,
                                                                        mcs_typeid=mcs_typeid)) # (h5py key/name, mcs instance id, mcs py key/name, mcs_typeid)
                except Exception as e:
                    print("Error opening group " + child + ": " + str(e))
        else:
            raise TypeError('The h5py_group_object \'{}\' is not an instance of the h5py.Group class.'.format(h5py_group_object.name))

    def __repr__(self):
        return '<McsGroup object at '+str(hex(id(self)))+'>'

    def __str__(self):
        column_width = 25
        bold_line = '='*(column_width*3+4)+'\n'
        line = '-'*(column_width*3+4)+'\n'
        out = line + 'Parent Group: <'+str(type(self)).strip('<>')+' object at '+str(hex(id(self)))+'>\n'
        header = '|'+'{:^{}}'.format('Mcs Type', column_width)+'|'+'{:^{}}'.format('HDF5 name', column_width)+'|'+'{:^{}}'.format('McsPy name', column_width)+'|\n'
        dataset = 'Datasets:\n'
        group = 'Groups:\n'
        for child in self._child_inventory:
            #h5py_key, mcs_typeid, mcspy_key, mcs_typeid   = child
            mcs_type = self._h5py_object[child.h5py].attrs['ID.Type'].decode('utf-8')
            if isinstance(child, McsGroup.IDSetGroup):
                group += '|'+'{:^{}}'.format(mcs_type, column_width)+'|'+'{:^{}}'.format(child.h5py, column_width)+'|'+'{:^{}}'.format(child.mcspy, column_width)+'|\n'
            if isinstance(child, McsGroup.IDSetDataset):
                dataset += '|'+'{:^{}}'.format(mcs_type, column_width)+'|'+'{:^{}}'.format(child.h5py, column_width)+'|'+'{:^{}}'.format(child.mcspy, column_width)+'|\n'
        if group.count('\n') == 1:
            group += ' '*4+'None\n'
        if dataset.count('\n') == 1:
            dataset += ' '*4+'None\n'
        out += line + '\n\n' + header + bold_line + group + line + dataset
        return out

    def __getattr__(self, name):
        id_set = self.ischild(name)
        if not id_set:
            raise AttributeError('There is no instance with name {} within this group'.format(name))
        return self._children[id_set.mcs_instanceid]

    def __dir__(self):
        return super().__dir__() + [s.mcspy for s in self._child_inventory]

    def ischild(self, id):
        """
        Takes an identifier and checks if it is a valid identifier for a child of this group:

        :param id: mcs instanceid, h5py name , mcspy name as instance of 'str'

        :return: False if id is not valid, set of identifiers of the child
        """
        if not isinstance(id, str):
            return False
        return next((set for set in self._child_inventory if id in set[0:3]), False)

    def _get_child(self, key):
        """
        Retrieves a child from the dictionary self._child_storage:

        :param key: mcs_instanceid which indentifies a subgroup of self._h5py_object
        """
        child_id_set = self.ischild(key)
        if not child_id_set:
            raise KeyError('key \'{}\' is not valid. Pass an instance of \'str\', which identifies a child of this group.')
        if not child_id_set.mcs_instanceid in self._child_storage.keys():
            self._read_child(child_id_set)
        return self._child_storage[child_id_set.mcs_instanceid]

    def _get_children(self, key):
        """
        Retrieves a set of children from the dictionary self._child_storage:

        :param key: list or tuple with mcs_instanceid which indentify a subgroup of self._h5py_object respectively
        """
        if isinstance(key, (list, tuple)):
            if len(key) == 1:
                return self._get_child(key[0])
            out = list()
            for id in key:
                try:
                    out.append(self._get_child(id))
                except KeyError as err:
                    print(err)
            return out

    _children = _property(_get_child, None, None)

    def _set_child(self, key, value):
        pass

    def _del_child(self, key):
        pass

    def _read_children_of_type(self, child_typeid, store_parents=True):
        """
        reads all children with given typeID

        :param child_typeid: mcs type id for a specific mcs hdf5 structure
        """
        for id_set in self._child_inventory:
            if child_typeid == id_set[3] and id_set[1] not in self._child_storage.keys():
                self._readf_child(id_set, store_parents)

    def _read_child(self, id_set, store_parent=True):
        """
        read given child
        
        :param id_set: id_set must be a valid id_set identifiying a child of this group
        """
        if store_parent:
            self._child_storage[id_set.mcs_instanceid] = self._get_mcspy_instance(self._h5py_object[id_set.h5py], self)
        else:
            self._child_storage[id_set.mcs_instanceid] = self._get_mcspy_instance(self._h5py_object[id_set.h5py])

    def tree(self, name='mcspy', mcs_type=False, max_level=None):
        """
        builds the hdf5 hierarchy beginning with the current group then traversing all subentities depth first as a string
        
        :param name: cfg variable for the type of name that is to be printed for each entity in the 
                        h5py group, default: 'h5py', options: 'mcspy'
        :param mcs_type: cfg variable to show mcs type in the tree, default: False
        :param max_level: cfg variable to limit the number of tree levels shown, default: None (show all)
        """
        if not hasattr(self, '_tree_string'):
            self._tree_string = ''
        if not hasattr(self, '_tree_mcs_type'):
            self._tree_mcs_type = ''
        if not hasattr(self, '_tree_names'):
            self._tree_names = ''
        if not hasattr(self, '_tree_level'):
            self._tree_level = None
        if self._tree_string == '' or mcs_type != self._tree_mcs_type or self._tree_names != name or self._tree_level != max_level:
            self._tree_string = ''
            self._tree_mcs_type = mcs_type
            self._tree_names = name
            self._tree_level = max_level
            if self.name == '/':
                print(self.name)
            else:
                print(self.name.split('/')[-1])
            name_width = 35
            base_level = self.name.count('/')
            if self._tree_names == 'mcspy':
                def _print_mcspy_tree(name):
                    level = name.count('/')+1
                    if max_level is None or level - base_level < max_level:
                        mcstype = ''
                        if 'ID.Type' in self[name].attrs and mcs_type:
                            mcstype += ' - '+self[name].attrs['ID.Type'].decode('UTF-8')
                        name = self.hdf5_to_mcspy(name.split('/')[-1])
                        self._tree_string +=' '*4*level+name.ljust(name_width)+mcstype+'\n'
                self.visit(_print_mcspy_tree)
            elif self._tree_names == 'h5py':
                def _print_h5py_tree(name):
                    level = name.count('/')+1
                    if max_level is None or level - base_level < max_level:
                        mcstype = ''
                        if 'ID.Type' in self[name].attrs and mcs_type:
                            mcstype += ' - '+self[name].attrs['ID.Type'].decode('UTF-8')
                        name = name.split('/')[-1]
                        self._tree_string +=' '*4*level+name.ljust(name_width)+mcstype+'\n'
                self.visit(_print_h5py_tree)
            else:
                raise ValueError('name \'{}\' is not a valid argument. Pass \'h5py\' or \'mcspy\''.format(name))
        return self._tree_string

class McsDataset(h5py.Dataset, McsHDF5):
    """
    This class subclasses the h5py.Dataset object and extends it with McsPy toolbox functionality
    """
    def __init__(self, h5py_dataset_object):
        h5py.Dataset.__init__(self, h5py_dataset_object.id)
        McsHDF5.__init__(self, h5py_dataset_object)
        self._compound_dataset_names = None #compound dataset names in mcs python syntax
        if self.dtype.names:
            self._compound_dataset_names = [ self.hdf5_to_mcspy(name) for name in self.dtype.names ]

    def __getattr__(self, name):
        if self._compound_dataset_names:
            if name in self._compound_dataset_names:
                name = self.dtype.names[self._compound_dataset_names.index(name)]
            if name in list(self.dtype.names):
                if hasattr(self._h5py_object[name], "decode"):
                    return self._h5py_object[name].decode('utf-8').rstrip()
                else:
                    return self[name]
            else:
                raise AttributeError('\'{}\' is not a valid attribute for: {}!'.format(name,self.__repr__()))
        else:
            raise AttributeError('\'{}\' is not a valid attribute for: {}!'.format(name,self.__repr__()))

    def iscompound(self):
        """
        Determines whether Dataset is a Compound Dataset

        :return Boolean: True if Dataset object represents h5py Compound Dataset, False otherwise
        """
        if self._compound_dataset_names:
            return True
        return False

    def __repr__(self):
        if self.iscompound():
            return '<McsDataset object representing a compound dataset at '+str(hex(id(self)))+'>'
        return '<McsDataset object at '+str(hex(id(self)))+'>'

    def __str__(self):
        first_col_width = 25
        if self.iscompound():
            out = 'Compound McsDataset '+self.name.split("/")[-1]+'\n\n'
        else:
            out = 'McsDataset '+self.name.split("/")[-1].ljust(first_col_width)+'\n\n'
        out += 'location in hdf5 file:'.ljust(first_col_width)+self.name+'\n'
        out += 'shape:'.ljust(first_col_width)+self.name+'{}'.format(self.shape)+'\n'
        out += 'dtype:'.ljust(first_col_width)+self.name+'{}'.format(self.dtype)+'\n'
        return out

    def to_pdDataFrame(self):
        """
        Returns the data set as a pandas DataFrame
        """
        return pd.DataFrame(self[()])

class McsStreamList(collections.UserList):
    """
    Creates helper class which is a list subclass. It is used to hand lists of streams to the McsPy user.

    :param list: list of streams
    """

    def __str__(self):
        stream_types = dict()
        column_width = 35
        line = '-'*(column_width*3+4)+'\n'
        bold_line = '='*(column_width*3+4)+'\n'
        out = '|'+'{:^{}}'.format('HDF5 name', column_width)+'|'+'{:^{}}'.format('McsPy name', column_width)+'|'+'{:^{}}'.format('Stream Subtype', column_width)+'|\n'
        out += bold_line
        for stream in self:
            if not stream.attributes['ID.Type'] in stream_types:
                stream_types[stream.attributes['ID.Type']] = list()
            if 'SubType' in stream.attributes:
                stream_types[stream.attributes['ID.Type']].append((stream.name.rsplit('/',1)[1], stream.hdf5_to_mcspy(stream.name.rsplit('/',1)[1]), stream.attributes['SubType'])) #hdf5_name, mcspy_name, subtype
            else:
                stream_types[stream.attributes['ID.Type']].append((stream.name.rsplit('/',1)[1], stream.hdf5_to_mcspy(stream.name.rsplit('/',1)[1]), '')) #hdf5_name, mcspy_name, subtype
        for stream_type in stream_types:
            out += stream_type +':\n'
            for stream in stream_types[stream_type]:
                out += '|'+'{:^{}}'.format(stream[0], column_width)+'|'+'{:^{}}'.format(stream[1], column_width)+'|'+'{:^{}}'.format(stream[2], column_width)+'|\n'
            out += line
        return out

class McsData(object):
    """
    Dummy class provides access to all types of mcs files by returning an instance the class that corresponds to the file type
    """
    def __new__(cls, file_path):
        """
        Creates a Data object this includes checking the validity of the passed HDF5 file and the return of a 
        an object that matches the MCS file type.

        :param file_path: path to a HDF5 file that contains data encoded in a supported MCS-HDF5 format version
        """
        h5_file = h5py.File(file_path, 'r')
        try:
            mcs_hdf5_protocol_type, _ = McsData.validate_mcs_hdf5_version(h5_file)
        except IOError as err:
            print(err)
        h5_file.close()
        if mcs_hdf5_protocol_type == 'CMOS_MEA':
            return McsCMOSMEAData(file_path)
        elif mcs_hdf5_protocol_type == 'RawData':
            return RawData(file_path)

    @staticmethod
    def validate_mcs_hdf5_version(mcs_h5_file_obj):
        "Check if the MCS-HDF5 protocol type and version of the file is supported by this class"
        root_grp = mcs_h5_file_obj['/']
        if 'McsHdf5ProtocolType' in root_grp.attrs: #check for old file type
            mcs_hdf5_protocol_type = root_grp.attrs['McsHdf5ProtocolType'].decode('UTF-8')
            if mcs_hdf5_protocol_type == "RawData":
                mcs_hdf5_protocol_type_version = root_grp.attrs['McsHdf5ProtocolVersion']
                supported_versions = McsHdf5Protocols.SUPPORTED_PROTOCOLS[mcs_hdf5_protocol_type]
                if ((mcs_hdf5_protocol_type_version < supported_versions[0]) or
                    (supported_versions[1] < mcs_hdf5_protocol_type_version)):
                    raise IOError('Given HDF5 file has MCS-HDF5 RawData protocol version %s and supported are all versions from %s to %s' %
                                  (mcs_hdf5_protocol_type_version, supported_versions[0], supported_versions[1]))
            else:
                raise IOError("The root group of this HDF5 file has no 'McsHdf5ProtocolVersion' attribute -> so it could't be checked if the version is supported!")
        elif 'ID.Type' in root_grp.attrs: #check for CMOS MEA file type
            mcs_hdf5_protocol_type = "CMOS_MEA"
            if 'FileVersion' in root_grp.attrs:
                mcs_hdf5_protocol_type_version = root_grp.attrs['FileVersion']
                supported_versions = McsHdf5Protocols.SUPPORTED_PROTOCOLS[mcs_hdf5_protocol_type]
                if ((mcs_hdf5_protocol_type_version[0] < supported_versions[0]) or
                    (supported_versions[1] < mcs_hdf5_protocol_type_version[0])):
                    raise IOError('Given HDF5 file has MCS-HDF5 CMOS-MEA version %s and supported are all versions from %s to %s' %
                                  (mcs_hdf5_protocol_type_version, supported_versions[0], supported_versions[1]))
            else:
                raise IOError("The root group of this HDF5 file has no 'FileID' attribute -> so it could't be checked if the version is supported!")
        else:
            raise IOError("The root group of this HDF5 file has no attribute that can be associated to a MCS HDF5 file type -> this file is not supported by McsPy!")
        return list((mcs_hdf5_protocol_type, mcs_hdf5_protocol_type_version))

class McsCMOSMEAData(McsGroup):
    """
    This class holds the information of a complete MCS CMOS-MEA data file system
    """

    sensorWidth: int = 65
    sensorHeight: int = 65

    def __init__(self, cmos_data_path):
        """
        Creates and initializes a McsCMOSMEAData object that provides access to the content of the given MCS-HDF5 file

        :param cmos_data_path: path to a HDF5 file that contains raw data encoded in a supported MCS-HDF5 format version
        """
        self.h5_file = h5py.File(cmos_data_path, 'r')
        super().__init__(self.h5_file)        
        self.mcs_hdf5_protocol_type, self.mcs_hdf5_protocol_type_version  = McsData.validate_mcs_hdf5_version(self.h5_file)
        #self._get_session_info()
        #self._acquisition       = None
        #self._filter_tool       = None
        #self._sta_explorer      = None
        #self._spike_explorer    = None
        #self._spike_sorter      = None

    def __del__(self):
        self.h5_file.close()

    def __repr__(self):
        return '<McsCMOSMEAData filename=' + self.attributes['ID.Instance'] + '>'

    def __str__(self):
        out: str = '<McsCMOSMEAData instance at '+str(hex(id(self)))+'>\n\n'
        out += 'This object represents the Mcs CMOS MEA file:\n'
        #out += ''*4+'Path:'.ljust(12)+'\\'.join(self.attributes['ID.Instance'].split('\\')[:-1])+'\n'
        out += ''*4+'Filename:'.ljust(12)+self.attributes['ID.Instance'].split('\\')[-1]+'\n\n'
        out += 'Date'.ljust(21)+'Program'.ljust(28)+'Version'.ljust(12)+'\n'
        out += '-'*19+' '*2+'-'*26+' '*2+'-'*10+'\n'
        out += self.attributes['DateTime'].ljust(21) + self.attributes['ProgramName'].ljust(28)+self.attributes['ProgramVersion'].ljust(12)+'\n\n'
        mcs_group_string = super().__str__().split('\n')
        return out+'\nContent:\n'+'\n'.join(mcs_group_string[4:])

    #def _get_session_info(self):
    #    "Read all session metadata/root group atributes of the Cmos mea file"
    #    root_grp_attributes = self.h5_file['/'].attrs.items()
    #    self.session_info   = {}
    #    for (name, value) in root_grp_attributes:
    #        #print(name, value)
    #        if hasattr(value, "decode"):
    #            self.session_info[name] = value.decode('utf-8').rstrip()
    #        else:
    #            self.session_info[name] = value

    def __read_acquisition(self):
        "Read aquisition group"
        if 'Acquisition' in list(self.h5_file.keys()):
            acquisition_folder = self.h5_file['Acquisition']
            #acquisition_attributes  = self.h5_file['Acquisition'].attrs.items()
            if len(acquisition_folder)>0:
                self._acquisition = Acquisition(acquisition_folder)
                for (name, value) in acquisition_folder.items():
                    dprint(name, value)
        else:
            raise AttributeError("The HDF5 file does not contain a group 'Acquisition'.")

    def __read_sta_explorer(self):
        if 'STA Explorer' in list(self.h5_file.keys()):
            "Read sta explorer group"
            network_explorer_folder = self.h5_file['STA Explorer']
            #sta_explorer_attributes = self.h5_file['STA Explorer'].attrs.items()
            if len(network_explorer_folder)>0:
                self._sta_explorer = NetworkExplorer(network_explorer_folder)
                for (name, value) in network_explorer_folder.items():
                    dprint(name, value)
        elif 'Network Explorer' in list(self.h5_file.keys()):
            "Read network explorer group"
            network_explorer_folder = self.h5_file['Network Explorer']
            #sta_explorer_attributes = self.h5_file['STA Explorer'].attrs.items()
            if len(network_explorer_folder)>0:
                self._sta_explorer = NetworkExplorer(network_explorer_folder)
                for (name, value) in network_explorer_folder.items():
                    dprint(name, value)
        else:
            raise AttributeError("The HDF5 file does not contain a group 'STA Explorer' or 'Network Explorer'.")

    def __read_filter_tool(self):
        if 'Filter Tool' in list(self.h5_file.keys()):
            pass
        else:
            raise AttributeError("The HDF5 file does not contain a group 'Filter Tool'.")

    def __read_spike_explorer(self):
        if 'Spike Explorer' in list(self.h5_file.keys()):
            pass
        else:
            raise AttributeError("The HDF5 file does not contain a group 'Spike Explorer'.")

    def __read_spike_sorter(self):
        if 'Spike Sorter' in list(self.h5_file.keys()):
            pass
        else:
            raise AttributeError("The HDF5 file does not contain a group 'Spike Sorter'.")

    @classmethod
    def sensorID_to_coordinates(self, sensorID):
        "Computes the [x,y] chip coordinates of a sensor. Note: both, sensor IDs and coordinates are base 1"
        if 0<sensorID and sensorID<=self.sensorWidth*self.sensorHeight:
            sensorID -= 1
            return np.array([(sensorID % self.sensorHeight)+1,(sensorID // self.sensorHeight)+1])
        else:
            raise KeyError('Sensor ID out of range!')

    @classmethod
    def coordinates_to_sensorID(self, row: int, col: int) -> int:
        "Computes the sensor ID for row and column coordinates. Note: sensor IDs and rows and columns are base 1"
        if 0<row and row<=self.sensorHeight and 0<col and col<=self.sensorWidth:
            return self.sensorHeight*(col-1)+row
        else:
            raise KeyError('Coordinates out of range!')

class Acquisition(McsGroup):
    """
    Container class for acquisition data.
    
    Acquisition Group can hold different types of streams: Analog Streams, Event Streams, Timestamp Streams, Segment Streams, Spike Streams
    """

    "holds allowed stream types in TypeID:Type pairs"
    _stream_types = {"AnalogStream"    : "9217aeb4-59a0-4d7f-bdcd-0371c9fd66eb",
                     "FrameStream"     : "15e5a1fe-df2f-421b-8b60-23eeb2213c45",
                     "SegmentStream"   : "35f15fa5-8427-4d07-8460-b77a7e9b7f8d",
                     "TimeStampStream" : "425ce2e0-f1d6-4604-8ab4-6a2facbb2c3e",
                     "SpikeStream"     : "26efe891-c075-409b-94f8-eb3a7dd68c94",
                     "EventStream"     : "09f288a5-6286-4bed-a05c-02859baea8e3"}

    def __init__(self, acquisition_group):
        super().__init__(acquisition_group)
        setattr(Acquisition, 'ChannelStreams', _list_property([id_set for id_set in self._child_inventory if id_set.mcs_typeid == Acquisition._stream_types["AnalogStream"]], self, fget=self._get_children, fset=None, fdel=None))
        setattr(Acquisition, 'SensorStreams', _list_property([id_set for id_set in self._child_inventory if id_set.mcs_typeid == Acquisition._stream_types["FrameStream"]], self, fget=self._get_children, fset=None, fdel=None))
        setattr(Acquisition, 'SegmentStreams', _list_property([id_set for id_set in self._child_inventory if id_set.mcs_typeid == Acquisition._stream_types["SegmentStream"]], self, fget=self._get_children, fset=None, fdel=None))
        setattr(Acquisition, 'SpikeStreams', _list_property([id_set for id_set in self._child_inventory if id_set.mcs_typeid == Acquisition._stream_types["SpikeStream"]], self, fget=self._get_children, fset=None, fdel=None))
        setattr(Acquisition, 'EventStreams', _list_property([id_set for id_set in self._child_inventory if id_set.mcs_typeid == Acquisition._stream_types["EventStream"]], self, fget=self._get_children, fset=None, fdel=None))
        
    def __str__(self) -> str:
        if self._child_inventory:
            column_width: int = 25
            bold_line: str = '='*(column_width*3+4)+'\n'
            line: str = '-'*(column_width*3+4)+'\n'
            out: str = line + 'Parent Group: <'+str(type(self)).strip('<>')+' object at '+str(hex(id(self)))+'>\n\n'
            header: str = '|'+'{:^{}}'.format('Subtype', column_width)+'|'+'{:^{}}'.format('HDF5 name', column_width)+'|'+'{:^{}}'.format('McsPy name', column_width)+'|\n'
            stream_types: Dict[str, str] = dict()
            for child in self._child_inventory:
                #h5py_key, mcs_typeid, mcspy_key, mcs_typeid   = child
                stream_type = self._h5py_object[child.h5py].attrs['ID.Type'].decode('utf-8')
                stream_subtype = self._h5py_object[child.h5py].attrs['SubType'].decode('utf-8')
                if not stream_type in stream_types:
                    stream_types[stream_type] = ""
                stream_types[stream_type] += '|'+'{:^{}}'.format(stream_subtype, column_width)+'|'+'{:^{}}'.format(child.h5py, column_width)+'|'+'{:^{}}'.format(child.mcspy, column_width)+'|\n'
            out += line + '\n\n' + header + bold_line
            for stream_type in stream_types:
                out += stream_type+'\n'+stream_types[stream_type] + line
        else:
            out = "No streams found"
        return out

    def __repr__(self):
        return '<Acquisition object at '+str(hex(id(self)))+', ChannelStreams='+str(len(self.ChannelStreams))+', SensorStreams='+str(len(self.SensorStreams))+', SegmentStreams='+str(len(self.SegmentStreams))+', SpikeStreams='+str(len(self.SpikeStreams))+', EventStreams='+str(len(self.EventStreams))+'>'

class McsInfo(McsDataset):
        """
        Container class for Stream Meta Data
        """
        def __init__(self, meta_data_set):
            """
            Initializes a Meta object from a provided HDF5 dataset
            """
            super().__init__(meta_data_set)

class McsStream(McsGroup):
    """
    Base class for all stream types
    """
    def __init__(self, stream_grp, data_typeid, meta_typeid, *args):
        """
        Initializes a stream object with its associated h5py group object

        :param stream_grp: group object correponding to a folder in the HDF5 file. It contains the data of this stream
        :param data_typeid: mcs type id of the data stored in the stream
        :param meta_typeid: mcs type id of the meta data stored in the stream
        """
        super().__init__(stream_grp)
        self._data_typeid = data_typeid
        self._meta_typeid = meta_typeid
        self._entities = None

    def _get_data_headers(self):
        """
        retrieves all headers present in a dataset

        return headers: all headers native to the data datasets in a certain stream instance
        """
        headers = list()
        try:
            data_name = next(child for child in self._h5py_object if self._h5py_object[child].attrs['ID.TypeID'].decode('UTF-8') == self._data_typeid)
        except StopIteration:
            return list()
        if hasattr(self._h5py_object[data_name].dtype, 'names'):
            headers = list(self._h5py_object[data_name].dtype.names)
        return headers

    def _get_meta_headers(self):
        """
        retrieves all headers of the meta data

        return headers: all headers native to the meta datasets in a certain stream instance
        """
        headers = list()
        try:
            meta_name = next(child for child in self._h5py_object if self._h5py_object[child].attrs['ID.TypeID'].decode('UTF-8') == self._meta_typeid)
        except StopIteration:
            pass
        if hasattr(self._h5py_object[meta_name].dtype, 'names'):
            headers = self._h5py_object[meta_name].dtype.names
        return headers

    @property
    def Data(self):
        "Access all datasets - collection of McsDataset objects"
        return McsStreamList([self._children[id_set.mcs_instanceid] for id_set in self._child_inventory if id_set.mcs_typeid == self._data_typeid])

    @property
    def Meta(self):
        "Access meta data"
        return McsStreamList([self._children[id_set.mcs_instanceid] for id_set in self._child_inventory if id_set.mcs_typeid == self._meta_typeid])
    
    Stream_Types = ["Analog Stream", "Event Stream", "Segment Stream", "TimeStamp Stream", "Frame Stream", "Spike Stream"]

class McsStreamEntity(object):
    """
    Base Class for a McsStreamEntity object
    """

    def __init__(self, parent, id):
        self.mcspy_parent = parent
        self._entity_id = id

class McsChannelStream(McsStream):
    """
    Container class for one analog stream of several channels.
    """

    channel_data_typeid = "5efe7932-dcfe-49ff-ba53-25accff5d622"
    channel_meta_typeid = "9e8ac9cd-5571-4ee5-bbfa-8e9d9c436daa"

    def __init__(self, channel_stream_grp):
        """
        Initializes a channel stream object containing several sweeps of channels over time

        :param channel_stream_grp: folder of the HDF5 file that contains the data of this analog stream
        """
        super().__init__(channel_stream_grp, McsChannelStream.channel_data_typeid, McsChannelStream.channel_meta_typeid)

    def __repr__(self):
        return '<McsChannelStream object at '+str(hex(id(self)))+'>'

    def _get_channel_sweeps_by_number(self, key):
        """
        retrieves all dataset that belong to sweep number 'key'

        :param key: key as int that identifies a sweep in the channel stream

        :return: list of id set that correlates with sweeps with number 'key' in a channel stream
        """
        if isinstance(key, int):
            out = list()
            for child in self._h5py_object.keys():
                sweep_number = [int(s) for s in child if s.isdigit()]
                try:
                    if sweep_number[0] == key and self._data_typeid == self.h5py_object[child].attrs["ID.TypeID"].decode('UTF-8'):
                        out.append(next(id_set for id_set in self._child_inventory if child in id_set))
                except IndexError:
                    pass
            return out
        raise KeyError('{} must be an instance of int!'.format(key))

    @property
    def DataChunk(self):
        """
        The continuous data segments in the stream
        """
        sweep_numbers = np.unique(self.ChannelMeta.GroupID).tolist()
        out = {}
        for sweep_number in sweep_numbers:
            out[sweep_number] = _list_property.McsProxy(self._get_channel_sweeps_by_number(sweep_number), obj=self, fget=self._get_children, fset=None, fdel=None)
        return out

class McsChannelEntity(McsDataset, McsStreamEntity):
    """
    Container class for one ChannelStream Entity.
    """
    def __init__(self, channel_stream_entity_dataset, mcspy_parent):
        """
        initializes a new McsChannelEntity from a h5py_dataset of a hdf5 ChannelData entity

        :param channel_stream_entity_dataset: h5py_dataset of a channel
        """
        id = int(channel_stream_entity_dataset.name.split()[-1]) #_entity_id is Group ID
        McsDataset.__init__(self, channel_stream_entity_dataset)
        McsStreamEntity.__init__(self, mcspy_parent, id)
        self.dimensions = '[ \'number of channels\' x \'samples\' ]'

    def __repr__(self):
        return '<McsChannelEntity object at '+str(hex(id(self)))+', channels='+str(self.shape[0])+', samples='+str(self.shape[1])+'>'

    @property
    def Meta(self):
        """
        reads the subset of Meta data that belongs to the channels
        """
        index = tuple(np.where(self.mcspy_parent.Meta[0].GroupID == self._entity_id)[0])
        return self.mcspy_parent.Meta[0][index,]

class McsEventStream(McsStream):
    """
    Container class for one Event Stream.
    """

    event_data_typeid =  "abca7b0c-b6ce-49fa-ad74-a20c352fe4a7"
    event_meta_typeid =  "8f58017a-1279-4d0f-80b0-78f2d80402b4"

    def __init__(self, event_stream_grp):
        """
        Initializes an event stream object

        :param event_stream_grp: folder of the HDF5 file that contains the data of this event stream
        """
        super().__init__(event_stream_grp, McsEventStream.event_data_typeid, McsEventStream.event_meta_typeid)

    def __repr__(self):
        return '<McsEventStream object at '+str(hex(id(self)))+', EventEntities='+str(len(self.EventEntity))+'>'

    def _read_entities(self, entity_class_name):
        """
        reads event stream entities into entity type associated objects

        :param entity_class_name: class name of the associated stream entity 
        """
        try:
            cls = globals()[entity_class_name] #getattr(__name__, entity_class_name)
        except KeyError as err:
            print(err)
        self._entities = list()
        for entity_type in np.unique(self.EventData.EventID):
            self._entities.append(cls(self, entity_type))
        
    @property
    def EventData(self):
        """
        All events of all event entities in the stream
        """
        return self.Data[0]

    @property
    def EventMeta(self):
        """
        The meta data for all event entities
        """
        return self.Meta[0]

    @property
    def EventEntity(self):
        """
        All event entities in the stream
        """
        if self._entities == None:
            self._read_entities('McsEventEntity')
        return self._entities

class McsEventEntity(McsStreamEntity):
    """
    Container class for Event Entity object
    """

    def __init__(self, parent, event_id):
        """
        Initializes an Mcs EventEntity Object

        :param parent: parent McsEventStream instances
        :param event_id: identifier of the event entity (the type of event)
        """
        super().__init__(parent, event_id)

    def _get_data_by_header(self, header):
        index = list(np.where(self.mcspy_parent.data[0]['EventID'] == self._entity_id)[0])
        return self.mcspy_parent.data[0][index,header]

    def _get_meta_by_header(self, header):
        index = list(np.where(self.mcspy_parent.meta[0]['EventID'] == self._entity_id)[0])
        return self.mcspy_parent.meta[0][index,header]

    def __getattr__(self, name):
        if name in self.mcspy_parent._get_data_headers():
            return self._get_data_by_header(name)
        if name in self.mcspy_parent._get_meta_headers():
            return self._get_meta_by_header(name)
        raise AttributeError('{} is not a valid event attribute'.format(name))

    def __str__(self):
        return 'Event Entity \"' + self.meta['Label'][0].decode('UTF-8') + '\" Headers:\n'+'Event Data Headers: '+', '.join(self.mcspy_parent._get_data_headers())+'\nEvent Meta Headers: '+', '.join(self.mcspy_parent._get_meta_headers())

    def __repr__(self):
        return '<McsEventEntity object at '+str(hex(id(self)))+', Label='+ self.meta['Label'][0].decode('UTF-8') +', events='+str(len(self.data))+'>'
    
    @property
    def events(self):
        """
        The ids, timestamps and durations of the occurences of the event entity 
        """
        index = list(np.where(self.mcspy_parent.EventData['EventID'] == self._entity_id)[0])
        return self.mcspy_parent.EventData[index]

    @property
    def meta(self):
        """
        The meta data for an event entity
        """
        index = list(np.where(self.mcspy_parent.EventMeta['EventID'] == self._entity_id)[0])
        return self.mcspy_parent.EventMeta[index]

class McsSensorStream(McsStream):
    """
    Container class for one Event Stream.
    """

    sensor_data_typeid =  "49da47df-f397-4121-b5da-35317a93e705"
    sensor_meta_typeid =  "ab2aa189-2e72-4148-a2ef-978119223412"

    def __init__(self, sensor_stream_grp):
        """
        Initializes an sensor stream object

        :param sensor_stream_grp: folder of the HDF5 file that contains the data of this sensor stream
        """
        super().__init__(sensor_stream_grp, McsSensorStream.sensor_data_typeid, McsSensorStream.sensor_meta_typeid)

    def __repr__(self):
        return '<McsSensorStream object at '+str(hex(id(self)))+'>'

    def _read_entities(self, entity_class_name):
        """
        reads event stream entities into entity type associated objects

        :param entity_class_name: class name of the associated stream entity 
        """
        try:
            cls = globals()[entity_class_name] #getattr(__name__, entity_class_name)
        except KeyError as err:
            print(err)
        self._entities = list()
        for entity_type in np.unique(self.EventData.EventID):
            self._entities.append(cls(self, entity_type))

    def _get_sensor_sweeps_by_number(self, key):
        """
        retrieves all dataset that belong to sweep number 'key' in a sensor stream

        :param key: key as int that identifies a sweep in the sensor stream

        :return: list of id set that correlates with sweeps with number 'key'
        """
        if isinstance(key, int):
            out = list()
            for child in self._h5py_object.keys():
                sweep_number = [int(s) for s in child if s.isdigit()]
                try:
                    if sweep_number[1] == key and self._data_typeid == self.h5py_object[child].attrs["ID.TypeID"].decode('UTF-8'):
                        out.append(next(id_set for id_set in self._child_inventory if child in id_set))
                except IndexError:
                    pass
            return out
        raise KeyError('{} must be an instance of int!'.format(key))

    def _get_sensor_rois_by_number(self, key):
        """
        retrieves all dataset that belong to roi number 'key' in a sensor stream

        :param key: key as int that identifies a roi in the sensor stream

        :return: list of id set that correlates with roi with number 'key'
        """
        if isinstance(key, int):
            out = list()
            for child in self._h5py_object.keys():
                roi_number = [int(s) for s in child if s.isdigit()]
                try:
                    if roi_number[0] == key and self._data_typeid == self.h5py_object[child].attrs["ID.TypeID"].decode('UTF-8'):
                        out.append(next(id_set for id_set in self._child_inventory if child in id_set))
                except IndexError:
                    pass
            return out
        raise KeyError('{} must be an instance of int!'.format(key))

    @property
    def DataChunk(self):
        """
        The groups of data that have been acquired. Intended for acquisition of multiple time windows
        """
        sweep_numbers = np.unique(self.SensorMeta.GroupID).tolist()
        out = dict()
        for sweep_number in sweep_numbers:
            out[sweep_number] = _list_property.McsProxy(self._get_sensor_sweeps_by_number(sweep_number), obj=self, fget=self._get_children, fset=None, fdel=None)
        return out
    
    @property
    def Regions(self):
        """
        The regions of interest (ROI) on the sensor for which data has been acquired, usually from a rectangular subset of the sensors
        """
        roi_numbers = np.unique(self.SensorMeta.RegionID).tolist()
        out = dict()
        for roi_number in roi_numbers:
            out[roi_number] = _list_property.McsProxy(self._get_sensor_rois_by_number(roi_number), obj=self, fget=self._get_children, fset=None, fdel=None)
        return out
        
    @property
    def SensorData(self):
        """
        The sensor data as a numpy array of shape (frames x sensors_Y x sensors_X)
        """
        return self.Data

    @property
    def SensorMeta(self):
        """
        The meta data for the acquired sensor data
        """
        return self.Meta[0]

class McsSensorEntity(McsDataset, McsStreamEntity):
    """
    Container class for one McsSensorEntity - a sensor stream entity.
    """
    def __init__(self, sensor_stream_entity_dataset, mcspy_parent):
        """
        initializes a new McsSensorEntity from a h5py_dataset of a hdf5 SensorData entity

        :param channel_stream_entity_dataset: h5py_dataset of a cahn
        """
        id = re.findall(r'\d+', sensor_stream_entity_dataset.name.split('/')[-1] )
        id = tuple(map(int, id))
        McsDataset.__init__(self, sensor_stream_entity_dataset)
        McsStreamEntity.__init__( self, mcspy_parent, id )
        self.dimensions = '[ \'frames\' x \'region height\' x \'region width\' ]'

    def __repr__(self):
        return '<McsSensorEntity object at '+str(hex(id(self)))+', frames='+str(self.shape[0])+', height='+str(self.shape[1])+', width='+str(self.shape[2])+'>'

class McsSpikeStream(McsStream):
    """
    Container class for one Spike Stream.
    """
    
    spike_data_typeid =  "3e8aaacc-268b-4057-b0bb-45d7dc9ec73b"
    spike_meta_typeid =  "e1d7616f-621c-4a26-8f60-a7e63a9030b7"

    def __init__(self, spike_stream_grp, spike_data_typeid="3e8aaacc-268b-4057-b0bb-45d7dc9ec73b"):
        """
        Initializes an event stream object

        :param spike_stream_grp: folder of the HDF5 file that contains the data of this spike stream
        """
        super().__init__(spike_stream_grp, spike_data_typeid, McsSpikeStream.spike_meta_typeid)

    def __repr__(self):
        return '<McsSpikeStream object at '+str(hex(id(self)))+'>'

    def get_spikes_at_sensor(self, sensor_id):
        """
        retrieves all spikes that occured at the sensor with id sensor_id

        :param sensor_id: valid identifier for a sensor on the MCS CMOS chip as int: 1 <= sensor_id <= 65*65

        :return: numpy structured array of all spikes that have been detected on the sensor with id sensor_id
        """
        if not isinstance(sensor_id, int):
            raise TypeError('The given sensor id \'{}\' must be of type \'int\'.'.format(sensor_id))
        if not sensor_id in range(1,65**2+1):
            raise ValueError('The given sensor id \'{}\' must satify 1 <= sensor_id <= 65*65'.format(sensor_id))
        row_numbers = np.where(self.SpikeData['SensorID'] == sensor_id)[0]
        return self.SpikeData[tuple(row_numbers),]

    def get_spikes_in_interval(self, interval):
        """
        Retrieves all spikes that occured in a given time interval. Intervals exceeding the time range of the dataset will throw a warning, 
        and retrieval of maximally sized subset of the interval is attempted.

        :param interval: interval in s as instance of
                                                - list(start,stop) of length 2
                                                - tuple(start,stop) of length 2
                         start must be a number, stop must be a number or the keyword 'end', start and stop must satisfy start < stop
                         
        :result: numpy structured array which includes all spikes occuring in the given interval
        """
        if not isinstance(interval, (list,tuple)):
            raise TypeError('The given interval \'{}\' must be an instance of list(start,stop) or tuple(start,stop)'.format(interval))
        if not len(interval) == 2:
            raise ValueError('The given interval \'{}\' must provide a start and a stop value'.format(interval))
        if not isinstance(interval[0], Number):
            raise TypeError('start \'{}\' must be a number'.format(interval[0]))
        if not (isinstance(interval[1], Number) or interval[1]=='end'):
            raise TypeError('stop \'{}\' must be a number or the keyword \'end\''.format(interval[0]))
        #tick = self.SpikeMeta.Tick[0]
        if interval[1]=='end':
            interval[1] = self.SpikeData.TimeStamp[-1]*(10**-6)
        if interval[0]>=interval[1]:
            raise ValueError('start={} and stop={} do not satisfy start < stop'.format(interval[0], interval[1]))
        interval[0] *= (10**6)
        interval[1] *= (10**6)
        row_numbers = np.logical_and(interval[0] <= self.SpikeData['TimeStamp'], self.SpikeData['TimeStamp'] <= interval[1])
        return self.SpikeData[row_numbers,]

    def get_spike_timestamps_at_sensors(self, sensor_ids):
        """
        Retrieves all spike timestamps for all given sensors as a dictionary

        :param sensor_ids: valid identifiers for sensors on the MCS CMOS chip as int: 1 <= sensor_id <= 65*65

        :return: dictionary of all spike timestamps that have been detected on the given sensors. Key: sensor_id, value: spike timestamps
        """
        if isinstance(sensor_ids, Number):
            sensor_ids = [sensor_ids]
        spike_dict = {}
        for sensor in sensor_ids:
            spikes = self.get_spikes_at_sensor(sensor)
            timestamps = [t[1] for t in spikes]
            spike_dict[sensor] = timestamps
        return spike_dict

    def get_spike_cutouts_at_sensor(self, sensor_id):
        """
        Retrieves the spike cutouts for all spikes for the given sensor_id

        :param sensor_id: valid identifier for a sensor on the MCS CMOS chip as int: 1 <= sensor_id <= 65*65

        :return: Numpy array spikes x samples of the spike cutouts
        """
        spikes = self.get_spikes_at_sensor(sensor_id)
        cutouts = [list(s)[2:] for s in spikes]
        return np.array(cutouts)

    @property
    def SpikeStreamEntity(self):
        return self.Data
        
    @property
    def SpikeData(self):
        """
        The detected spikes, each with a sensor ID, a timestamp and (optionally) with a cutout
        """
        return self.Data[0]

    @property
    def SpikeMeta(self):
        """
        The meta data for spike detection, e.g. pre- and post interval
        """
        return self.Meta[0]

class McsSpikeEntity(McsDataset, McsStreamEntity):
    """
    Container class for one SpikeStream Entity.
    """
    def __init__(self, spike_stream_entity_dataset, mcspy_parent):
        """
        initializes a new McsSpikeEntity from a h5py_dataset of a hdf5 SpikeData entity

        :param spike_stream_entity_dataset: h5py_dataset of a cahn
        """
        McsDataset.__init__(self, spike_stream_entity_dataset)
        McsStreamEntity.__init__(self, mcspy_parent, 0)
        self.dimensions = '[ \'# of spikes\' x \'SensorID + Timestamp + n cutout values\' ]'

    def __repr__(self):
        return '<McsSpikeEntity object at '+str(hex(id(self)))+', spikes='+str(self.shape[0])+'>'

class McsSegmentStream(McsStream):
    """
    Container class for one segment stream of different segment entities
    """
    def __init__(self, segment_stream_grp):
        super().__init__(self, segment_stream_grp)

    def __repr__(self):
        return '<McsSegmentStream object at '+str(hex(id(self)))+'>'

class McsSegmentStreamEntity(object):
    """
    Segment entity class,
    """
    pass

class McsTimeStampStream(McsStream):
    """
    Container class for one TimeStamp stream 
    """
    def __init__(self, timestamp_stream_grp):
        super().__init__(self, timestamp_stream_grp)

    def __repr__(self):
        return '<McsTimeStampStream object at '+str(hex(id(self)))+'>'

class McsTimeStampStreamEntity(object):
    """
    TimeStamp stream entity class
    """
    pass

class NetworkExplorer(McsGroup):
    """
    Container class for a NetworkExplorer object
    """

    def  __init__(self, network_explorer_group):
        self.__network_explorer_group       = network_explorer_group
        self._sta_key_type                  = self.get_sta_entity_by_sourceID
        self._map_sensorID_to_sourceID      = {}
        self._sta_entity                    = None
        super().__init__(network_explorer_group)

    def __str__(self):
        """
        provides a string method that prepares the object attributes for printing
        """
        if(self.__network_explorer_group):
            out =   'The NetworkExplorer objects hold the following information:\n'
            out +=  'Attributes:\n'
            for (name, value) in self.__network_explorer_group.attrs.items():
                if hasattr(value, "decode"):
                    out += ("\t"+name.ljust(20)+"\t"+value.decode('UTF-8')+"\n")
                else:
                    out += ("\t"+name.ljust(20)+"\t"+str(value).strip('[]')+"\n")
            out +=  '------------------------------------------\nSubgroups\n'
            out +=  '------------------------------------------\nDatasets\n'
            for (name, value) in self.__network_explorer_group.items():
                if hasattr(value, "decode"):
                    out += ("\t"+name.ljust(20)+"\t"+value.decode('UTF-8')+"\n")
                else:
                    out += ("\t"+name.ljust(20)+"\t"+str(value).strip('[]')+"\n")
            return out

    def __repr__(self):
        if self._sta_entity is None:
            return '<NetworkExplorer object at '+str(hex(id(self)))+'>'
        else:
            return '<NetworkExplorer object at '+str(hex(id(self)))+', entities='+str(len(self._sta_entity))+'>'

    def _read_sta_entities(self):
        """
        Retrieves all stored sta_entities and saves them in a dictionary with special access methods
        """
        self._sta_entity = {}
        self._neural_network = {}
        entity_dict = {}
        sta_type     = b'442b7514-fe3a-4c66-8ae9-4f249ef48f2f'
        spikes_type  = b'1b4e0b8b-6af1-4b55-a685-a6d28a922eb3'
        stddev_type  = b'a056832a-013d-4215-b8a6-cb1debeb1c56'
        network_type = b'235c3c9c-1e94-40ca-8d4b-c5db5b079f16'
        for (name, _) in self.__network_explorer_group.items():
            type_id = self.__network_explorer_group[name].attrs['ID.TypeID']
            if type_id in [sta_type, spikes_type, stddev_type]:
                source_id = int(self.__network_explorer_group[name].attrs['SourceID'])
                if not source_id in entity_dict.keys():
                    entity_dict[source_id] = {}
                entity_dict[source_id][type_id] = name
            elif type_id == network_type:
                self._read_neural_network(self.__network_explorer_group[name])
                    
        for source_id in entity_dict.keys():
            new_sta_entity = STAEntity(self.__network_explorer_group,entity_dict[source_id][sta_type], 
                                       entity_dict[source_id].get(spikes_type, None), entity_dict[source_id].get(stddev_type, None),
                                       self.get_axon_for_entity_by_sourceID(source_id))
            self._sta_entity[new_sta_entity._sta_entity_sourceID] = new_sta_entity
            self._map_sensorID_to_sourceID[new_sta_entity._sta_entity_sensorID] = new_sta_entity._sta_entity_sourceID

    def _read_neural_network(self, group):
        for entry in group:
            unit_id = int(entry['UnitID'])
            axon_id = int(entry['AxonID'])
            segment_id = int(entry['SegmentID'])
            if not unit_id in self._neural_network.keys():
                self._neural_network[unit_id] = {}
            if axon_id != -1 and not axon_id in self._neural_network[unit_id].keys():
                self._neural_network[unit_id][axon_id] = {}
            if segment_id != -1 and not segment_id in self._neural_network[unit_id][axon_id].keys():
                self._neural_network[unit_id][axon_id][segment_id] = []
            if axon_id != -1 and segment_id != -1:
                self._neural_network[unit_id][axon_id][segment_id].append((entry['PosX'], entry['PosY']))

    def get_sta_entity_by_sourceID(self, key):
        """
        Retrieve the STA Entity for the given source ID. 

        :param key: A valid source ID. See the sourceIDs attribute for a list of valid source IDs

        :return: The STA Entity for the given source ID
        """
        if self._sta_entity is None:
            self._read_sta_entities()
        try:
            return self._sta_entity[key]
        except KeyError:
            print("Oops!  That was not a valid sourceID. For a list of all available sourceIDs use My_sta_explorer_object.sourceIDs ")
        except TypeError as err:
            print(err)

    def get_sta_entity_by_sensorID(self, key):
        """
        Retrieve the STA Entity for the given sensor ID. 

        :param key: A valid sensor ID. See the sensorIDs attribute for a list of valid sensor IDs

        :return: The STA Entity for the given sensor ID
        """
        if self._sta_entity is None:
            self._read_sta_entities()
        try:
            return self._sta_entity[self._map_sensorID_to_sourceID[key]]
        except KeyError:
            print("Oops!  That was not a valid sensorID. For a list of all available sensorIDs use My_sta_explorer_object.sensorIDs ")
        except TypeError as err:
            print(err)

    def get_sta_entity(self, key):
        """
        Retrieve the STA Entity for the given key.

        :param key: A valid key, either a sensor or a source ID, depending on the sta_key_type attribute

        :return: The STA Entity for the given key
        """
        if self._sta_entity is None:
            self._read_sta_entities()
        return self._sta_key_type(key)
        #if self.sta_key_type == 'sensorID':
        #    return self._sta_entity[self._map_sensorID_to_sourceID[key]].data
        #return self._sta_entity[key].data

    def set_sta_entity(self, key, value):
        """
        Sets an entity to a value
        """
        dprint("Setting _sta_entity[",key,"] to ",value)
        self._sta_entity[key]=value

    def del_sta_entity(self, key):
        """
        Deletes an entity
        """
        dprint("Deleting _sta_entity[",key,"]")
        del self._sta_entity[key]

    def get_axon_for_entity_by_sourceID(self, key, axon=1, segment=1):
        """
        Retrieve the path of the axon for a given sensor or source ID. 

        :param key: A valid key, either a sensor or a source ID, depending on the sta_key_type attribute
        :param axon: A valid axon ID, in case multiple axons have been found for a unit. Default: 1
        :param segment: A valid axon ID, in case multiple segments have been found for an axon. Default: 1

        :return: The axon path as a list of (X,Y) tuples in sensor coordinates. Returns None if no axon is found
        """
        if self._sta_entity is None:
            self._read_sta_entities()
        if not key in self._neural_network.keys():
            return None
        if not axon in self._neural_network[key] or not segment in self._neural_network[key][axon]:
            return None
        return self._neural_network[key][axon][segment]

    sta_entity = DictProperty_for_Classes(get_sta_entity, set_sta_entity, del_sta_entity)

    @property
    def sta_key_type(self):
        """
        The type of key used in the access functions. Either 'sourceID' or 'sensorID'
        """
        if self._sta_key_type == self.get_sta_entity_by_sourceID:
            return 'sourceID'
        elif self._sta_key_type == self.get_sta_entity_by_sensorID:
            return 'sensorID'
        else:
            return None

    @sta_key_type.setter
    def sta_key_type(self, value):
        if value=='sourceID':
            print("All STA entity retrievals are now by "+value)
            _sta_key_type = self.get_sta_entity_by_sourceID
        elif value=='sensorID':
            print("All STA entity retrievals are now by "+value)
            _sta_key_type = self.get_sta_entity_by_sourceID
        else:
            print("Oops!  That is not a valid way of selecting STA entities.  Try 'sourceID' or 'sensorID'")

    @property
    def sourceIDs(self):
        """
        A list of valid source IDs
        """
        if self._sta_entity is None:
            self._read_sta_entities()
        return list(self._map_sensorID_to_sourceID.values())

    @property
    def sensorIDs(self):
        """
        A list of valid sensor IDs
        """
        if self._sta_entity is None:
            self._read_sta_entities()
        return list(self._map_sensorID_to_sourceID.keys())

    @property
    def attributes(self):
        return self.__network_expl_group.attrs.items()

class STAEntity(object):
    """
    Container Class for a STAEntity object
    """
    def __init__(self, sta_explorer, sta_entity, spikes_entity=None, stastddev_entity=None, axon=None):
        self._sta_explorer             = sta_explorer
        self._sta_entity_string        = sta_entity
        self._sta_entity_sourceID      = int(sta_explorer[sta_entity].attrs['SourceID'])
        self._sta_entity_sensorID      = int(sta_explorer[sta_entity].attrs['SensorID'])
        x,y                            = McsCMOSMEAData.sensorID_to_coordinates(self._sta_entity_sensorID)
        self._sta_entity_coordinates   = np.array([int(x),int(y)])
        self._spikes_entity            = spikes_entity
        self._stastddev_entity         = stastddev_entity
        self._axon                     = axon

    def __repr__(self):
        return '<STAEntity object at '+str(hex(id(self)))+'>'

    @property
    def data(self):
        """
        The STA data as a numpy array of shape (frames x sensors_Y x sensor_X)
        """
        return self._sta_explorer[self._sta_entity_string]

    @property
    def spikes(self):
        """
        Detected spikes in the STA
        """
        if self._spikes_entity is None:
            return None
        return self._sta_explorer[self._spikes_entity]

    @property
    def sta_stddev(self):
        """
        Returns the standard deviation for each channel in the STA. Used for spike detection on the STA
        """
        if self._stastddev_entity is None:
            return None
        return self._sta_explorer[self._stastddev_entity]

    @property
    def sensor_coordinates(self):
        """
        Returns the STA source coordinates on the chip as [X,Y]. Note: X and Y are 1-based
        """
        return self._sta_entity_coordinates

    @property
    def axon(self):
        """
        Returns the axon path as a list of (X,Y) tuples in sensor coordinates. None if no axon has been found
        """
        return self._axon


class SpikeExplorer(McsSpikeStream):
    """
    Container Class for an SpikeExplorer object
    """
    def __init__(self, spike_explorer_group):
        self._spike_explorer_group = spike_explorer_group
        super().__init__(spike_explorer_group, spike_data_typeid='1b4e0b8b-6af1-4b55-a685-a6d28a922eb3')

    def __repr__(self):
        return '<SpikeExplorer object at '+str(hex(id(self)))+'>'

class SpikeSorter(McsGroup):
    """
    Container for SpikeSorter object
    """
    def __init__(self, spike_sorter_group):
        self._spike_sorter_group = spike_sorter_group
        self._units = {}
        super().__init__(spike_sorter_group)
        unit_type = b'0e5a97df-9de0-4a22-ab8c-54845c1ff3b9'
        for (name, _) in self._spike_sorter_group.items():
            type_id = self._spike_sorter_group[name].attrs['ID.TypeID']
            if type_id == unit_type:
                unit_id = int(self._spike_sorter_group[name].attrs['UnitID'])
                child = self.ischild(name)
                self._units[unit_id] = getattr(self, child.mcspy)

    def __repr__(self):
        return '<SpikeSorter object at '+str(hex(id(self)))+'>'

    def get_unit(self, unit_id):
        """
        Retrieves a single unit by its UnitID

        :param unit_id: A valid unit ID. 
        """
        return self._units[unit_id]

    def get_units_by_id(self):
        """
        Returns a list of units sorted by unit ID
        """
        unit_ids = list(self._units.keys())
        unit_ids.sort()
        return [self._units[i] for i in unit_ids]

    def get_units_by_measure(self, measure, descending=True):
        """
        Returns a list of units ordered by the given quality measure.

        :param measure: The name of a quality measure. See get_unit_measures() for a list of valid quality measure names.
        :param descending: The ordering of the list. Default: True (=descending order)
        """
        if not measure in self.get_unit_measures():
            raise ValueError(measure + " is not a valid measure. See get_unit_measures() for valid parameters")
        m = self.Units[measure]
        idx = np.argsort(m)
        ids = self.Units['UnitID'][idx]
        if descending:
            ids = ids[::-1]
        return [self._units[i] for i in ids]

    def get_unit_measures(self):
        """
        Returns a list of the available unit quality measure names
        """
        lf = list(self.Units.dtype.fields)
        return lf[5:]


class SpikeSorterUnitEntity(McsGroup):
    """
    Container for Spike Sorter Units
    """
    def __init__(self, unit_group):
        self._unit_group = unit_group
        self._unit_entity_unitID       = int(unit_group.attrs['UnitID'])
        self._unit_entity_sensorID     = int(unit_group.attrs['SensorID'])
        x,y                            = McsCMOSMEAData.sensorID_to_coordinates(self._unit_entity_sensorID)
        self._unit_entity_coordinates  = np.array([int(x),int(y)])
        self._included_peaks = None
        super().__init__(unit_group)

    def __repr__(self):
        return '<SpikeSorterUnitEntity object at '+str(hex(id(self)))+', id='+str(self._unit_entity_unitID)+', sensor='+str(self._unit_entity_coordinates)+'>'

    def get_peaks(self):
        """
        Retrieves all peaks in the source signal where the 'IncludePeak' flag is set. 
        """
        if self._included_peaks is None:
            self._included_peaks = self.Peaks['IncludePeak'] == 1
        return self.Peaks[self._included_peaks]

    def get_peaks_timestamps(self):
        """
        Retrieves the timestamps for all peaks in the source signal where the 'IncludePeak' flag is set.
        """
        return self.get_peaks()['Timestamp']

    def get_peaks_amplitudes(self):
        """
        Retrieves the peak amplitudes for all peaks in the source signal where the 'IncludePeak' flag is set.
        """
        return self.get_peaks()['PeakAmplitude']

    def get_peaks_cutouts(self):
        """
        Retrieves the cutouts for all peaks in the source signal where the 'IncludePeak' flag is set.
        """
        peaks = self.get_peaks()
        cutouts = [list(p)[3:] for p in peaks]
        return np.stack(cutouts)

    def get_measures(self):
        """
        Gets a list of valid unit quality measures names
        """
        lf = list(self.Unit_Info.dtype.fields)
        return lf[5:]

    def get_measure(self, measure):
        """
        Gets a quality measure for this unit

        :param measure: The name of a quality measure. See get_measures() for a list of valid quality measure names.
        """
        if not measure in self.get_measures():
            raise ValueError(measure + " is not a valid measure. See get_measures() for valid parameters")
        return self.Unit_Info[measure][0]

class FilterTool(McsGroup):
    """
    Container for FilterTool object
    """
    def __init__(self, filter_tool):
        self._filter_tool = filter_tool
        super().__init__(filter_tool)

    def __repr__(self):
        return '<SpikeSorter object at '+str(hex(id(self)))+'>'

class ActivitySummary(McsGroup):
    """
    Container for ActivitySummary object
    """
    def __init__(self, activity_summary):
        self._activity_summary = activity_summary
        super().__init__(activity_summary)

    def __repr__(self):
        return '<ActivitySummary object at '+str(hex(id(self)))+'>'
