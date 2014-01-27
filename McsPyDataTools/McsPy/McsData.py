import sys
import h5py

class RawData(object):
    """description of class"""
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.h5_file = h5py.File(raw_data_path,'r')
        self.__session_info = dict()

    def __str__(self):
        #return '[RawData: File Path %s]' % self.raw_data_path
        return super(RawData, self).__str__()

    def __get_session_info(self):
        data_attrs = self.h5_file['Data'].attrs.iteritems()
        session_attributes = data_attrs;
        session_info = {}
        for (name, value) in session_attributes: 
            #print(name, value)
            session_info[name] = value; #value.rstrip()

        return session_info

    @property
    def session_info(self):
        #if (self.__session_info.len() == 0): 
        self.__session_info = self.__get_session_info();
        return self.__session_info
            
