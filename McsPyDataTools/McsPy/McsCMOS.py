"""
    McsCMOS
    ~~~~~~~

    Wrapper and Helper to access MCS CMOS Data within H5 Files 
    
    :copyright: (c) 2014 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""


import h5py
import numpy

class CMOSData(h5py.File):
    """Wrapper for a HDF5 File containing CMOS Data
    """
    def __init__(self,path):
        """Creates a CMOSData file and links it to a H5 File
        :param path: Path to a H5 File containing CMOS Data
        :type path: string 
        """
        super(CMOSData,self).__init__(path,mode='r')

        McsHdf5Protocols.ch

        # -- map data --
        self.raw_data= self['/Data/Recording_0/FrameStream/Stream_0/FrameDataEntity_0/FrameData']
        self.conv_factors= self['/Data/Recording_0/FrameStream/Stream_0/FrameDataEntity_0/ConversionFactors']

        # - proxy -
        self.conv_data = CMOSConvProxy(self)

        # -- map meta --
        self.meta={}
        
        # - from InfoFrame
        info_frame= self['/Data/Recording_0/FrameStream/Stream_0/InfoFrame']
        
        for key in info_frame.dtype.names:
            self.meta[key]=info_frame[key][0]

        if("Tick" in self.meta):
            self.meta["FrameRate"] = 10.0**6/self.meta["Tick"]
        
        # - from File
        for key,value in self.attrs.items():
            self.meta[key]=value

        # - from Data Group
        for key,value in self['/Data'].attrs.items():
            self.meta[key]=value

class CMOSConvProxy:
    """Proxy that transparently converts raw data to calibrated data.
    """

    def __init__(self,parent):
        """Creates a new CMOSConvProxy
        :param parent: Object that can provide raw_data and conv_factors
        :type parent: CMOSData
        """
        self._parent=parent
        self.dtype=numpy.int32

    def __getitem__(self,slices):
        """Sliced access to converted data
        :param slices: Data-slices to retrive
        :returns: converted data
        """
        raw_data=self._parent.raw_data.__getitem__(slices)
        conv_fasctors=self._parent.conv_factors.__getitem__((slices[0],slices[1]))
        return (raw_data*conv_fasctors).astype(self.dtype)

    @property
    def shape(self):
        """Shape of the data
        """
        return self._parent.raw_data.shape

