"""
    McsCMOS
    ~~~~~~~

    Wrapper and Helper to access MCS CMOS Data within H5 Files 
    
    :copyright: (c) 2018 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

import h5py
import numpy

class CMOSData(h5py.File):
    """
    Wrapper for a HDF5 File containing CMOS Data
    """
    def __init__(self, path):
        """
        Creates a CMOSData file and links it to a H5 File
        :param path: Path to a H5 File containing CMOS Data
        :type path: string 
        """
        super(CMOSData, self).__init__(path, mode='r')
        
        # -- map raw data --
        self.raw_data= self['/Data/Recording_0/FrameStream/Stream_0/FrameDataEntity_0/FrameData']
        self.conv_factors= self['/Data/Recording_0/FrameStream/Stream_0/FrameDataEntity_0/ConversionFactors']

        # - map proxy data -
        self.conv_data = CMOSConvProxy(self)

        # -- map meta --
        self.meta={}
        
        # - from InfoFrame
        info_frame= self['/Data/Recording_0/FrameStream/Stream_0/InfoFrame']
        
        for key in info_frame.dtype.names:
            if hasattr(info_frame[key][0], "decode"):
                self.meta[key]=info_frame[key][0].decode('utf-8')
            else:
                self.meta[key]=info_frame[key][0]

        if("Tick" in self.meta):
            self.meta["FrameRate"] = 10.0**6/self.meta["Tick"]
        
        # - from File
        for key, value in self.attrs.items():
            if hasattr(value, "decode"):
                self.meta[key]= value.decode('utf-8')
            else:
                self.meta[key]= value

        # - from Data Group
        for key, value in self['/Data'].attrs.items():
            if hasattr(value, "decode"):
                self.meta[key]= value.decode('utf-8')
            else:
                self.meta[key]= value

        # -- map events --
        if("EventStream" in self["Data/Recording_0/"].keys()):
            event_group = self["Data/Recording_0/EventStream/Stream_0/"]
            event_info = self["Data/Recording_0/EventStream/Stream_0/InfoEvent"]

            self.events={}
            self.event_frames={}
        
            for key in event_group.keys():
                if "EventEntity" in key:
                    info = event_info["Label"][event_info["EventID"]==int(key.split("_")[1])][0]
                    self.events[info] = event_group[key][0, 0]
                    self.event_frames[info] = event_group[key][0, 0]/self.meta["Tick"]



class CMOSConvProxy:
    """
    Private Class, should be embedded within a CMOSData Object.
    A proxy that transparently converts raw data to calibrated data. 
    """

    def __init__(self, parent):
        """
        Creates a new CMOSConvProxy
        :param parent: Object that can provide raw_data and conv_factors
        :type parent: CMOSData
        """
        self._parent = parent
        self.dtype = numpy.int32

    def __getitem__(self, slices):
        """
        Sliced access to converted data
        :param slices: Data-slices to retrieve
        :returns: converted data
        """
        raw_data = self._parent.raw_data.__getitem__(slices)
        conv_fasctors = self._parent.conv_factors.__getitem__((slices[0], slices[1]))
        return (raw_data*conv_fasctors).astype(self.dtype)

    @property
    def shape(self):
        """
        Shape of the data
        """
        return self._parent.raw_data.shape


class CMOSSpikes(h5py.File):
    """
    Wrapper for a HDF5 File containing CMOS Spike Data.
    Spike Information is accessible through the .spike Member,
    Waveform Information (if available) through the .waveforms Member.
    """
    def __init__(self, path):
        super(CMOSSpikes, self).__init__(path)

        # -- Check for right structure --
        if("data" in self.keys() and "spikes" in self['data'].keys()):
            
            # -- Map Spike-Data to RecordArray
            self.spikes = np.core.records.fromarrays(self['data/spikes'][:,:], 
                                                 names='time, col, row',
                                                 formats = 'int64, int64, int64')
            # -- Map Waveforms to Array
            if("waveforms" in self['data'].keys()):
                self.waveforms = self['data/waveforms'][:,:].transpose()
                
        else:
            raise IOError(path+ " has no valid CMOSSpikeFile Structure")