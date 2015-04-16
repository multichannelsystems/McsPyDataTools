"""
    McsData
    ~~~~~~~

    Data classes to wrap and hide raw data handling of the HDF5 data files. 
    It is based on the MCS-HDF5 definitions of the given compatible versions. 
    
    :copyright: (c) 2014 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

import h5py
import datetime
import math
import uuid
import exceptions
import numpy as np

from McsPy import *
from pint import UndefinedUnitError

mcs_tick = 1 * ureg.us
clr_tick = 100 * ureg.ns

# day -> number of clr ticks (100 ns)
day_to_clr_time_tick = 24 * 60 * 60 * (10**7)

verbose = True

def dprint_name_value(n, v):
    if verbose:
        print(n, v)

class RawData(object):
    """
    This class holds all information of a complete MCS raw data file
    """
    def __init__(self, raw_data_path):
        """
        Crates and initializes a RawData object that provides access to the content of the given MCS-HDF5 file

        :param raw_data_path: path to a HDF5 file that contains raw data encoded in a supported MCS-HDF5 format version
        """
        self.raw_data_path = raw_data_path
        self.h5_file = h5py.File(raw_data_path,'r')
        self.__validate_mcs_hdf5_version()
        self.__get_session_info()
        self.__recordings = None

    def __del__(self):
        self.h5_file.close()

    # Stub for with-Statement:
    #def __enter_(self):
    #    return self
    #
    #def __exit__(self, type, value, traceback):
    #    self.h5_file.close()

    def __str__(self):
        #return '[RawData: File Path %s]' % self.raw_data_path
        return super(RawData, self).__str__()

    def __validate_mcs_hdf5_version(self):
        "Check if the MCS-HDF5 protocol type and version of the file is supported by this class"
        root_grp = self.h5_file['/']
        if ('McsHdf5ProtocolType' in root_grp.attrs):
            self.mcs_hdf5_protocol_type = root_grp.attrs['McsHdf5ProtocolType']
            #if (McsHdf5Protocols.SUPPORTED_PROTOCOLS.has_key(self.mcs_hdf5_protocol_type)):
            if (self.mcs_hdf5_protocol_type == "RawData"):
                self.mcs_hdf5_protocol_type_version = root_grp.attrs['McsHdf5ProtocolVersion']
                supported_versions = McsHdf5Protocols.SUPPORTED_PROTOCOLS[self.mcs_hdf5_protocol_type]
                if ((self.mcs_hdf5_protocol_type_version < supported_versions[0]) or 
                    (supported_versions[1] < self.mcs_hdf5_protocol_type_version)):
                    raise IOError('Given HDF5 file has MCS-HDF5 RawData protocol version %s and supported are all versions from %s to %s' % 
                                  (self.mcs_hdf5_protocol_type_version, supported_versions[0], supported_versions[1]))
            else:
                raise IOError("The root group of this HDF5 file has no 'McsHdf5ProtocolVersion' attribute -> so it could't be checked if the version is supported!")
        else:
            raise IOError("The root group of this HDF5 file has no 'McsHdf5ProtocolType attribute' -> this file is not supported by McsPy!")

    def __get_session_info(self):
        "Read all session metadata"
        data_attrs = self.h5_file['Data'].attrs.iteritems()
        session_attributes = data_attrs;
        session_info = {}
        for (name, value) in session_attributes: 
            #print(name, value)
            session_info[name] = value #.rstrip()
        self.comment = session_info['Comment'].rstrip()
        self.clr_date = session_info['Date'].rstrip()
        self.date_in_clr_ticks = session_info['DateInTicks']
        # self.date =  datetime.datetime.fromordinal(int(math.ceil(self.date_in_clr_ticks / day_to_clr_time_tick)) + 1)
        self.date = datetime.datetime(1, 1, 1) + datetime.timedelta(microseconds = int(self.date_in_clr_ticks)/10)  
        # self.file_guid = session_info['FileGUID'].rstrip()
        self.file_guid = uuid.UUID(session_info['FileGUID'].rstrip())
        self.mea_layout = session_info['MeaLayout'].rstrip() 
        self.mea_sn = session_info['MeaSN'].rstrip()
        self.mea_name = session_info['MeaName'].rstrip()
        self.program_name = session_info['ProgramName'].rstrip()
        self.program_version = session_info['ProgramVersion'].rstrip()
        #return session_info

    def __read_recordings(self):
        "Read all recordings"
        data_folder = self.h5_file['Data']
        if (len(data_folder) > 0):
            self.__recordings = {}
        for (name, value) in data_folder.iteritems():
            dprint_name_value(name,value)
            recording_name = name.split('_')
            if ((len(recording_name) == 2) and (recording_name[0] == 'Recording')):
                self.__recordings[int(recording_name[1])] = Recording(value)

    @property
    def recordings(self):
        "Access recordings"
        if (self.__recordings is None): 
            self.__read_recordings()
        return self.__recordings
            

class Recording(object):
    """
    Container class for one recording
    """
    def __init__(self, recording_grp):
        self.__recording_grp = recording_grp
        self.__get_recording_info()
        self.__analog_streams = None
        self.__frame_streams = None
        self.__event_streams = None
        self.__segment_streams = None
        self.__timestamp_streams = None

    def __get_recording_info(self):
        "Read metadata for this recording"
        recording_info = {}
        for (name, value) in self.__recording_grp.attrs.iteritems(): 
            #print(name, value)
            recording_info[name] = value
        self.comment = recording_info['Comment'].rstrip()
        self.duration = recording_info['Duration']
        self.label = recording_info['Label'].rstrip()
        self.recording_id = recording_info['RecordingID']
        self.recording_type = recording_info['RecordingType'].rstrip()
        self.timestamp = recording_info['TimeStamp']

    def __read_analog_streams(self):
        "Read all contained analog streams"
        analog_stream_folder = self.__recording_grp['AnalogStream']
        if (len(analog_stream_folder) > 0):
            self.__analog_streams = {}
        for (name, value) in analog_stream_folder.iteritems():
            dprint_name_value(name,value)
            stream_name = name.split('_')
            if ((len(stream_name) == 2) and (stream_name[0] == 'Stream')):
                self.__analog_streams[int(stream_name[1])] = AnalogStream(value)

    def __read_frame_streams(self):
        "Read all contained frame streams"
        frame_stream_folder = self.__recording_grp['FrameStream']
        if (len(frame_stream_folder) > 0):
            self.__frame_streams = {}
        for (name, value) in frame_stream_folder.iteritems():
            dprint_name_value(name,value)
            stream_name = name.split('_')
            if ((len(stream_name) == 2) and (stream_name[0] == 'Stream')):
                self.__frame_streams[int(stream_name[1])] = FrameStream(value)

    def __read_event_streams(self):
        "Read all contained event streams"
        event_stream_folder = self.__recording_grp['EventStream']
        if (len(event_stream_folder) > 0):
            self.__event_streams = {}
        for (name, value) in event_stream_folder.iteritems():
            dprint_name_value(name,value)
            stream_name = name.split('_')
            if ((len(stream_name) == 2) and (stream_name[0] == 'Stream')):
                self.__event_streams[int(stream_name[1])] = EventStream(value)

    def __read_segment_streams(self):
        "Read all contained segment streams"
        segment_stream_folder = self.__recording_grp['SegmentStream']
        if len(segment_stream_folder) > 0:
            self.__segment_streams = {}
        for (name, value) in segment_stream_folder.iteritems():
            dprint_name_value(name,value)
            stream_name = name.split('_')
            if (len(stream_name) == 2) and (stream_name[0] == 'Stream'):
                self.__segment_streams[int(stream_name[1])] = SegmentStream(value)

    def __read_timestamp_streams(self):
        "Read all contained timestamp streams"
        timestamp_stream_folder = self.__recording_grp['TimeStampStream']
        if len(timestamp_stream_folder) > 0:
            self.__timestamp_streams = {}
        for (name, value) in timestamp_stream_folder.iteritems():
            dprint_name_value(name,value)
            stream_name = name.split('_')
            if (len(stream_name) == 2) and (stream_name[0] == 'Stream'):
                self.__timestamp_streams[int(stream_name[1])] = TimeStampStream(value)

    @property
    def analog_streams(self):
        "Access analog streams"
        if (self.__analog_streams is None): 
            self.__read_analog_streams()
        return self.__analog_streams

    @property
    def frame_streams(self):
        "Access frame streams"
        if (self.__frame_streams is None): 
            self.__read_frame_streams()
        return self.__frame_streams

    @property
    def event_streams(self):
        "Access event streams"
        if (self.__event_streams is None):
            self.__read_event_streams()
        return self.__event_streams

    @property
    def segment_streams(self):
        "Access segment streams"
        if (self.__segment_streams is None):
            self.__read_segment_streams()
        return self.__segment_streams

    @property
    def timestamp_streams(self):
        "Access timestamp streams"
        if (self.__timestamp_streams is None):
            self.__read_timestamp_streams()
        return self.__timestamp_streams

    @property
    def duration_time(self):
        "Duration of the recording"
        dur_time = (self.duration - self.timestamp) * ureg.us
        return dur_time

class Stream(object):
    """
    Base class for all stream types
    """
    def __init__(self, stream_grp, info_type_name = None):
        """
        Initializes a stream object with its associated HDF5 folder

        :param stream_grp: folder of the HDF5 file that contains the data of this stream
        :param info_type_name: name of the Info-Type as given in class McsHdf5Protocols (default None -> no version check is executed)
        """
        self.stream_grp = stream_grp
        info_version = self.stream_grp.attrs["StreamInfoVersion"]
        if (info_type_name != None):
             McsHdf5Protocols.check_protocol_type_version(info_type_name, info_version)
        self.__get_stream_info()

    def __get_stream_info(self):
        "Read all describing meta data common to each stream -> HDF5 folder attributes"
        stream_info = {}
        for (name, value) in self.stream_grp.attrs.iteritems(): 
            #print(name, value)
            stream_info[name] = value
        self.info_version = stream_info['StreamInfoVersion']
        self.data_subtype = stream_info['DataSubType'].rstrip()
        self.label = stream_info['Label'].rstrip()
        self.source_stream_guid = uuid.UUID(stream_info['SourceStreamGUID'].rstrip()) 
        self.stream_guid = uuid.UUID(stream_info['StreamGUID'].rstrip()) 
        self.stream_type = stream_info['StreamType'].rstrip()

class AnalogStream(Stream):
    """
    Container class for one analog stream of several channels
    """
    def __init__(self, stream_grp):
        """
        Initializes an analog stream object containing several analog channels

        :param stream_grp: folder of the HDF5 file that contains the data of this analog stream
        """
        #McsHdf5Protocols.check_protocol_type_version("AnalogStreamInfoVersion", info_version)
        Stream.__init__(self, stream_grp, "AnalogStreamInfoVersion")
        self.__read_channels()

    def __read_channels(self):
        "Read all channels -> create Info structure and connect datasets"
        assert len(self.stream_grp) == 3
        for (name, value) in self.stream_grp.iteritems():
            dprint_name_value(name, value)
        # Read timestamp index of channels:
        self.timestamp_index = self.stream_grp['ChannelDataTimeStamps'][...]
        
        # Read infos per channel 
        ch_infos = self.stream_grp['InfoChannel'][...]
        ch_info_version = self.stream_grp['InfoChannel'].attrs['InfoVersion']
        self.channel_infos = {}
        self.__map_row_to_channel_id = {}
        for channel_info in ch_infos:
            self.channel_infos[channel_info['ChannelID']] = ChannelInfo(ch_info_version, channel_info)
            self.__map_row_to_channel_id[channel_info['RowIndex']] = channel_info['ChannelID']

        # Connect the data set 
        self.channel_data = self.stream_grp['ChannelData']

    def get_channel_in_range(self, channel_id, idx_start, idx_end):
        """
        Get the signal of the given channel over the curse of time and in its measured range. 

        :param channel_id: ID of the channel
        :param idx_start: index of the first sampled signal value that should be returned (0 <= idx_start < idx_end <= count samples)
        :param idx_end: index of the last sampled signal value that should be returned (0 <= idx_start < idx_end <= count samples)  
        :return: Tuple (vector of the signal, unit of the values)
        """
        if (channel_id in self.channel_infos.keys()):
            if (idx_start < 0):
                idx_start = 0
            if (idx_end > self.channel_data.shape[1]):
                idx_end = self.channel_data.shape[1]
            else:
                idx_end += 1
            signal = self.channel_data[self.channel_infos[channel_id].row_index, idx_start : idx_end]
            scale = self.channel_infos[channel_id].adc_step.magnitude
            #scale = self.channel_infos[channel_id].get_field('ConversionFactor') * (10**self.channel_infos[channel_id].get_field('Exponent'))
            signal_corrected =  (signal - self.channel_infos[channel_id].get_field('ADZero'))  * scale
            return (signal_corrected, self.channel_infos[channel_id].adc_step.units)

    def get_channel_sample_timestamps(self, channel_id, idx_start, idx_end):
        """
        Get the timestamps of the sampled values. 

        :param channel_id: ID of the channel
        :param idx_start: index of the first signal timestamp that should be returned (0 <= idx_start < idx_end <= count samples)
        :param idx_end: index of the last signal timestamp that should be returned (0 <= idx_start < idx_end <= count samples)  
        :return: Tuple (vector of the timestamps, unit of the timestamps)
        """
        if (channel_id in self.channel_infos.keys()):
            start_ts = 0L
            channel = self.channel_infos[channel_id]
            tick = channel.get_field('Tick')
            for ts_range in self.timestamp_index:
                if (idx_end < ts_range[1]): # nothing to do anymore ->
                    break 
                if (ts_range[2] < idx_start): # start is behind the end of this range ->
                    continue
                else:
                    idx_segment = idx_start - ts_range[1]
                    start_ts = ts_range[0] + idx_segment * tick # timestamp of first index
                if (idx_end <= ts_range[2]):
                    time_range = start_ts + np.arange(0, (idx_end - ts_range[1] + 1) - idx_segment, 1) * tick
                else:
                    time_range = start_ts + np.arange(0, (ts_range[2] - ts_range[1] + 1) - idx_segment, 1) * tick
                    idx_start = ts_range[2] + 1
                if 'time' in locals():
                    time = np.append(time,time_range)
                else:
                    time = time_range
            return (time, mcs_tick.units)

class Info(object):
    """
    Base class of all info classes
    """
    def __init__(self, info_data):
        self.info = {}
        for name in info_data.dtype.names:
            self.info[name] = info_data[name]

    def get_field(self, name):
        "Get the field with that name -> access to the raw info array"
        return self.info[name]

    @property
    def group_id(self):
        "Get the id of the group that the objects belongs to"
        return self.info["GroupID"]

    @property
    def label(self):
        "Label of this object"
        return self.info['Label']

    @property
    def data_type(self):
        "Raw data type of this object"
        return self.info['RawDataType']

class InfoSampledData(Info):
    """
    Base class of all info classes for sampled data
    """
    def __init__(self, info):
        """
        Initialize an info object for sampled data

        :param info: array of info descriptors for this info object
        """
        Info.__init__(self, info)

    @property
    def sampling_frequency(self):
        "Get the used sampling frequency in Hz"
        frequency = 1 / self.sampling_tick.to_base_units()
        return frequency.to(ureg.Hz)

    @property
    def sampling_tick(self):
        "Get the used sampling tick"
        tick_time = self.info['Tick']  * mcs_tick
        return tick_time

class ChannelInfo(InfoSampledData):
    """
    Contains all describing meta data for one sampled channel
    """
    def __init__(self, info_version, info):
        """
        Initialize an info object for sampled channel data

        :param info_version: number of the protocol version used by the following info structure
        :param info: array of info descriptors for this channel info object
        """
        InfoSampledData.__init__(self, info)
        McsHdf5Protocols.check_protocol_type_version("InfoChannel", info_version)
        self.__version = info_version

    @property
    def channel_id(self):
        "Get the ID of the channel"
        return self.info['ChannelID']

    @property
    def row_index(self):
        "Get the index of the row that contains the associated channel data inside the data matrix"
        return self.info['RowIndex']

    @property
    def adc_step(self):
        "Size and unit of one ADC step for this channel"
        unit_name = self.info['Unit']
        # Should be tested that unit_name is a available in ureg (unit register)
        step = self.info['ConversionFactor'] * (10 ** self.info['Exponent']) * ureg[unit_name]
        return step
    
    @property
    def version(self):
        "Version number of the Type-Definition"
        return self.__version

class FrameStream(Stream):
    """
    Container class for one frame stream with different entities
    """
    def __init__(self, stream_grp):
        """
        Initializes an frame stream object that contains all frame entities that belong to it.

        :param stream_grp: folder of the HDF5 file that contains the data of this frame stream
        """
        Stream.__init__(self, stream_grp, "FrameStreamInfoVersion")
        self.__read_frame_entities()

    def __read_frame_entities(self):
        "Read all fream entities for this frame stream inside the associated frame entity folder"
        #assert len(self.stream_grp) == 3
        for (name, value) in self.stream_grp.iteritems():
            dprint_name_value(name, value)
        # Read infos per frame 
        fr_infos = self.stream_grp['InfoFrame'][...]
        fr_info_version = self.stream_grp['InfoFrame'].attrs['InfoVersion']
        self.frame_entity = {}
        for frame_entity_info in fr_infos:
            frame_entity_group = "FrameDataEntity_" + str(frame_entity_info['FrameDataID'])
            conv_fact = self.__read_conversion_factor_matrix(frame_entity_group)
            frame_info = FrameEntityInfo(fr_info_version, frame_entity_info, conv_fact)
            self.frame_entity[frame_entity_info['FrameID']] = FrameEntity(self.stream_grp[frame_entity_group], frame_info)

    def __read_conversion_factor_matrix(self, frame_entity_group):
        "Read matrix of conversion factors inside the frame data entity folder"
        frame_entity_conv_matrix = frame_entity_group + "/ConversionFactors"
        conv_fact = self.stream_grp[frame_entity_conv_matrix][...]
        return conv_fact;

class FrameEntity(object):
    """
    Frame entity
    """
    def __init__(self, frame_entity_group, frame_info):
        """
        Initializes an frame entity object

        :param frame_entity_group: folder/group of the HDF5 file that contains the data for this frame entity
        :param frame_info: object of type FrameEntityInfo that contains the description of this frame entity 
        """
        self.info = frame_info
        self.group = frame_entity_group
        self.timestamp_index = self.group['FrameDataTimeStamps'][...]
        # Connect the data set 
        self.data = self.group['FrameData']

    def get_sensor_signal(self, sensor_x, sensor_y , idx_start, idx_end):
        """
        Get the signal of a single sensor over the curse of time and in its measured range. 

        :param sensor_x: x coordinate of the sensor
        :param sensor_y: y coordinate of the sensor
        :param idx_start: index of the first sampled frame that should be returned (0 <= idx_start < idx_end <= count frames)
        :param idx_end: index of the last sampled frame that should be returned (0 <= idx_start < idx_end <= count frames)  
        :return: Tuple (vector of the signal, unit of the values)
        """
        if (sensor_x < 0 or self.data.shape[0] < sensor_x or sensor_y < 0 or self.data.shape[1] < sensor_y):
            raise exceptions.IndexError
        if (idx_start < 0):
            idx_start = 0
        if (idx_end > self.data.shape[2]):
            idx_end = self.data.shape[2]
        else:
            idx_end += 1
        sensor_signal = self.data[sensor_x, sensor_y, idx_start : idx_end]
        scale_factor = self.info.adc_step_for_sensor(sensor_x,sensor_y)
        scale = scale_factor.magnitude
        sensor_signal_corrected =  (sensor_signal - self.info.get_field('ADZero'))  * scale
        return (sensor_signal_corrected, scale_factor.units)

    def get_frame_timestamps(self, idx_start, idx_end):
        """
        Get the timestamps of the sampled frames. 

        :param idx_start: index of the first sampled frame that should be returned (0 <= idx_start < idx_end <= count frames)
        :param idx_end: index of the last sampled frame that should be returned (0 <= idx_start < idx_end <= count frames)  
        :return: Tuple (vector of the timestamps, unit of the timestamps)
        """
        if (idx_start < 0 or self.data.shape[2] < idx_start or idx_end < idx_start or self.data.shape[2] < idx_end):
                raise exceptions.IndexError
        start_ts = 0L
        tick = self.info.get_field('Tick')
        for ts_range in self.timestamp_index:
            if (idx_end < ts_range[1]): # nothing to do anymore ->
                break 
            if (ts_range[2] < idx_start): # start is behind the end of this range ->
                continue
            else:
                idx_segment = idx_start - ts_range[1]
                start_ts = ts_range[0] + idx_segment * tick # timestamp of first index
            if (idx_end <= ts_range[2]):
                time_range = start_ts + np.arange(0, (idx_end - ts_range[1] + 1) - idx_segment, 1) * tick
            else:
                time_range = start_ts + np.arange(0, (ts_range[2] - ts_range[1] + 1) - idx_segment, 1) * tick
                idx_start = ts_range[2] + 1
            if 'time' in locals():
                time = np.append(time,time_range)
            else:
                time = time_range
        return (time, mcs_tick.units)        
        

class Frame(object):
    """
    Frame definition
    """
    def __init__(self, left, top, right, bottom):
        self.__left = left
        self.__top = top
        self.__right = right
        self.__bottom = bottom

    @property
    def left(self):
        return self.__left

    @property
    def top(self):
        return self.__top

    @property
    def right(self):
        return self.__right

    @property
    def bottom(self):
        return self.__bottom

    @property
    def width(self):
        return self.__right - self.__left + 1

    @property
    def height(self):
        return self.__bottom - self.__top + 1

class FrameEntityInfo(InfoSampledData):
    """
    Contains all describing meta data for one frame entity
    """
    def __init__(self, info_version, info, conv_factor_matrix):
        """
        Initializes an describing info object that contains all descriptions of this frame entity.

        :param info_version: number of the protocol version used by the following info structure
        :param info: array of frame entity descriptors as represented by one row of the InfoFrame structure inside the HDF5 file
        :param conv_factor_matrix: matrix of conversion factor as represented by the ConversionFactors structure inside one FrameDataEntity folder of the HDF5 file
        """
        InfoSampledData.__init__(self, info)
        McsHdf5Protocols.check_protocol_type_version("FrameEntityInfo", info_version)
        self.__version = info_version
        self.frame = Frame(info['FrameLeft'], info['FrameTop'], info['FrameRight'], info['FrameBottom'])
        self.reference_frame = Frame(info['ReferenceFrameLeft'], info['ReferenceFrameTop'], info['ReferenceFrameRight'], info['ReferenceFrameBottom'])
        self.conversion_factors = conv_factor_matrix

    @property
    def frame_id(self):
        "ID of the frame"
        return self.info['FrameID']
    
    @property
    def sensor_spacing(self):
        "Returns the spacing of the sensors in micro-meter"
        return self.info['SensorSpacing']

    @property
    def adc_basic_step(self):
        "Returns the value of one basic ADC-Step"
        unit_name = self.info['Unit']
        # Should be tested that unit_name is a available in ureg (unit register)
        basic_step = (10 ** self.info['Exponent']) * ureg[unit_name]
        return basic_step

    def adc_step_for_sensor(self, x, y):
        "Returns the combined (virtual) ADC-Step for the sensor (x,y)"
        adc_sensor_step = self.conversion_factors[x,y] * self.adc_basic_step
        return adc_sensor_step

    @property
    def version(self):
        "Version number of the Type-Definition"
        return self.__version

class EventStream(Stream):
    """
    Container class for one event stream with different entities
    """
    def __init__(self, stream_grp):
        """
        Initializes an event stream object that contains all entities that belong to it.

        :param stream_grp: folder of the HDF5 file that contains the data of this event stream
        """
        Stream.__init__(self, stream_grp, "EventStreamInfoVersion")
        self.__read_event_entities()

    def  __read_event_entities(self):
        "Create all event entities of this event stream"
        for (name, value) in self.stream_grp.iteritems():
            dprint_name_value(name, value)
        # Read infos per event entity 
        event_infos = self.stream_grp['InfoEvent'][...]
        event_entity_info_version = self.stream_grp['InfoEvent'].attrs['InfoVersion']
        self.event_entity = {}
        for event_entity_info in event_infos:
            event_entity_name = "EventEntity_" + str(event_entity_info['EventID'])
            event_info = EventEntityInfo(event_entity_info_version, event_entity_info)
            self.event_entity[event_entity_info['EventID']] = EventEntity(self.stream_grp[event_entity_name], event_info)
        
class EventEntity(object):
    """
    Event entity class
    """
    def __init__(self, event_data, event_info):
        """
        Initializes an event entity object

        :param event_data: dataset of the HDF5 file that contains the data for this event entity
        :param event_info: object of type EventEntityInfo that contains the description of this entity 
        """
        self.info = event_info
        # Connect the data set 
        self.data = event_data

    @property
    def count(self):
        """Number of contained events"""
        dim = self.data.shape 
        return dim[1]

    def __handle_indices(self, idx_start, idx_end):
        """Check indices for consistency and set default values if nothing was provided"""
        if idx_start == None:
            idx_start = 0
        if idx_end == None:
            idx_end = self.count
        if idx_start < 0 or self.data.shape[1] < idx_start or idx_end < idx_start or self.data.shape[1] < idx_end:
                raise exceptions.IndexError
        return (idx_start, idx_end)

    def get_events(self, idx_start = None, idx_end = None):
        """Get all n events of this entity of the given index range (idx_start <= idx < idx_end)
        
        :param idx_start: start index of the range (including), if nothing is given -> 0
        :param idx_end: end index of the range (excluding, if nothing is given -> last index
        :return: Tuple of (2 x n matrix of timestamp (1. row) and duration (2. row), Used unit of time)   
        """
        idx_start, idx_end = self.__handle_indices(idx_start, idx_end)
        events = self.data[...,idx_start:idx_end]
        return (events * mcs_tick.magnitude, mcs_tick.units)

    def get_event_timestamps(self, idx_start = None, idx_end = None):
        """Get all n event timestamps of this entity of the given index range 
        
        :param idx_start: start index of the range, if nothing is given -> 0
        :param idx_end: end index of the range, if nothing is given -> last index
        :return: Tuple of (n-length array of timestamps, Used unit of time)   
        """
        idx_start, idx_end = self.__handle_indices(idx_start, idx_end)
        events = self.data[0, idx_start:idx_end]
        return (events * mcs_tick.magnitude, mcs_tick.units)

    def get_event_durations(self, idx_start = None, idx_end = None):
        """Get all n event durations of this entity of the given index range 
        
        :param idx_start: start index of the range, if nothing is given -> 0
        :param idx_end: end index of the range, if nothing is given -> last index
        :return: Tuple of (n-length array of duration, Used unit of time)   
        """
        idx_start, idx_end = self.__handle_indices(idx_start, idx_end)
        events = self.data[1, idx_start:idx_end]
        return (events * mcs_tick.magnitude, mcs_tick.units)

class EventEntityInfo(Info):
    """
    Contains all meta data for one event entity
    """
    def __init__(self, info_version, info):
        """
        Initializes an describing info object with an array that contains all descriptions of this event entity.

        :param info_version: number of the protocol version used by the following info structure
        :param info: array of event entity descriptors as represented by one row of the InfoEvent structure inside the HDF5 file
        """
        Info.__init__(self, info)
        McsHdf5Protocols.check_protocol_type_version("EventEntityInfo", info_version)
        self.__version = info_version
        source_channel_ids = map(lambda x: int(x), info['SourceChannelIDs'].split(','))
        source_channel_labels = map(lambda x: x.strip(), info['SourceChannelLabels'].split(','))
        self.__source_channels = {}
        for idx, id in enumerate(source_channel_ids):
            self.__source_channels[id] = source_channel_labels[idx]
            
    @property
    def id(self):
        "Event ID"
        return self.info['EventID']

    @property
    def raw_data_bytes(self):
        "Lenght of raw data in bytes"
        return self.info['RawDataBytes']

    @property
    def source_channel_ids(self):
        "ID's of all channels that were involved in the event generation." 
        return self.__source_channels.keys()

    @property
    def source_channel_labels(self):
        "Labels of the channels that were involved in the event generation."
        return self.__source_channels;

    @property
    def version(self):
        "Version number of the Type-Definition"
        return self.__version

class SegmentStream(Stream):
    """
    Container class for one segment stream of different segment entities
    """
    def __init__(self, stream_grp):
        Stream.__init__(self, stream_grp, "SegmentStreamInfoVersion")
        self.__read_segment_entities()

    def  __read_segment_entities(self):
        "Read and initialize all segment entities"
        for (name, value) in self.stream_grp.iteritems():
            dprint_name_value(name, value)
        # Read infos per segment entity 
        segment_infos = self.stream_grp['InfoSegment'][...]
        segment_info_version = self.stream_grp['InfoSegment'].attrs['InfoVersion']
        self.segment_entity = {}
        for segment_entity_info in segment_infos:
            segment_entity_data_name = "SegmentData_" + str(segment_entity_info['SegmentID'])
            segment_entity_ts_name = "SegmentData_ts_" + str(segment_entity_info['SegmentID'])
            ch_info_version = self.stream_grp['SourceInfoChannel'].attrs['InfoVersion']
            source_channel_infos = self.__get_source_channel_infos(ch_info_version, self.stream_grp['SourceInfoChannel'][...])
            segment_info = SegmentEntityInfo(segment_info_version, segment_entity_info, source_channel_infos)
            self.segment_entity[segment_entity_info['SegmentID']] = SegmentEntity(self.stream_grp[segment_entity_data_name], 
                                                                                self.stream_grp[segment_entity_ts_name],
                                                                                segment_info)

    def __get_source_channel_infos(self, ch_info_version, source_channel_infos):
        "Create a dictionary of all present source channels"
        source_channels = {}
        for source_channel_info in source_channel_infos:
            source_channels[source_channel_info['ChannelID']] = ChannelInfo(ch_info_version, source_channel_info)
        return source_channels

class SegmentEntity(object):
    """
    Segment entity class
    """
    def __init__(self, segment_data, segment_ts, segment_info):
        """
        Initializes a segment entity.

        :param segment_data: 2d-matrix (one segment) or 3d-cube (n segments) of segment data
        :param segment_ts: timestamp vector for every segment (2d) or multi-segments (3d)
        :param segment_info: segment info object that contains all meta data for this segment entity
        :return: Segment entity
        """
        self.info = segment_info
        # connect the data set 
        self.data = segment_data
        # connect the timestamp vector
        self.data_ts = segment_ts
        assert self.segment_sample_count == self.data_ts.shape[1], 'Timestamp index is not compatible with dataset!!!'

    @property
    def segment_sample_count(self):
        "Number of contained samples of segments (2d) or multi-segments (3d)"
        dim = self.data.shape 
        if len(dim) == 3:
            return dim[2]
        else:
            return dim[1]

    @property
    def segment_count(self):
        "Number of segments that are sampled for one time point (2d) -> 1 and (3d) -> n"
        dim = self.data.shape 
        if len(dim) == 3:
            return dim[1]
        else:
            return 1

    def __handle_indices(self, idx_start, idx_end):
        """Check indices for consistency and set default values if nothing was provided"""
        sample_count = self.segment_sample_count
        if idx_start == None:
            idx_start = 0
        if idx_end == None:
            idx_end = sample_count
        if idx_start < 0 or sample_count < idx_start or idx_end < idx_start or sample_count < idx_end:
                raise exceptions.IndexError
        return (idx_start, idx_end)

    def get_segment_in_range(self, segment_id, flat = False, idx_start = None, idx_end = None):
        """
        Get the a/the segment signals in its measured range. 

        :param segment_id: id resp. number of the segment (0 if only one segment is present or the index inside the multi-segment collection)
        :param flat: true -> one-dimensional vector of the sequentially ordered segments, false -> k x n matrix of the n segments of k sample points  
        :param idx_start: index of the first segment that should be returned (0 <= idx_start < idx_end <= count segments)
        :param idx_end: index of the last segment that should be returned (0 <= idx_start < idx_end <= count segments)  
        :return: Tuple (of a flat vector of the sequentially ordered segments or a k x n matrix of the n segments of k sample 
        points depending on the value of *flat* , and the unit of the values)
        """
        if segment_id in self.info.source_channel_of_segment.keys():
            idx_start, idx_end = self.__handle_indices(idx_start, idx_end)
            if self.segment_count == 1:
                signal = self.data[..., idx_start : idx_end]
            else:
                signal = self.data[..., segment_id, idx_start : idx_end]
            source_channel = self.info.source_channel_of_segment[segment_id]
            scale = source_channel.adc_step.magnitude
            signal_corrected =  (signal - source_channel.get_field('ADZero'))  * scale
            if flat:
                signal_corrected = np.reshape(signal_corrected, -1, 'F')
            return (signal_corrected, source_channel.adc_step.units)

    def get_segment_sample_timestamps(self, segment_id, flat = False, idx_start = None, idx_end = None):
        """
        Get the timestamps of the sample points of the measured segment. 

        :param segment_id: id resp. number of the segment (0 if only one segment is present or the index inside the multi-segment collection)
        :param flat: true -> one-dimensional vector of the sequentially ordered segment timestamps, false -> k x n matrix of the k timestamps of n segments  
        :param idx_start: index of the first segment for that timestamps should be returned (0 <= idx_start < idx_end <= count segments)
        :param idx_end: index of the last segment for that timestamps should be returned (0 <= idx_start < idx_end <= count segments)  
        :return: Tuple (of a flat vector of the sequentially ordered segments or a k x n matrix of the n segments of k sample 
        points depending on the value of *flat* , and the unit of the values)
        """
        if segment_id in self.info.source_channel_of_segment.keys():
            idx_start, idx_end = self.__handle_indices(idx_start, idx_end)
            data_ts = self.data_ts[idx_start:idx_end]
            source_channel = self.info.source_channel_of_segment[segment_id]
            signal_ts = np.zeros((self.data.shape[0], data_ts.shape[1]), dtype = np.long)
            segment_ts = np.zeros(self.data.shape[0], dtype = np.long) + source_channel.sampling_tick.magnitude
            segment_ts[0] = 0
            segment_ts = np.cumsum(segment_ts)
            for i in range(data_ts.shape[1]):
                col = (data_ts[0,i] - self.info.pre_interval.magnitude) + segment_ts
                signal_ts[:, i] = col
            if flat:
                signal_ts = np.reshape(signal_ts, -1, 'F')
            return (signal_ts , source_channel.sampling_tick.units)

class SegmentEntityInfo(Info):
    """
    Contains all meta data for one segment entity
    """
    def __init__(self, info_version, info, source_channel_infos):
        """
        Initializes an describing info object with an array that contains all descriptions of this segment entity.

        :param info_version: number of the protocol version used by the following info structure
        :param info: array of segment entity descriptors as represented by one row of the SegmentEvent structure inside the HDF5 file
        :param source_channel_infos: dictionary of source channels from where the segments were taken 
        """
        Info.__init__(self, info)
        McsHdf5Protocols.check_protocol_type_version("SegmentEntityInfo", info_version)
        self.__version = info_version
        source_channel_ids = map(lambda x: int(x), info['SourceChannelIDs'].split(','))
        self.source_channel_of_segment = {}
        for idx, id in enumerate(source_channel_ids):
            self.source_channel_of_segment[idx] = source_channel_infos[id]
            
    @property
    def id(self):
        "Segment ID"
        return self.info['SegmentID']

    @property
    def pre_interval(self):
        "Interval [start of the segment <- defining event timestamp]"
        return self.info['PreInterval'] * mcs_tick

    @property
    def post_interval(self):
        "Interval [defining event timestamp -> end of the segment]"
        return self.info['PostInterval'] * mcs_tick

    @property
    def type(self):
        return self.info['SegmentType']

    @property
    def count(self):
        "Count of segments inside the segment entity"
        return len(self.source_channel_of_segment)

    @property
    def version(self):
        "Version number of the Type-Definition"
        return self.__version

class TimeStampStream(Stream):
    """
    Container class for one timestamp stream with different entities
    """
    def __init__(self, stream_grp):
        """
        Initializes an timestamp stream object that contains all entities that belong to it.

        :param stream_grp: folder of the HDF5 file that contains the data of this timestamp stream
        """
        Stream.__init__(self, stream_grp, "TimeStampStreamInfoVersion")
        self.__read_timestamp_entities()

    def  __read_timestamp_entities(self):
        "Create all timestamp entities of this timestamp stream"
        for (name, value) in self.stream_grp.iteritems():
            dprint_name_value(name, value)
        # Read infos per timestamp entity 
        timestamp_infos = self.stream_grp['InfoTimeStamp'][...]
        timestamp_info_version = self.stream_grp['InfoTimeStamp'].attrs['InfoVersion']
        self.timestamp_entity = {}
        for timestamp_entity_info in timestamp_infos:
            timestamp_entity_name = "TimeStampEntity_" + str(timestamp_entity_info['TimeStampEntityID'])
            timestamp_info = TimeStampEntityInfo(timestamp_info_version, timestamp_entity_info)
            self.timestamp_entity[timestamp_entity_info['TimeStampEntityID']] = TimeStampEntity(self.stream_grp[timestamp_entity_name], timestamp_info)
        
class TimeStampEntity(object):
    """
    Time-Stamp entity class
    """
    def __init__(self, timestamp_data, timestamp_info):
        """
        Initializes an timestamp entity object

        :param timestamp_data: dataset of the HDF5 file that contains the data for this timestamp entity
        :param timestamp_info: object of type TimeStampEntityInfo that contains the description of this entity 
        """
        self.info = timestamp_info
        # Connect the data set 
        self.data = timestamp_data

    @property
    def count(self):
        """Number of contained timestamps"""
        dim = self.data.shape 
        return dim[1]

    def __handle_indices(self, idx_start, idx_end):
        """Check indices for consistency and set default values if nothing was provided"""
        if idx_start == None:
            idx_start = 0
        if idx_end == None:
            idx_end = self.count
        if idx_start < 0 or self.data.shape[1] < idx_start or idx_end < idx_start or self.data.shape[1] < idx_end:
                raise exceptions.IndexError
        return (idx_start, idx_end)

    def get_timestamps(self, idx_start = None, idx_end = None):
        """Get all n time stamps of this entity of the given index range (idx_start <= idx < idx_end)
        
        :param idx_start: start index of the range (including), if nothing is given -> 0
        :param idx_end: end index of the range (excluding, if nothing is given -> last index
        :return: Tuple of (n-length array of timestamps, Used unit of time)    
        """
        idx_start, idx_end = self.__handle_indices(idx_start, idx_end)
        timestamps = self.data[idx_start:idx_end]
        scale = self.info.measuring_unit
        return (timestamps, scale)

class TimeStampEntityInfo(Info):
    """
    Contains all meta data for one timestamp entity
    """
    def __init__(self, info_version, info):
        """
        Initializes an describing info object with an array that contains all descriptions of this timestamp entity.

        :param info_version: number of the protocol version used by the following info structure
        :param info: array of event entity descriptors as represented by one row of the InfoTimeStamp structure inside the HDF5 file
        """
        Info.__init__(self, info)
        McsHdf5Protocols.check_protocol_type_version("TimeStampEntityInfo", info_version)
        self.__version = info_version
        source_channel_ids = map(lambda x: int(x), info['SourceChannelIDs'].split(','))
        source_channel_labels = map(lambda x: x.strip(), info['SourceChannelLabels'].split(','))
        self.__source_channels = {}
        for idx, id in enumerate(source_channel_ids):
            self.__source_channels[id] = source_channel_labels[idx]
            
    @property
    def id(self):
        "Timestamp entity ID"
        return self.info['TimeStampEntityID']

    @property
    def unit(self):
        "Unit in which the timestamps are measured"
        return self.info['Unit']

    @property
    def exponent(self):
        "Exponent for the unit in which the timestamps are measured"
        return int(self.info['Exponent'])

    @property
    def measuring_unit(self):
        try:
            provided_base_unit = ureg.parse_expression(self.unit);
        except  UndefinedUnitError as unit_undefined:
            print("Could not find unit \'%s\' in the Unit-Registry" % self.unit) #unit_undefined.unit_names
            return None
        else:
            return (10**self.exponent) * provided_base_unit;

    @property
    def data_type(self):
        "DataType for the timestamps"
        return 'Long'

    @property
    def source_channel_ids(self):
        "ID's of all channels that were involved in the timestamp generation." 
        return self.__source_channels.keys()

    @property
    def source_channel_labels(self):
        "Labels of the channels that were involved in the timestamp generation."
        return self.__source_channels;

    @property
    def version(self):
        "Version number of the Type-Definition"
        return self.__version