import sys
import h5py
import datetime
import math
import uuid
import numpy as np

from McsPy import ureg, Q_, supported_mcs_hdf5_versions

# day -> number of clr ticks (100 ns)
clr_tick = 100 * ureg.ns
day_to_clr_time_tick = 24 * 60 * 60 * (10**7)


class RawData(object):
    """This class holds all information of a complete MCS raw data file"""
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.h5_file = h5py.File(raw_data_path,'r')
        self.__validate_mcs_hdf5_version()
        self.__get_session_info()
        self.__recordings = None

    def __str__(self):
        #return '[RawData: File Path %s]' % self.raw_data_path
        return super(RawData, self).__str__()

    def __validate_mcs_hdf5_version(self):
        root_grp = self.h5_file['/']
        if ('McsHdf5Version' in root_grp.attrs):
            self.mcs_hdf5_version = root_grp.attrs['McsHdf5Version']
            if ((self.mcs_hdf5_version < supported_mcs_hdf5_versions[0]) or 
                (supported_mcs_hdf5_versions[1] < self.mcs_hdf5_version)):
                raise IOError('Given HDF5 file has version %s and supported are all versions from %s to %s' % 
                              (self.mcs_hdf5_version, supported_mcs_hdf5_versions[0], supported_mcs_hdf5_versions[1]))    
        else:
            raise IOError('The root group of this HDF5 file has no MCS-HDF5-Version attribute -> this file is not supported by McsPy!')

    def __get_session_info(self):
        data_attrs = self.h5_file['Data'].attrs.iteritems()
        session_attributes = data_attrs;
        session_info = {}
        for (name, value) in session_attributes: 
            #print(name, value)
            session_info[name] = value #.rstrip()
        self.comment = session_info['Comment'].rstrip()
        self.clr_date = session_info['Date'].rstrip()
        self.date_in_clr_ticks = session_info['DateInTicks']
        self.date =  datetime.datetime.fromordinal(int(math.ceil(self.date_in_clr_ticks / day_to_clr_time_tick)) + 1)
        #self.file_guid = session_info['FileGUID'].rstrip()
        self.file_guid = uuid.UUID(session_info['FileGUID'].rstrip()) 
        self.mea_id = session_info['MeaID']
        self.mea_name = session_info['MeaName'].rstrip()
        self.program_name = session_info['ProgramName'].rstrip()
        self.program_version = session_info['ProgramVersion'].rstrip()
        #return session_info

    def __read_recordings(self):
        data_folder = self.h5_file['Data']
        if (len(data_folder) > 0):
            self.__recordings = {}
        for (name, value) in data_folder.iteritems():
            print(name,value)
            recording_name = name.split('_')
            if ((len(recording_name) == 2) and (recording_name[0] == 'Recording')):
                self.__recordings[int(recording_name[1])] = Recording(value)

    @property
    def recordings(self):
        if (self.__recordings is None): 
            self.__read_recordings()
        return self.__recordings
            

class Recording(object):
    """Container class for one recording"""
    def __init__(self, recording_grp):
        self.__recording_grp = recording_grp
        self.__get_recording_info()
        self.__analog_streams = None

    def __get_recording_info(self):
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
        analog_stream_folder = self.__recording_grp['AnalogStream']
        if (len(analog_stream_folder) > 0):
            self.__analog_streams = {}
        for (name, value) in analog_stream_folder.iteritems():
            print(name,value)
            stream_name = name.split('_')
            if ((len(stream_name) == 2) and (stream_name[0] == 'Stream')):
                self.__analog_streams[int(stream_name[1])] = AnalogStream(value)

    @property
    def analog_streams(self):
        if (self.__analog_streams is None): 
            self.__read_analog_streams()
        return self.__analog_streams

    @property
    def duration_time(self):
        dur_time = (self.duration - self.timestamp) * 100 * ureg.ns
        return dur_time

class AnalogStream(object):
    """Container class for one analog stream"""
    def __init__(self, stream_grp):
        self.__stream_grp = stream_grp
        self.__get_stream_info()
        self.__read_channels()

    def __get_stream_info(self):
        stream_info = {}
        for (name, value) in self.__stream_grp.attrs.iteritems(): 
            #print(name, value)
            stream_info[name] = value
        self.data_subtype = stream_info['DataSubType'].rstrip()
        self.label = stream_info['Label'].rstrip()
        self.source_stream_guid = uuid.UUID(stream_info['SourceStreamGUID'].rstrip()) 
        self.stream_guid = uuid.UUID(stream_info['StreamGUID'].rstrip()) 
        self.stream_type = stream_info['StreamType'].rstrip()

    def __read_channels(self):
        assert len(self.__stream_grp) == 3
        for (name, value) in self.__stream_grp.iteritems():
            print name, value
        # Read time stamp index of channels:
        #ts_index = self.__stream_grp['ChannelDataTimeStamps']
        #self.time_stamp_index = np.empty(ts_index.shape, dtype = ts_index.dtype)
        #ts_index.read_direct(self.time_stamp_index)
        self.time_stamp_index = self.__stream_grp['ChannelDataTimeStamps'][...]
        
        # Read infos per channel 
        ch_infos = self.__stream_grp['InfoChannel'][...]
        self.channel_infos = {}
        self.__map_row_to_channel_id = {}
        for channel_info in ch_infos:
            self.channel_infos[channel_info['ChannelID']] = ChannelInfo(channel_info)
            self.__map_row_to_channel_id[channel_info['RowIndex']] = channel_info['ChannelID']

        # Connect the data set 
        self.channel_data = self.__stream_grp['ChannelData']

    def get_channel_in_range(self, channel_id, idx_start, idx_end):
        if (channel_id in self.channel_infos.keys()):
            if (idx_start < 0):
                idx_start = 0
            if (idx_end > self.channel_data.shape[1]):
                idx_end = self.channel_data.shape[1]
            signal = self.channel_data[self.channel_infos[channel_id].row_index, idx_start : idx_end]
            scale = self.channel_infos[channel_id].adc_step.magnitude
            #scale = self.channel_infos[channel_id].get_field('ConversionFactor') * (10**self.channel_infos[channel_id].get_field('Exponent'))
            signal_corrected =  (signal - self.channel_infos[channel_id].get_field('ADZero'))  * scale
            return (signal_corrected, str(self.channel_infos[channel_id].adc_step.units))

    def get_channel_timepoints(self, channel_id, idx_start, idx_end):
        if (channel_id in self.channel_infos.keys()):
            start_ts = 0L
            channel = self.channel_infos[channel_id]
            tick = channel.get_field('Tick')
            for ts_range in self.time_stamp_index.T:
                if (ts_range[2] < idx_start): # start is behind the end of this range ->
                    continue
                else:
                    start_ts = ts_range[0] + idx_start * tick # time stamp of first index
                if (idx_end <= ts_range[2]):
                    time_range = start_ts + np.arange(0, idx_end - idx_start, 1) * tick
                else:
                    time_range = start_ts + np.arange(0, ts_range[2] - idx_start, 1) * tick
                    idx_start = ts_range[2] + 1
                if 'time' in locals():
                    time = np.append(time,time_range)
                else:
                    time = time_range
            return (time * clr_tick.to_base_units().magnitude, clr_tick.to_base_units().units)

    #def get_signal_in_range(self):
    #    signal = (self.channel_data[...] - self.channel_infos[0].get_field('ADZero')) * self.channel_infos[0].get_field('ConversionFactor') * (10**self.channel_infos[0].get_field('Exponent'))   
    #    return signal

class ChannelInfo(object):
    """Contains all meta data for one channel"""
    def __init__(self, ch_info):
        self.info = {}
        for name in ch_info.dtype.names:
            self.info[name] = ch_info[name]

    def get_field(self, name):
         return self.info[name]
    @property
    def row_index(self):
        return self.info['RowIndex']

    @property
    def sampling_frequency(self):
        frequency = 1 / self.sampling_tick.to_base_units()
        return frequency.to(ureg.Hz)

    @property
    def sampling_tick(self):
        tick_time = self.info['Tick']  * clr_tick # clr tick is 100 ns
        return tick_time

    @property
    def label(self):
        return self.info['Label']

    @property
    def data_type(self):
        return self.info['RawDataType']

    @property
    def adc_step(self):
        unit_name = self.info['Unit']
        # Should be tested that unit_name is a available in ureg (unit register)
        step = self.info['ConversionFactor'] * (10 ** self.info['Exponent']) * ureg[unit_name]
        return step