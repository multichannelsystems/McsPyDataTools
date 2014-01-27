import sys
import h5py
import datetime
import math
import uuid

# day -> number of clr ticks (100 ns)
time_tick_clr = 24 * 60 * 60 * (10**7)


class RawData(object):
    """This class holds all information of a complete MCS raw data file"""
    def __init__(self, raw_data_path):
        self.raw_data_path = raw_data_path
        self.h5_file = h5py.File(raw_data_path,'r')
        self.__get_session_info()
        self.__recordings = None

    def __str__(self):
        #return '[RawData: File Path %s]' % self.raw_data_path
        return super(RawData, self).__str__()

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
        self.date =  datetime.datetime.fromordinal(int(math.ceil(self.date_in_clr_ticks / time_tick_clr)) + 1)
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



class AnalogStream(object):
    """Container class for one analog stream"""
    def __init__(self, stream_grp):
        self.__stream_grp = stream_grp
        self.__get_stream_info()

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
