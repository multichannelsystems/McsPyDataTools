import unittest
import McsData
import datetime

from McsPy import ureg, Q_

test_raw_data_file_path = "d:\\Programming\\MCSuite\\McsPyDataTools\\McsPyDataTools\\McsPyTests\\TestData\\Experiment.h5"

@unittest.skip("showing the principle structure of python unit tests")
class Test_TestRawDataStructures(unittest.TestCase):
    def test_A(self):
        self.fail("Not implemented")

class Test_RawData(unittest.TestCase):
    def setUp(self):
        self.raw_data = McsData.RawData(test_raw_data_file_path)

class Test_RawDataContainer(Test_RawData):
    # Test MCS-HDF5 version
    def test_mcs_hdf5_version(self):
        self.assertEqual(self.raw_data.mcs_hdf5_version, 1, 'The MCS-HDF5-Version is different from the expected one!')

    # Test session:
    def test_session_attributes(self):
        self.assertEqual(self.raw_data.comment, '', 'Comment is different!')
        self.assertEqual(self.raw_data.clr_date, 'Mittwoch, 15. Januar 2014', 'Clr-Date is different!')
        self.assertEqual(self.raw_data.date_in_clr_ticks, 635253816315519835, 'Clr-Date-Ticks are different!')
        self.assertEqual(self.raw_data.date, datetime.datetime(2014,1,15), 'Date is different!');
        self.assertEqual(str(self.raw_data.file_guid), '285e858a-5541-41d5-a3e5-5e3ef84d23f7', 'FileGUID is different!')
        self.assertEqual(self.raw_data.mea_id, -1, 'MeaID is different!')
        self.assertEqual(self.raw_data.mea_name, '', 'MeaName is different!')
        self.assertEqual(self.raw_data.program_name, 'Multi Channel Experimenter', 'Program name is different!')
        self.assertEqual(self.raw_data.program_version, '0.8.4.3', 'Program version is different!') 


    # Test recording:
    def test_count_recordings(self):
        self.assertEqual(len(self.raw_data.recordings), 1, 'There should be only one recording!')

    def test_recording_attributes(self):
        first_recording = self.raw_data.recordings[0]
        self.assertEqual(first_recording.comment, '', 'Recording comment is different!')
        self.assertEqual(first_recording.duration, 53700000, 'Recording duration is different!')
        self.assertEqual(first_recording.label, '', 'Recording label is different!')
        self.assertEqual(first_recording.recording_id, 0, 'Recording ID is different!')
        self.assertEqual(first_recording.recording_type, 'xyz', 'Recording type is different!')
        self.assertEqual(first_recording.timestamp, 45700000, 'Recording time stamp is different!')
        self.assertEqual(first_recording.duration_time.to(ureg.sec), 0.8 * ureg.sec, 'Recording time stamp is different!')

    # Test analog streams:
    def test_count_analog_streams(self):
         self.assertEqual(len(self.raw_data.recordings[0].analog_streams), 1, 'There should be only one recording!')

    def test_analog_stream_attributes(self):
        first_analog_stream = self.raw_data.recordings[0].analog_streams[0]
        self.assertEqual(first_analog_stream.data_subtype, 'Electrode', 'Analog stream data sub type is different!')
        self.assertEqual(first_analog_stream.label, '', 'Analog stream label is different!')
        self.assertEqual(str(first_analog_stream.source_stream_guid), '00000000-0000-0000-0000-000000000000', 'Analog stream source GUID is different!')
        self.assertEqual(str(first_analog_stream.stream_guid), 'b616b008-c7a8-47c3-b835-44a531cbe079', 'Analog stream GUID is different!')
        self.assertEqual(first_analog_stream.stream_type, 'Analog', 'Analog stream type is different!')
    
    def test_analog_stream_data(self):
        data_set = self.raw_data.recordings[0]. analog_streams[0].channel_data
        self.assertEqual(data_set.shape, (8, 8100), 'Shape of dataset is different!')
        
        time_stamp_index = self.raw_data.recordings[0]. analog_streams[0].time_stamp_index
        self.assertEqual(time_stamp_index.shape, (3, 1), 'Shape of time stamp index is different!')

        channel_infos =  self.raw_data.recordings[0]. analog_streams[0].channel_infos
        self.assertEqual(len(channel_infos), 8, 'Number of channel info objects is different!')
        self.assertEqual(len(channel_infos[0].info), 16, 'Number of of components of an channel info object is different!')

if __name__ == '__main__':
    unittest.main()
