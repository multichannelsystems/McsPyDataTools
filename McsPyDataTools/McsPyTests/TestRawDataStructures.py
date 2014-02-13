import unittest
import McsData
import datetime
import exceptions
import numpy as np

from McsPy import ureg, Q_

test_raw_data_file_path = "d:\\Programming\\MCSuite\\McsPyDataTools\\McsPyDataTools\\McsPyTests\\TestData\\Experiment.h5"

test_raw_frame_data_file_path = "d:\\Programming\\MCSuite\\McsPyDataTools\\McsPyDataTools\\McsPyTests\\TestData\\Sensor200ms.h5"

#@unittest.skip("showing the principle structure of python unit tests")
#class Test_TestRawDataStructures(unittest.TestCase):
#    def test_A(self):
#        self.fail("Not implemented")

class Test_RawData(unittest.TestCase):
    def setUp(self):
        self.raw_data = McsData.RawData(test_raw_data_file_path)
        self.raw_frame_data = McsData.RawData(test_raw_frame_data_file_path)

class Test_RawDataContainer(Test_RawData):
    # Test MCS-HDF5 version
    def test_mcs_hdf5_version(self):
        self.assertEqual(self.raw_data.mcs_hdf5_version, 1, 'The MCS-HDF5-Version is different from the expected one!')

    def test_mcs_hdf5_version_frame(self):
        self.assertEqual(self.raw_frame_data.mcs_hdf5_version, 1, 'The MCS-HDF5-Version is different from the expected one!')

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

    # Test frame streams:
    def test_count_frame_streams(self):
         self.assertEqual(len(self.raw_frame_data.recordings[0].frame_streams), 1, 'There should be only one frame stream!')

    def test_frame_stream_attributes(self):
        first_frame_stream = self.raw_frame_data.recordings[0].frame_streams[0]
        self.assertEqual(first_frame_stream.data_subtype, 'Unknown', 'Frame stream data sub type is different!')
        self.assertEqual(first_frame_stream.label, '', 'Frame stream label is different!')
        self.assertEqual(str(first_frame_stream.source_stream_guid), '9d5c7555-5804-4d87-bfd1-0eeafc20e3e2', 'Frame stream source GUID is different!')
        self.assertEqual(str(first_frame_stream.stream_guid), 'd9b62795-9fb8-48c7-a9bf-97cb2f66a7b2', 'Frame stream GUID is different!')
        self.assertEqual(first_frame_stream.stream_type, 'Frame', 'Frame stream type is different!')
    
    def test_frame_infos(self):
        conv_fact_expected = np.zeros(shape=(65,65), dtype=np.int32) + 100
        conv_fact_expected[64, : ] = 1
        conv_fact_expected[ : , 64] = 1
        info_expected = {
                     'FrameLeft': 1, 'Exponent': -9, 'RawDataType': 'Short', 'LowPassFilterCutOffFrequency': '-1', 'Label': 'ROI 1', 
                     'FrameTop': 1, 'ADZero': 0, 'LowPassFilterOrder': -1, 'ReferenceFrameTop': 1, 'FrameRight': 65, 'HighPassFilterType': '', 
                     'Tick': 200, 'SensorSpacing': 1, 'HighPassFilterCutOffFrequency': '-1', 'FrameDataID': 0, 'FrameID': 1, 'GroupID': 1, 
                     'ReferenceFrameRight': 65, 'ReferenceFrameBottom': 65, 'LowPassFilterType': '', 'HighPassFilterOrder': -1, 
                     'ReferenceFrameLeft': 1, 'FrameBottom': 65, 'Unit': 'V'
        }
        frame_infos =  self.raw_frame_data.recordings[0].frame_streams[0].frame_infos
        self.assertEqual(len(frame_infos), 1, 'Number of frame info objects is different!')
        self.assertEqual(len(frame_infos[1].info), 24, 'Number of of components of an channel info object is different!')
        frame_info = frame_infos[1]
        info_key_diff = set(frame_info.info.keys()) - set(info_expected.keys())
        if not info_key_diff:
            for key, value in frame_info.info.items():
                self.assertEqual(
                    value, info_expected[key], 
                    "Frame info object for key '%(k)s' is ('%(val)s') not as expected ('%(ex_val)s')!" % {'k':key, 'val':value, 'ex_val':info_expected[key]}
                )
        self.assertEqual(frame_info.frame.height, 65, "Frame height was '%s' and not '65' as expected!" % frame_info.frame.height)
        self.assertEqual(frame_info.frame.width, 65, "Frame height was '%s' and not '65' as expected!" % frame_info.frame.width)
        self.assertEqual(frame_info.reference_frame.height, 65, "Frame height was '%s' and not '65' as expected!" % frame_info.reference_frame.height)
        self.assertEqual(frame_info.reference_frame.width, 65, "Frame height was '%s' and not '65' as expected!" % frame_info.reference_frame.width)
        self.assertEqual(frame_info.adc_basic_step.magnitude, 10**-9, "Frame height was '%s' and not '10^-9 V' as expected!" % frame_info.adc_basic_step)
        self.assertEqual(frame_info.adc_step_for_sensor(0,0).magnitude, 100 * 10**-9, "Frame height was '%s' and not '100 * 10^-9 V' as expected!" % frame_info.adc_step_for_sensor(0,0))
        self.assertEqual(frame_info.adc_step_for_sensor(1,1).magnitude, 100 * 10**-9, "Frame height was '%s' and not '100 * 10^-9 V' as expected!" % frame_info.adc_step_for_sensor(1,1))
        self.assertEqual(frame_info.adc_step_for_sensor(63,63).magnitude, 100 * 10**-9, "Frame height was '%s' and not '100 * 10^-9 V' as expected!" % frame_info.adc_step_for_sensor(63,63))
        self.assertTrue((frame_info.conversion_factors == conv_fact_expected).all(), "Frame sensor conversion factors matrix is different from the expected one!")
        self.assertRaises(exceptions.IndexError, frame_info.adc_step_for_sensor, 65,65)          

    def test_frame_data(self):
        frame_data = self.raw_frame_data.recordings[0].frame_streams[0].frame_data[0]
        frame = frame_data[:,:,1]
        a = frame_data[0,0,0]


if __name__ == '__main__':
    unittest.main()
