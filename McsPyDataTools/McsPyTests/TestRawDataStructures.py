import unittest
import McsData
import datetime
import exceptions
import numpy as np

from McsPy import ureg, Q_

test_raw_frame_data_file_path = ".\\TestData\\Sensors-1x100ms-10kHz.h5"

test_data_file_path = ".\\TestData\\2014-02-27T08-30-03W8SpikeCutoutsAndTimestampsAndRawData.h5"

#@unittest.skip("showing the principle structure of python unit tests")
#class Test_TestRawDataStructures(unittest.TestCase):
#    def test_A(self):
#        self.fail("Not implemented")

class Test_RawData(unittest.TestCase):
    def setUp(self):
        self.data = McsData.RawData(test_data_file_path)
        self.raw_frame_data = McsData.RawData(test_raw_frame_data_file_path)

class Test_RawDataContainer(Test_RawData):
    # Test MCS-HDF5 version
    def test_mcs_hdf5_version(self):
        self.assertEqual(self.data.mcs_hdf5_version, 1, 'The MCS-HDF5-Version is different from the expected one!')

    def test_mcs_hdf5_version_frame(self):
        self.assertEqual(self.raw_frame_data.mcs_hdf5_version, 1, 'The MCS-HDF5-Version is different from the expected one!')

    # Test session:
    def test_session_attributes(self):
        self.assertEqual(self.data.comment, '', 'Comment is different!')
        self.assertEqual(self.data.clr_date, 'Donnerstag, 27. Februar 2014', 'Clr-Date is different!')
        self.assertEqual(self.data.date_in_clr_ticks, 635290866032185769, 'Clr-Date-Ticks are different!')
        self.assertEqual(self.data.date, datetime.datetime(2014,2,27), 'Date is different!');
        self.assertEqual(str(self.data.file_guid), '3be1f837-7374-4600-9f69-c86bcee5ef41', 'FileGUID is different!')
        self.assertEqual(str(self.data.mea_layout), 'Linear8', 'Mea-Layout is different!')
        self.assertEqual(self.data.mea_sn, '', 'MeaSN is different!')
        self.assertEqual(self.data.mea_name, 'Linear8', 'MeaName is different!')
        self.assertEqual(self.data.program_name, 'Multi Channel Experimenter', 'Program name is different!')
        self.assertEqual(self.data.program_version, '0.8.6.0', 'Program version is different!') 


    # Test recording:
    def test_count_recordings(self):
        self.assertEqual(len(self.data.recordings), 1, 'There should be only one recording!')

    def test_recording_attributes(self):
        first_recording = self.data.recordings[0]
        self.assertEqual(first_recording.comment, '', 'Recording comment is different!')
        self.assertEqual(first_recording.duration, 7900000, 'Recording duration is different!')
        self.assertEqual(first_recording.label, '', 'Recording label is different!')
        self.assertEqual(first_recording.recording_id, 0, 'Recording ID is different!')
        self.assertEqual(first_recording.recording_type, '', 'Recording type is different!')
        self.assertEqual(first_recording.timestamp, 0, 'Recording time stamp is different!')
        self.assertAlmostEqual(first_recording.duration_time.to(ureg.sec).magnitude, 7.9, places = 1, msg = 'Recording time stamp is different!')

    # Test analog streams:
    def test_count_analog_streams(self):
         self.assertEqual(len(self.data.recordings[0].analog_streams), 1, 'There should be only one analog stream inside the recording!')

    def test_analog_stream_attributes(self):
        first_analog_stream = self.data.recordings[0].analog_streams[0]
        self.assertEqual(first_analog_stream.data_subtype, 'Electrode', 'Analog stream data sub type is different!')
        self.assertEqual(first_analog_stream.label, '', 'Analog stream label is different!')
        self.assertEqual(str(first_analog_stream.source_stream_guid), '00000000-0000-0000-0000-000000000000', 'Analog stream source GUID is different!')
        self.assertEqual(str(first_analog_stream.stream_guid), '3a1054d5-2c9f-4ddf-877b-282b86c1d5ab', 'Analog stream GUID is different!')
        self.assertEqual(first_analog_stream.stream_type, 'Analog', 'Analog stream type is different!')
    
    def test_analog_stream_data(self):
        data_set = self.data.recordings[0]. analog_streams[0].channel_data
        self.assertEqual(data_set.shape, (8, 158024), 'Shape of dataset is different!')
        
        time_stamp_index = self.data.recordings[0]. analog_streams[0].time_stamp_index
        self.assertEqual(time_stamp_index.shape, (1, 3), 'Shape of time stamp index is different!')

        channel_infos =  self.data.recordings[0]. analog_streams[0].channel_infos
        self.assertEqual(len(channel_infos), 8, 'Number of channel info objects is different!')
        self.assertEqual(len(channel_infos[0].info), 16, 'Number of of components of an channel info object is different!')

    # Test frame streams:
    def test_count_frame_streams(self):
         self.assertEqual(len(self.raw_frame_data.recordings[0].frame_streams), 1, 'There should be only one frame stream!')
         self.assertEqual(len(self.raw_frame_data.recordings[0].frame_streams[0].frame_entity), 1, 'There should be only one frame entity inside the stream!')

    def test_frame_stream_attributes(self):
        first_frame_stream = self.raw_frame_data.recordings[0].frame_streams[0]
        self.assertEqual(first_frame_stream.data_subtype, 'Unknown', 'Frame stream data sub type is different!')
        self.assertEqual(first_frame_stream.label, '', 'Frame stream label is different!')
        self.assertEqual(str(first_frame_stream.source_stream_guid), 'a7559975-5bf6-4252-b3ec-1557e97dca41', 'Frame stream source GUID is different!')
        self.assertEqual(str(first_frame_stream.stream_guid), '7627058d-b597-45b2-86b6-57819d38756e', 'Frame stream GUID is different!')
        self.assertEqual(first_frame_stream.stream_type, 'Frame', 'Frame stream type is different!')
    
    #def test_frame_entity(self):
    #    frame_entity =  self.raw_frame_data.recordings[0].frame_streams[0].frame_entity[1]

    def test_frame_infos(self):
        conv_fact_expected = np.zeros(shape=(65,65), dtype=np.int32) + 1000
        info_expected = {
                     'FrameLeft': 1, 'Exponent': -9, 'RawDataType': 'Short', 'LowPassFilterCutOffFrequency': '-1', 'Label': 'ROI 1', 
                     'FrameTop': 1, 'ADZero': 0, 'LowPassFilterOrder': -1, 'ReferenceFrameTop': 1, 'FrameRight': 65, 'HighPassFilterType': '', 
                     'Tick': 100, 'SensorSpacing': 1, 'HighPassFilterCutOffFrequency': '-1', 'FrameDataID': 0, 'FrameID': 1, 'GroupID': 1, 
                     'ReferenceFrameRight': 65, 'ReferenceFrameBottom': 65, 'LowPassFilterType': '', 'HighPassFilterOrder': -1, 
                     'ReferenceFrameLeft': 1, 'FrameBottom': 65, 'Unit': 'V'
        }
        frame_info =  self.raw_frame_data.recordings[0].frame_streams[0].frame_entity[1].info
        self.assertEqual(len(frame_info.info), 24, 'Number of of components of an channel info object is different!')
        info_key_diff = set(frame_info.info.keys()) - set(info_expected.keys())
        if not info_key_diff:
            for key, value in frame_info.info.items():
                self.assertEqual(
                    value, info_expected[key], 
                    "Frame info object for key '%(k)s' is ('%(val)s') not as expected ('%(ex_val)s')!" % {'k':key, 'val':value, 'ex_val':info_expected[key]}
                )
        self.assertEqual(frame_info.frame.height, 65, "Frame height was '%s' and not '65' as expected!" % frame_info.frame.height)
        self.assertEqual(frame_info.frame.width, 65, "Frame width was '%s' and not '65' as expected!" % frame_info.frame.width)
        self.assertEqual(frame_info.reference_frame.height, 65, "Frame height was '%s' and not '65' as expected!" % frame_info.reference_frame.height)
        self.assertEqual(frame_info.reference_frame.width, 65, "Frame width was '%s' and not '65' as expected!" % frame_info.reference_frame.width)
        self.assertEqual(frame_info.adc_basic_step.magnitude, 10**-9, "ADC step was '%s' and not '10^-9 V' as expected!" % frame_info.adc_basic_step)
        self.assertEqual(frame_info.adc_step_for_sensor(0,0).magnitude, 1000 * 10**-9, "ADC step was '%s' and not '1000 * 10^-9 V' as expected!" % frame_info.adc_step_for_sensor(0,0))
        self.assertEqual(frame_info.adc_step_for_sensor(1,1).magnitude, 1000 * 10**-9, "ADC step was '%s' and not '1000 * 10^-9 V' as expected!" % frame_info.adc_step_for_sensor(1,1))
        self.assertEqual(frame_info.adc_step_for_sensor(63,63).magnitude, 1000 * 10**-9, "ADC step was '%s' and not '1000 * 10^-9 V' as expected!" % frame_info.adc_step_for_sensor(63,63))
        self.assertTrue((frame_info.conversion_factors == conv_fact_expected).all(), "Frame sensor conversion factors matrix is different from the expected one!")
        self.assertRaises(exceptions.IndexError, frame_info.adc_step_for_sensor, 65,65)          

    def test_frame_data(self):
        frame_entity =  self.raw_frame_data.recordings[0].frame_streams[0].frame_entity[1]
        frame_data = frame_entity.data
        frame = frame_data[:,:,1]
        time_stamps = frame_entity.get_frame_timestamps(0,1000)
        sensor_signal = frame_entity.get_sensor_signal(30, 30, 0, 1000)


    # Test event streams:
    def test_count_event_streams(self):
        self.assertEqual(len(self.data.recordings[0].event_streams), 1, 'There should be only one event stream inside the recording!')
        self.assertEqual(len(self.data.recordings[0].event_streams[0].event_entity), 8, 'There should be 8 event entities inside the stream!')

    def test_event_stream_attributes(self):
        first_event_stream = self.data.recordings[0].event_streams[0]
        self.assertEqual(first_event_stream.data_subtype, 'SpikeTimeStamp', 'Event stream data sub type is different from expected \'SpikeTimeStamp\'!')
        self.assertEqual(first_event_stream.label, '', 'Event stream label is different!')
        self.assertEqual(str(first_event_stream.source_stream_guid), '3a1054d5-2c9f-4ddf-877b-282b86c1d5ab', 'Event stream source GUID is different!')
        self.assertEqual(str(first_event_stream.stream_guid), '5a12d97b-f119-4ed6-aab7-5ab57a6f9f41', 'Event stream GUID is different!')
        self.assertEqual(first_event_stream.stream_type, 'Event', 'Event stream type is different!')

    def test_event_infos(self):
        first_event_entity = self.data.recordings[0].event_streams[0].event_entity[0]
        self.assertEqual(first_event_entity.info.id, 0, "ID is not as expected!")
        self.assertEqual(first_event_entity.info.raw_data_bytes, 4, "RawDataBytes is not as expected!")
        self.assertEquals(first_event_entity.info.source_channel_ids, [0],"Source channel IDs are different!") 
        self.assertEquals(first_event_entity.info.source_channel_labels.values(), 
                          ["E1"],"Source channels label is different (was '%s' instead of '['E1']')!" % 
                          first_event_entity.info.source_channel_labels.values()) 

    def test_event_data(self):
        first_event_entity = self.data.recordings[0].event_streams[0].event_entity[0]
        self.assertEqual(first_event_entity.count, 36, "Count was expected to be 36 but was %s!" % first_event_entity.count)
        events = first_event_entity.get_events()
        self.assertEqual(str(events[1]), 'second', "Event time unit was expected to be 'second' but was '%s'!" % str(events[1]))
        self.assertEqual((events[0]).shape, (2,36), "Event structured was expected to be (2,36) but was %s!" % str(events[0].shape))
        events_ts = first_event_entity.get_event_timestamps(0,3)
        #self.assertAlmostEquals(events[0],[1.204050, 2.099150, 2.106800] , places = 5, msg = "Event time stamps were not as expected!")
        np.testing.assert_almost_equal(events_ts[0],[1.204050, 2.099150, 2.106800], decimal = 5)
        events_ts = first_event_entity.get_event_timestamps(35,36)
        self.assertAlmostEqual(events_ts[0][0], 7.491100, places = 4, msg = "Last event time stamp was %s and not as expected 7.4911!" % events[0][0])
        events_duration = first_event_entity.get_event_durations(15,22)
        np.testing.assert_almost_equal(events_duration[0],[0, 0, 0, 0, 0, 0, 0], decimal = 5)
        self.assertRaises(exceptions.IndexError, first_event_entity.get_events, 16, 4)
        self.assertRaises(exceptions.IndexError, first_event_entity.get_events, 412, 500)   
        self.assertRaises(exceptions.IndexError, first_event_entity.get_events, -1, 5) 

    # Test segment streams:
    def test_count_segment_streams(self):
        self.assertEqual(len(self.data.recordings[0].segment_streams), 1, 'There should be only one segment stream inside the recording!')
        self.assertEqual(len(self.data.recordings[0].segment_streams[0].segment_entity), 8, 'There should be 8 segment entities inside the stream!')

    def test_segment_stream_attributes(self):
        first_segment_stream = self.data.recordings[0].segment_streams[0]
        self.assertEqual(first_segment_stream.stream_type, 'Segment', "Segment stream type was '%s' and not 'Segment'!" % first_segment_stream.stream_type)
        self.assertEqual(first_segment_stream.data_subtype, 'Spike', "Segment stream data sub type was '%s' and not 'Spike' as expected!" % first_segment_stream.data_subtype)
        self.assertEqual(first_segment_stream.label, '', "Segment label was '%s' and not '' as expected!" % first_segment_stream.label)
        self.assertEqual(str(first_segment_stream.source_stream_guid), '3a1054d5-2c9f-4ddf-877b-282b86c1d5ab', 
                         "Segment stream source GUID was '%s' and not '3a1054d5-2c9f-4ddf-877b-282b86c1d5ab' as expected!" % str(first_segment_stream.source_stream_guid))
        self.assertEqual(str(first_segment_stream.stream_guid), '45a5873f-963a-4a48-a18e-2a9b0ff69005', 
                         "Segment stream GUID was '%s' and not '45a5873f-963a-4a48-a18e-2a9b0ff69005' as expected!" % str(first_segment_stream.stream_guid))

    def test_segment_infos(self):
        fifth_segment_entity = self.data.recordings[0].segment_streams[0].segment_entity[4]
        self.assertEqual(fifth_segment_entity.info.id, 4, "ID was '%s' and not '4' as expected!" % fifth_segment_entity.info.id)
        self.assertEqual(fifth_segment_entity.info.group_id, 0, "Group ID was '%s' and not '0' as expected!" % fifth_segment_entity.info.group_id)
        self.assertEqual(fifth_segment_entity.info.pre_interval.magnitude, 1000, "Pre-Interval was '%s' and not '1000' as expected!" % fifth_segment_entity.info.pre_interval.magnitude)
        self.assertEqual(str(fifth_segment_entity.info.pre_interval.units), 'microsecond', "Pre-Interval unit was '%s' and not 'microsecond' as expected!" % str(fifth_segment_entity.info.pre_interval.units))
        self.assertEqual(fifth_segment_entity.info.post_interval.magnitude, 2000, "Post-Interval was '%s' and not '2000' as expected!" % fifth_segment_entity.info.post_interval.magnitude)
        self.assertEqual(str(fifth_segment_entity.info.post_interval.units), 'microsecond', "Post-Interval unit was '%s' and not 'microsecond' as expected!" % str(fifth_segment_entity.info.post_interval.units))
        self.assertEqual(fifth_segment_entity.info.type, 'Cutout', "Type was '%s' and not 'Cutout' as expected!" % fifth_segment_entity.info.type)
        self.assertEqual(fifth_segment_entity.info.count, 1, "Count of segments was '%s' and not '1' as expected!" % fifth_segment_entity.info.count)
        self.assertEquals(fifth_segment_entity.info.source_channel_of_segment.keys(), [0], 
                          "Source channel dataset index was different (was '%s' instead of '['0']')!" % fifth_segment_entity.info.source_channel_of_segment.keys()) 
        self.assertEquals(fifth_segment_entity.info.source_channel_of_segment[0].channel_id, 4, 
                          "Source channel ID was different (was '%s' instead of '4')!" % fifth_segment_entity.info.source_channel_of_segment[0].channel_id) 

    def test_segment_data(self):
        first_segment_entity = self.data.recordings[0].segment_streams[0].segment_entity[0]
        self.assertEqual(first_segment_entity.segment_sample_count, 36, "Segment sample count was expected to be  but was %s!" % first_segment_entity.segment_sample_count)
        signal = first_segment_entity.get_segment_in_range(0)
        self.assertEqual(signal[0].shape, (61, 36), "Matrix of segment signal points was expected to be '(61,36)' but was '%s'!" % str(signal[0].shape))
        self.assertEqual(str(signal[1]), 'volt', "Unit of segment signal was expected to be 'volt' but was '%s'!" % str(signal[1]))
        signal_flat = first_segment_entity.get_segment_in_range(0, flat = True)
        self.assertEqual(len(signal_flat[0]), 2196, "Vector ('flat = True') of segment signal points was expected to be '2196' but was '%s'!" % len(signal_flat[0]))
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_in_range, segment_id = 0, flat = False, idx_start = 16, idx_end = 4)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_in_range, segment_id = 0, flat = False, idx_start = 40, idx_end = 49)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_in_range, segment_id = 0, flat = False, idx_start = -1, idx_end = 10)

    def test_segment_data_timestamps(self):
        first_segment_entity = self.data.recordings[0].segment_streams[0].segment_entity[0]
        signal_ts = first_segment_entity.get_segment_sample_timestamps(0)
        self.assertEqual(signal_ts[0].shape, (61, 36), "Matrix of segment time stamps was expected to be '(61,36)' but was '%s'!" % str(signal_ts[0].shape))
        self.assertEqual(str(signal_ts[1]), 'microsecond', "Unit of time stamps was expected to be 'second' but was '%s'!" % str(signal_ts[1]))
        ts_selected = [signal_ts[0][0,0], signal_ts[0][1,0], signal_ts[0][2,0], signal_ts[0][0,1], signal_ts[0][1,1], signal_ts[0][2,1]]
        expected_ts = [1204050, 1204100, 1204150, 2099150, 2099200, 2099250]
        self.assertEquals(ts_selected, expected_ts, "Time stamps were '%s' and not as expected '%s" % (ts_selected, expected_ts))
        signal_flat_ts = first_segment_entity.get_segment_sample_timestamps(0, flat = True)
        self.assertEqual(len(signal_flat_ts[0]), 2196, "Vector ('flat = True') of segment signal points was expected to be '2196' but was '%s'!" % len(signal_flat_ts[0]))
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_sample_timestamps, segment_id = 0, flat = False, idx_start = 16, idx_end = 4)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_sample_timestamps, segment_id = 0, flat = False, idx_start = 40, idx_end = 49)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_sample_timestamps, segment_id = 0, flat = False, idx_start = -1, idx_end = 10)

if __name__ == '__main__':
    unittest.main()
