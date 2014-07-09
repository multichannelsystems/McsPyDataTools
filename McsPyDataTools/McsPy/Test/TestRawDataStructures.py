import unittest
import McsData
import datetime
import exceptions
import numpy as np

from McsPy import *

test_raw_frame_data_file_path = ".\\TestData\\Sensors-10x100ms-10kHz.h5"

test_data_file_path = ".\\TestData\\2014-07-09T10-17-35W8 Standard all 500 Hz.h5"

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
        self.assertEqual(self.data.mcs_hdf5_protocol_type, McsHdf5Protocols.RAW_DATA[0], 
                         "The MCS-HDF5 protocol type was '%s' and not '%s' as expected!" % (self.data.mcs_hdf5_protocol_type, McsHdf5Protocols.RAW_DATA[0]))
        self.assertEqual(self.data.mcs_hdf5_protocol_type_version, 1, 
                         "The MCS-HDF5 protocol version was '%s' and not '1' as expected!" % self.data.mcs_hdf5_protocol_type_version)

    def test_mcs_hdf5_version_frame(self):
        self.assertEqual(self.data.mcs_hdf5_protocol_type, McsHdf5Protocols.RAW_DATA[0], 
                         "The MCS-HDF5 protocol type was '%s' and not '%s' as expected!" % (self.data.mcs_hdf5_protocol_type, McsHdf5Protocols.RAW_DATA[0]))
        self.assertEqual(self.data.mcs_hdf5_protocol_type_version, 1, 
                         "The MCS-HDF5 protocol version was '%s' and not '1' as expected!" % self.data.mcs_hdf5_protocol_type_version)

    # Test session:
    def test_session_attributes(self):
        self.assertEqual(self.data.comment, '', 'Comment is different!')
        self.assertEqual(self.data.clr_date, 'Mittwoch, 9. Juli 2014', 'Clr-Date is different!')
        self.assertEqual(self.data.date_in_clr_ticks, 635404978551720981, 'Clr-Date-Ticks are different!')
        self.assertEqual(self.data.date, datetime.datetime(2014,7,9), 'Date is different!');
        self.assertEqual(str(self.data.file_guid), '700b3ec2-d406-4943-bcef-79d73f0ac4d3', 'FileGUID is different!')
        self.assertEqual(str(self.data.mea_layout), 'Linear8', 'Mea-Layout is different!')
        self.assertEqual(self.data.mea_sn, '', 'MeaSN is different!')
        self.assertEqual(self.data.mea_name, 'Linear8', 'MeaName is different!')
        self.assertEqual(self.data.program_name, 'Multi Channel Experimenter', 'Program name is different!')
        self.assertEqual(self.data.program_version, '0.9.8.2', 'Program version is different!') 


    # Test recording:
    def test_count_recordings(self):
        self.assertEqual(len(self.data.recordings), 1, 'There should be only one recording!')

    def test_recording_attributes(self):
        first_recording = self.data.recordings[0]
        self.assertEqual(first_recording.comment, '', 'Recording comment is different!')
        self.assertEqual(first_recording.duration, 19700000, 'Recording duration is different!')
        self.assertEqual(first_recording.label, '', 'Recording label is different!')
        self.assertEqual(first_recording.recording_id, 0, 'Recording ID is different!')
        self.assertEqual(first_recording.recording_type, '', 'Recording type is different!')
        self.assertEqual(first_recording.timestamp, 0, 'Recording timestamp is different!')
        self.assertAlmostEqual(first_recording.duration_time.to(ureg.sec).magnitude, 19.7, places = 1, msg = 'Recording timestamp is different!')

    # Test analog streams:
    def test_count_analog_streams(self):
         self.assertEqual(len(self.data.recordings[0].analog_streams), 3, 'There should be only one analog stream inside the recording!')

    def test_analog_stream_attributes(self):
        first_analog_stream = self.data.recordings[0].analog_streams[0]
        self.assertEqual(first_analog_stream.data_subtype, 'Electrode', 'Analog stream data sub type is different!')
        self.assertEqual(first_analog_stream.label, 'Filter (1) Filter Data', 'Analog stream label is different!')
        self.assertEqual(str(first_analog_stream.source_stream_guid), '43f795b0-7881-408f-a840-0207bc8e203c', 'Analog stream source GUID is different!')
        self.assertEqual(str(first_analog_stream.stream_guid), 'a9d1ab04-2cf8-489c-a861-595e662fba4e', 'Analog stream GUID is different!')
        self.assertEqual(first_analog_stream.stream_type, 'Analog', 'Analog stream type is different!')
    
    def test_analog_stream(self):
        data_set = self.data.recordings[0].analog_streams[0].channel_data
        self.assertEqual(data_set.shape, (8, 9850), 'Shape of dataset is different!')
        
        timestamp_index = self.data.recordings[0].analog_streams[0].timestamp_index
        self.assertEqual(timestamp_index.shape, (1, 3), 'Shape of timestamp index is different!')

        channel_infos =  self.data.recordings[0].analog_streams[0].channel_infos
        self.assertEqual(len(channel_infos), 8, 'Number of channel info objects is different!')
        self.assertEqual(len(channel_infos[0].info), 16, 'Number of of components of an channel info object is different!')

    def test_analog_stream_data(self):
        analog_stream =  self.data.recordings[0].analog_streams[0]
        signal = analog_stream.get_channel_in_range(0, 1569, 1584)
        sig = signal[0]
        scale = 381469 * 10**-9
        expected_sig = np.array([4, 5, 0, -3, 2, -1, -6, 6, 0, 0, 0, 0, 0, 0, 3, -9], dtype=np.float) * scale
        np.testing.assert_almost_equal(sig, expected_sig, decimal = 5)
        #self.assertEquals(map(lambda x: x, sig), expected_sig, "Signal values were '%s' and not as expected '%s'" % (sig, expected_sig))
        self.assertEqual(str(signal[1]), 'volt', "Unit of sampled values was expected to be 'volt' but was '%s'!" % str(signal[1]))

    def test_analog_stream_data_timestamps(self):
        analog_stream =  self.data.recordings[0].analog_streams[0]
        signal_ts = analog_stream.get_channel_sample_timestamps(6, 1996, 2000)
        sig_ts = signal_ts[0]
        expected_ts = [3992000, 3994000, 3996000, 3998000, 4000000]
        self.assertEquals(map(lambda x: x, sig_ts), expected_ts, "Selected timestamps were '%s' and not as expected '%s'" % (sig_ts, expected_ts))
        self.assertEqual(str(signal_ts[1]), 'microsecond', "Unit of timestamps was expected to be 'microsecond' but was '%s'!" % str(signal_ts[1]))

    # Test frame streams:
    def test_count_frame_streams(self):
         self.assertEqual(len(self.raw_frame_data.recordings[0].frame_streams), 1, 'There should be only one frame stream!')
         self.assertEqual(len(self.raw_frame_data.recordings[0].frame_streams[0].frame_entity), 1, 'There should be only one frame entity inside the stream!')

    def test_frame_stream_attributes(self):
        first_frame_stream = self.raw_frame_data.recordings[0].frame_streams[0]
        self.assertEqual(first_frame_stream.data_subtype, 'Unknown', 'Frame stream data sub type is different!')
        self.assertEqual(first_frame_stream.label, '', 'Frame stream label is different!')
        self.assertEqual(str(first_frame_stream.source_stream_guid), '11bee63c-8714-4b2b-8cf9-228b1915f183', 'Frame stream source GUID is different!')
        self.assertEqual(str(first_frame_stream.stream_guid), '784bf2ba-0e1b-4f3a-acc6-825af9bd1bf1', 'Frame stream GUID is different!')
        self.assertEqual(first_frame_stream.stream_type, 'Frame', 'Frame stream type is different!')
    
    def test_frame_infos(self):
        conv_fact_expected = np.zeros(shape=(65,65), dtype=np.int32) + 1000
        info_expected = {
                     'FrameLeft': 1, 'Exponent': -9, 'RawDataType': 'Short', 'LowPassFilterCutOffFrequency': '-1', 'Label': 'ROI 1', 
                     'FrameTop': 1, 'ADZero': 0, 'LowPassFilterOrder': -1, 'ReferenceFrameTop': 1, 'FrameRight': 65, 'HighPassFilterType': '', 
                     'Tick': 50, 'SensorSpacing': 1, 'HighPassFilterCutOffFrequency': '-1', 'FrameDataID': 0, 'FrameID': 1, 'GroupID': 1, 
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
        self.assertEqual(frame.shape, (65,65), "Second slice was '%s' and not '(65,65)' as expected!" % str(frame.shape))
        selected_values = [frame[0,0], frame[9,3], frame[0,5]]
        expected_values = [    -10000,        211,       -727]
        self.assertEquals(selected_values, expected_values, "Selected ADC values were '%s' and not as expected '%s'" % (selected_values, expected_values))
        sensor_signal = frame_entity.get_sensor_signal(30, 30, 0, 1000)
        sig = sensor_signal[0]
        self.assertEquals(len(sig), 1001, "Length of sensor signal was '%s' and not as expected '1001'" % len(sig))

    def test_frame_data_timestamps(self):
        frame_entity =  self.raw_frame_data.recordings[0].frame_streams[0].frame_entity[1]
        timestamps = frame_entity.get_frame_timestamps(0,2000)
        ts = timestamps[0]
        self.assertEqual(len(ts), 2001, "Number oftime stamps were '%s' and not as expected '2001'" % len(ts))
        timestamps = frame_entity.get_frame_timestamps(1995,2005)
        ts = timestamps[0]
        self.assertEqual(len(ts), 11, "Number of timestamps were '%s' and not as expected '11'" % len(ts))
        expected_ts = [199750, 199800, 199850, 199900, 199950, 200000,  1000000,  1000050,  1000100, 1000150, 1000200]
        self.assertEquals(map(lambda x: x, ts), expected_ts, "Timestamps were '%s' and not as expected '%s'" % (ts, expected_ts))
        timestamps = frame_entity.get_frame_timestamps(0,5000)
        ts = timestamps[0]
        self.assertEqual(len(ts), 5001, "Number of timestamps were '%s' and not as expected '5001'" % len(ts))
        selected_ts = [ ts[0], ts[1], ts[2000], ts[2001], ts[2002], ts[4001], ts[4002], ts[4003]]
        expected_ts = [100000,100050,   200000,  1000000,  1000050,  1100000,  3000000,  3000050]
        self.assertEquals(selected_ts, expected_ts, "Selected timestamps were '%s' and not as expected '%s'" % (selected_ts, expected_ts))
        timestamps = frame_entity.get_frame_timestamps(16008,16008)
        ts = timestamps[0]
        self.assertEqual(len(ts), 1, "Number of timestamps were '%s' and not as expected '1'" % len(ts))
        self.assertEquals(ts[0], 12500000, "Timestamps were '%s' and not as expected '%s'" % (ts, expected_ts))
        self.assertEqual(str(timestamps[1]), 'microsecond', "Unit of timestamps was expected to be 'microsecond' but was '%s'!" % str(timestamps[1]))

    # Test event streams:
    def test_count_event_streams(self):
        self.assertEqual(len(self.data.recordings[0].event_streams), 1, 'There should be only one event stream inside the recording!')
        self.assertEqual(len(self.data.recordings[0].event_streams[0].event_entity), 1, 'There should be 1 event entities inside the stream!')

    def test_event_stream_attributes(self):
        first_event_stream = self.data.recordings[0].event_streams[0]
        self.assertEqual(first_event_stream.data_subtype, 'DigitalPort', 'Event stream data sub type is different from expected \'DigitalPort\'!')
        self.assertEqual(first_event_stream.label, 'Digital Events 1', 'Event stream label is different!')
        self.assertEqual(str(first_event_stream.source_stream_guid), '0696bca6-7c30-4024-8e58-72da383aa248', 'Event stream source GUID is different!')
        self.assertEqual(str(first_event_stream.stream_guid), '92bc437b-7655-4673-adfa-abbeca2c53e0', 'Event stream GUID is different!')
        self.assertEqual(first_event_stream.stream_type, 'Event', 'Event stream type is different!')

    def test_event_infos(self):
        first_event_entity = self.data.recordings[0].event_streams[0].event_entity[0]
        self.assertEqual(first_event_entity.info.id, 0, "ID is not as expected!")
        self.assertEqual(first_event_entity.info.raw_data_bytes, 4, "RawDataBytes is not as expected!")
        self.assertEquals(first_event_entity.info.source_channel_ids, [8],"Source channel IDs are different!") 
        self.assertEquals(first_event_entity.info.source_channel_labels.values(), 
                          ["1"],"Source channels label is different (was '%s' instead of '['1']')!" % 
                          first_event_entity.info.source_channel_labels.values()) 

    def test_event_data(self):
        first_event_entity = self.data.recordings[0].event_streams[0].event_entity[0]
        self.assertEqual(first_event_entity.count, 12, "Count was expected to be 12 but was %s!" % first_event_entity.count)
        events = first_event_entity.get_events()
        self.assertEqual(str(events[1]), 'microsecond', "Event time unit was expected to be 'microsecond' but was '%s'!" % str(events[1]))
        self.assertEqual((events[0]).shape, (2,12), "Event structured was expected to be (2,12) but was %s!" % str(events[0].shape))
        events_ts = first_event_entity.get_event_timestamps(0,3)
        #self.assertAlmostEquals(events[0],[1.204050, 2.099150, 2.106800] , places = 5, msg = "Event timestamps were not as expected!")
        np.testing.assert_almost_equal(events_ts[0],[216000, 1916000, 3616000], decimal = 5)
        events_ts = first_event_entity.get_event_timestamps(4,5)
        self.assertAlmostEqual(events_ts[0][0], 7016000, places = 4, msg = "Last event timestamp was %s and not as expected 216000!" % events[0][0])
        events_duration = first_event_entity.get_event_durations(2,8)
        np.testing.assert_almost_equal(events_duration[0],[0, 0, 0, 0, 0, 0], decimal = 5)
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
        self.assertEqual(first_segment_stream.label, 'Spike Detector (1) Spike Data', "Segment label was '%s' and not '' as expected!" % first_segment_stream.label)
        self.assertEqual(str(first_segment_stream.source_stream_guid), 'a9d1ab04-2cf8-489c-a861-595e662fba4e', 
                         "Segment stream source GUID was '%s' and not 'a9d1ab04-2cf8-489c-a861-595e662fba4e' as expected!" % str(first_segment_stream.source_stream_guid))
        self.assertEqual(str(first_segment_stream.stream_guid), '7c2105e5-5ea4-4fdc-91d8-6b85f47773c2', 
                         "Segment stream GUID was '%s' and not '7c2105e5-5ea4-4fdc-91d8-6b85f47773c2' as expected!" % str(first_segment_stream.stream_guid))

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
        self.assertEqual(first_segment_entity.segment_sample_count, 26, "Segment sample count was expected to be  but was %s!" % first_segment_entity.segment_sample_count)
        signal = first_segment_entity.get_segment_in_range(0)
        self.assertEqual(signal[0].shape, (2, 26), "Matrix of segment signal points was expected to be '(2,26)' but was '%s'!" % str(signal[0].shape))
        self.assertEqual(str(signal[1]), 'volt', "Unit of segment signal was expected to be 'volt' but was '%s'!" % str(signal[1]))
        signal_flat = first_segment_entity.get_segment_in_range(0, flat = True)
        self.assertEqual(len(signal_flat[0]), 52, "Vector ('flat = True') of segment signal points was expected to be '52' but was '%s'!" % len(signal_flat[0]))
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_in_range, segment_id = 0, flat = False, idx_start = 16, idx_end = 4)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_in_range, segment_id = 0, flat = False, idx_start = 40, idx_end = 49)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_in_range, segment_id = 0, flat = False, idx_start = -1, idx_end = 10)

    def test_segment_data_timestamps(self):
        first_segment_entity = self.data.recordings[0].segment_streams[0].segment_entity[0]
        signal_ts = first_segment_entity.get_segment_sample_timestamps(0)
        self.assertEqual(signal_ts[0].shape, (2, 26), "Matrix of segment timestamps was expected to be '(2,26)' but was '%s'!" % str(signal_ts[0].shape))
        self.assertEqual(str(signal_ts[1]), 'microsecond', "Unit of timestamps was expected to be 'microsecond' but was '%s'!" % str(signal_ts[1]))
        ts_selected = (signal_ts[0][:,0]).tolist()
        expected_ts_first_segment = [943000, 945000]
        self.assertEquals(ts_selected, expected_ts_first_segment, "Timestamps for the first segment were '%s' and not as expected '%s" % (ts_selected, expected_ts_first_segment))
        ts_selected = (signal_ts[0][:,2]).tolist()
        expected_ts_third_segment = [963000, 965000]
        self.assertEquals(ts_selected, expected_ts_third_segment, "Timestamps for the third segment were '%s' and not as expected '%s" % (ts_selected, expected_ts_third_segment))
        signal_flat_ts = first_segment_entity.get_segment_sample_timestamps(0, flat = True)
        self.assertEqual(len(signal_flat_ts[0]), 52, "Vector ('flat = True') of segment signal points was expected to be '52' but was '%s'!" % len(signal_flat_ts[0]))
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_sample_timestamps, segment_id = 0, flat = False, idx_start = 16, idx_end = 4)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_sample_timestamps, segment_id = 0, flat = False, idx_start = 40, idx_end = 49)
        self.assertRaises(exceptions.IndexError, first_segment_entity.get_segment_sample_timestamps, segment_id = 0, flat = False, idx_start = -1, idx_end = 10)

    # Test timestamp streams:
    def test_count_timestamp_streams(self):
        self.assertEqual(len(self.data.recordings[0].timestamp_streams), 1, 'There should be only one timestamp stream inside the recording!')
        self.assertEqual(len(self.data.recordings[0].timestamp_streams[0].timestamp_entity), 8, 
                         'There should be 8 event entities inside the stream (found %s)!' % len(self.data.recordings[0].timestamp_streams[0].timestamp_entity))

    def test_timestamp_stream_attributes(self):
         first_timestamp_stream = self.data.recordings[0].timestamp_streams[0]
         self.assertEqual(first_timestamp_stream.data_subtype, 'NeuralSpike', 'Timestamp stream data sub type is different from expected \'NeuralSpike\'!')
         self.assertEqual(first_timestamp_stream.label, 'Spike Detector (1) Spike Timestamps', 'Timestamp stream label is different!')
         self.assertEqual(str(first_timestamp_stream.source_stream_guid), 'a9d1ab04-2cf8-489c-a861-595e662fba4e', 'Timestamp stream source GUID is different!')
         self.assertEqual(str(first_timestamp_stream.stream_guid), 'b71fc432-be6a-4135-9d15-3c7c1a4b4ed6', 'TimeStamp stream GUID is different!')
         self.assertEqual(first_timestamp_stream.stream_type, 'TimeStamp', 'Timestamp stream type is different!')

    def test_timestamp_infos(self):
         first_timestamp_entity = self.data.recordings[0].timestamp_streams[0].timestamp_entity[0]
         self.assertEqual(first_timestamp_entity.info.id, 0, "ID is not as expected!")
         self.assertEqual(first_timestamp_entity.info.group_id, 0, "Group ID is not as expected!")
         self.assertEqual(first_timestamp_entity.info.data_type, 'Long', "DataType is not as expected!")
         self.assertEqual(first_timestamp_entity.info.unit, 's', "Unit is not as expected (was %s instead of \'s\')!" % first_timestamp_entity.info.unit)
         self.assertEqual(first_timestamp_entity.info.exponent, -6, "Exponent is not as expected (was %s instead of \'-6\')!" % first_timestamp_entity.info.exponent)
         self.assertEqual(first_timestamp_entity.info.measuring_unit, 1 * ureg.us , 
                          "Exponent is not as expected (was %s instead of \'us\')!" % first_timestamp_entity.info.measuring_unit)
         self.assertEquals(first_timestamp_entity.info.source_channel_ids, [0],"Source channel IDs are different!") 
         self.assertEquals(first_timestamp_entity.info.source_channel_labels.values(), 
                           ["E1"],"Source channels label is different (was '%s' instead of '['E1']')!" % 
                           first_timestamp_entity.info.source_channel_labels.values())

    def test_timestamp_data(self):
        first_timestamp_entity = self.data.recordings[0].timestamp_streams[0].timestamp_entity[0]
        self.assertEqual(first_timestamp_entity.count, 26, "Count was expected to be 26 but was %s!" % first_timestamp_entity.count)
        timestamps = first_timestamp_entity.get_timestamps()
        self.assertEqual(timestamps[1], 1 * ureg.us, "Timestamp time unit was expected to be 'us' but was '%s'!" % timestamps[1])
        expected_ts = [944000, 954000, 964000, 3030000, 3040000, 3052000, 
                       3096000, 5104000, 5116000, 5126000, 7204000, 7212000, 
                       7226000, 9290000, 9298000, 11376000, 11386000, 11442000, 
                       13462000, 13472000, 13528000, 15548000, 15558000, 17634000, 
                       17644000, 17686000]
        self.assertEquals(timestamps[0][0].tolist(), expected_ts, "Timestamps of the first TS-Entity were '%s' and not as expected '%s" % (timestamps[0], expected_ts))

if __name__ == '__main__':
    unittest.main()
