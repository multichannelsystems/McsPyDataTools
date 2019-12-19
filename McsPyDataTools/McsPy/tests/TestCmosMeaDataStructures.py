import unittest
import McsPy.McsCMOSMEA
import datetime
from builtins import IndexError
#import sys # exceptions
import os
import numpy as np

from McsPy import *

test_data_file_path             = os.path.join(os.path.dirname(__file__), 'TestData', 'CMOS_MEA_Results.cmtr')
test_rawdata_file_path          = os.path.join(os.path.dirname(__file__), 'TestData', '2017.12.14-18.38.15-GFP8ms_470nm_100pc_10rep_nofilter.cmcr')
test_multiple_roi_file_path     = os.path.join(os.path.dirname(__file__), 'TestData', 'Rec_1-2-3_1s.cmcr')
test_multiple_streams_file_path = os.path.join(os.path.dirname(__file__), 'TestData', 'V200-SensorRoi-3Aux-Dig-Stim2-DiginEvts-5kHz.cmcr')
test_spike_stream_file_path     = os.path.join(os.path.dirname(__file__), 'TestData', 'ReRec-2017.03.15-15.36.07-chip1617-3s-Spikes.cmcr')

class Test_CmosData(unittest.TestCase):
    def setUp(self):
        self.data                   = McsCMOSMEA.McsData(test_data_file_path)

        self.rawdata                = McsCMOSMEA.McsData(test_rawdata_file_path)

        self.multiple_roi           = McsCMOSMEA.McsData(test_multiple_roi_file_path)

        self.multiple_stream_data   = McsCMOSMEA.McsData(test_multiple_streams_file_path)

        self.spike_stream_data      = McsCMOSMEA.McsData(test_spike_stream_file_path)
    
    def test_CmosDataWrapper(self):
        pass

class Test_CmosDataContainer(Test_CmosData):
    # Test functionality with RawData file
    def test_with_rawdata_file(self):

        # Test MCS-HDF5 version
        def test_mcs_hdf5_version(self):
            self.assertEqual(self.rawdata.mcs_hdf5_protocol_type, "CMOS_MEA", 
                             "The MCS-HDF5 protocol type was '%s' and not '%s' as expected!" % (self.data.mcs_hdf5_protocol_type, "CMOS_MEA"))
            self.assertEqual(self.rawdata.mcs_hdf5_protocol_type_version, 1, 
                             "The MCS-HDF5 protocol version was '%s' and not '1' as expected!" % self.data.mcs_hdf5_protocol_type_version)
        
        test_mcs_hdf5_version(self)


        # Test Acquisition
        def test_acquisition(self):
            self.acquisition_data   = self.rawdata.Acquisition
            print(self.acquisition_data)
            #Test Analog streams
            self.digital_data       = self.acquisition_data.Digital_Data
            self.channel_meta       = self.digital_data.ChannelMeta

        test_acquisition(self)

        # Test Filter Tool, STA Explorer, Spike Explorer and Spike --> should throw exception
        def test_processed_data_groups(self):
            self.assertRaises(AttributeError, self.rawdata.__getattr__,'filter_tool')
            self.assertRaises(AttributeError, self.rawdata.__getattr__, 'sta_explorer')
            self.assertRaises(AttributeError, self.rawdata.__getattr__, 'spike_explorer')
            self.assertRaises(AttributeError, self.rawdata.__getattr__, 'spike_sorter')

        test_processed_data_groups(self)


    # Test functionality with ProcessedData file
    def test_with_processeddata_file(self):

        # Test MCS-HDF5 version
        def test_mcs_hdf5_version(self):
            self.assertEqual(self.data.mcs_hdf5_protocol_type, "CMOS_MEA", 
                             "The MCS-HDF5 protocol type was '%s' and not '%s' as expected!" % (self.data.mcs_hdf5_protocol_type, "CMOS_MEA"))
            self.assertEqual(self.data.mcs_hdf5_protocol_type_version, 1, 
                             "The MCS-HDF5 protocol version was '%s' and not '1' as expected!" % self.data.mcs_hdf5_protocol_type_version)
        
        test_mcs_hdf5_version(self)


        # Test Acquisition
        def test_acquisition(self):
            self.acquisition_data   = self.rawdata.Acquisition
            #Test Digital Data
            self.digital_data       = self.acquisition_data.Digital_Data
            self.assertEqual(self.digital_data.attributes['ID.Instance'],'Digital Data',
                             "The Digital Data object has attribute ID.Instance set to %s and not %s as expected!" %(self.digital_data.attributes['ID.Instance'],'Digital Data'))
            self.channel_data       = self.digital_data.ChannelData_1
            self.assertEqual(self.channel_data.attributes['ID.Type'],'ChannelData',
                             "The channel data has attribute ID.Type set to %s and not %s as expected!" %(self.channel_data.attributes['ID.Type'],'ChannelData'))
            self.assertEqual(self.channel_data[0][0], 1,
                             "The channel data at channel_data[0][0] is %d and not %d as ecpected" % (self.channel_data[0][0], 1))
            self.channel_meta       = self.digital_data.ChannelMeta
            self.assertEqual(self.channel_meta.attributes['ID.Type'],'ChannelMeta',
                             "The ChannelMeta object has attribute ID.Type set to %s and not %s as expected!" %(self.channel_meta.attributes['ID.Type'],'ChannelMeta'))
            self.assertEqual(self.channel_meta.ChannelID[0], 1,
                             "The channel meta at channel_meta.ChannelID is %d and not %d as ecpected" % (self.channel_meta.ChannelID, 1))
            #Test Event Data
            self.event_data         = self.acquisition_data.EventTool_at_Digital_Data
            self.assertEqual(self.event_data.attributes['ID.Instance'],'EventTool @ Digital Data',
                             "The EventToolAtDigitalData object has attribute ID.Instance set to %s and not %s as expected!" %(self.event_data.attributes['ID.Instance'],'EventTool @ Digital Data'))
            self.event_data_data    = self.event_data.EventData
            #self.assertEqual(self.event_data_data.attributes['ID.Instance'],'EventTool @ Digital Data',
            #                 "The EventData has attribute ID.Instance set to %s and not %s as expected!" %(self.event_data_data.attributes['ID.Instance'],'EventTool @ Digital Data'))
            #self.assertEqualself.event_data_data.[0][0], 1,
            #                 "The channel data at channel_data[0][0] is %d and not %d as ecpected" % (self.channel_data[0][0], 1))
            self.event_data_meta  = self.event_data.EventMeta
            #self.assertEqual(self.channel_meta.attributes['ID.Type'],'ChannelMeta',
            #                 "The ChannelMeta object has attribute ID.Type set to %s and not %s as expected!" %(self.channel_meta.attributes['ID.Type'],'ChannelMeta'))
            #self.assertEqual(self.channel_meta.ChannelID, 1,
            #                 "The channel meta at channel_meta.ChannelID is %d and not %d as ecpected" % (self.channel_meta.ChannelID, 1))
            self.event_entity   = self.multiple_stream_data.Acquisition.STG_Sideband_Events.EventEntity
            #Test Sensor Data
            self.sensor_data        = self.acquisition_data.Sensor_Data
            self.sensor_data_data   = self.sensor_data.SensorData_1_1
            self.sweeps             = self.multiple_roi.Acquisition.Sensor_Data.DataChunk[1]
            self.rois               = self.multiple_roi.Acquisition.Sensor_Data.Regions[2]

        test_acquisition(self)


        #Test 
        def test_network_explorer(self):
            network_explorer = self.data.Network_Explorer
            print(network_explorer)
            print(network_explorer.sta_entity[0])
            print(network_explorer.sta_entity[10])

        test_network_explorer(self)

        def test_spike_sorter(self):
            spike_sorter = self.data.Spike_Sorter
            print(spike_sorter)
            units = spike_sorter.get_units_by_id()
            self.assertEqual(len(units), 90, "The number of units is {} and not {} as expected!".format(len(units), 90))
            ordered = spike_sorter.get_units_by_measure(measure='Separability', descending=False)
            self.assertEqual(len(ordered), 90, "The number of ordered units is {} and not {} as expected!".format(len(ordered), 90))
            ordered_sep = [u.get_measure('Separability') for u in ordered]
            s = sorted(ordered_sep)
            self.assertEqual(ordered_sep, s, "The units are not ordered correctly!")
            unit = spike_sorter.get_unit(14)
            print(unit)
            pks = unit.get_peaks()
            for p in pks:
                self.assertEqual(p['IncludePeak'], 1, "IncludePeak is not set!")
            ts = unit.get_peaks_timestamps()
            self.assertEqual(len(pks), len(ts), "The number of timestamps is {} and not {} as expected!".format(len(pks), len(ts)))
            cutouts = unit.get_peaks_cutouts()
            self.assertEqual(len(pks), len(cutouts), "The number of cutouts is {} and not {} as expected!".format(len(pks), len(cutouts)))
            amplitudes = unit.get_peaks_amplitudes()
            self.assertEqual(len(pks), len(amplitudes), "The number of amplitudes is {} and not {} as expected!".format(len(pks), len(amplitudes)))

        test_spike_sorter(self)


     #Test access to streams if multiple streams of one type exists: example multiple analog streams
    def test_analog_streams(self):
        self.acquisition_data = self.multiple_stream_data.Acquisition
        #Test Access to specific analog stream
        self.analog_stream = self.acquisition_data.Analog_Data
        #Test Access to range of streams
        self.analog_streams = self.acquisition_data.ChannelStreams
        self.assertEqual(len(self.analog_streams),3,
                                "The number of analog streams is {} and not {} as expected!".format(len(self.analog_streams), 3))

    #Test Spike Streams
    def test_spike_stream(self):
        self.spike_stream = self.spike_stream_data.Acquisition.SpikeStreams[0]
        self.assertEqual(self.spike_stream.get_spikes_at_sensor(1565).shape,(58,),
                         "The number of retrieved spikes is {} and not {} as expected!".format(self.spike_stream.get_spikes_at_sensor(1565).shape, (58,)))
        self.spike_stream_entity = self.spike_stream.SpikeData
        self.assertEqual(self.spike_stream.get_spikes_in_interval([1.1, 'end']).shape,(11813,),
                         "The number of retrieved spikes is {} and not {} as expected!".format(self.spike_stream.get_spikes_in_interval([1.1, 'end']).shape, (11813,)))

if __name__ == '__main__':
    unittest.main()