import McsPy
import McsPy.McsData
import McsPy.functions_info
import matplotlib
import pylab as pl
import numpy as np

from McsPy import ureg, Q_
#from McsPy.functions_info import print_file_info
import functions_info as fi

def show_image_plot(data, aspect_ratio = 10000):
    #matshow(data)
    #imshow(data, interpolation='nearest', cmap='bone', origin='lower')
    pl.figure(figsize=(20,12))
    pl.imshow(data, interpolation='nearest', aspect=aspect_ratio)
    #colorbar(shrink=.92)
    #xticks([]), yticks([])
    pl.xlabel('Sample Index')
    pl.ylabel('Channel Number')
    pl.title('Heatmap of sampled wireless Signal')
    pl.show()

def draw_raw_data(stream):
    '''Draw all raw data streams'''
    channels = stream.channel_data
    pl.figure(figsize=(20,12))
    pl.plot(np.transpose(channels))
    pl.title('Signal for Wireless (Simulation) / Raw ADC-Values (%s)' % stream.label)
    pl.xlabel('Sample Index')
    pl.ylabel('ADC Value')
    pl.grid()
    pl.show()

def draw_channel_overlay_in_range(stream1, stream2, channel_id):
    ''' Draw an overlay of two streams for one channel of given ID within its original range '''
    time1 = stream1.get_channel_sample_timestamps(channel_id,0,3000)
    signal1 = stream1.get_channel_in_range(channel_id,0,3000)
    time2 = stream2.get_channel_sample_timestamps(channel_id,0,3000)
    signal2 = stream2.get_channel_in_range(channel_id,0,3000)
    pl.figure(figsize=(20,12))
    pl.plot(time1[0], signal1[0])
    pl.plot(time2[0], signal2[0])
    pl.xlabel('Time (%s)' % time1[1])
    pl.ylabel('Voltage (%s)' % signal1[1])
    pl.title('Sampled signal overlay \'%s\' and \'%s\'' % (stream1.label, stream2.label))
    pl.show()

def draw_channel_overlay_in_range_with_events(stream1, stream2, channel_id, timestamps):
    ''' Draw an overlay of two streams for one channel of given ID within its original range '''
    time1 = stream1.get_channel_sample_timestamps(channel_id,0,3000)
    signal1 = stream1.get_channel_in_range(channel_id,0,3000)
    time2 = stream2.get_channel_sample_timestamps(channel_id,0,3000)
    signal2 = stream2.get_channel_in_range(channel_id,0,3000)
    pl.figure(figsize=(20,12))
    pl.plot(time1[0], signal1[0])
    pl.plot(time2[0], signal2[0])
    max_time = max(time1[0][-1],time2[0][-1])
    [pl.axvline(timestamp, color='r') for timestamp in timestamps[0,:] if timestamp < max_time]
    #pl.axvline(1200000)
    pl.xlabel('Time (%s)' % time1[1])
    pl.ylabel('Voltage (%s)' % signal1[1])
    pl.title('Sampled signal overlay \'%s\' and \'%s\'' % (stream1.label, stream2.label))
    pl.show()

def draw_channel_with_spectrogram(stream, channel_id):
    ''' Draw one channel of given ID within its original range '''
    time = stream.get_channel_sample_timestamps(channel_id, 0, 10000)
    # scale time to seconds:
    scale_factor_for_second = Q_(1,time[1]).to(ureg.s).magnitude
    time_in_sec = time[0] * scale_factor_for_second

    signal = stream.get_channel_in_range(channel_id, 0, 10000)
    sampling_frequency = stream.channel_infos[channel_id].sampling_frequency.magnitude # already in Hz
    pl.figure(figsize=(20,12))
    # time domain
    axtp = pl.subplot(211)
    pl.plot(time_in_sec, signal[0])
    pl.xlabel('Time (%s)' % time[1])
    pl.ylabel('Voltage (%s)' % signal[1])
    pl.title('Sampled signal (%s)' % stream.label)
    # frequency domain
    pl.subplot(212)
    pl.specgram(signal[0], NFFT=512, noverlap = 128, Fs = sampling_frequency, cmap = pl.cm.gist_heat, scale_by_freq = False)

    pl.xlabel('Time (%s)' % time[1])
    pl.ylabel('Frequency (Hz)')
    pl.show()

def test_channel_raw_data():
    test_raw_data_file_path = ".\\McsPy\\tests\\TestData\\2014-07-09T10-17-35W8 Standard all 500 Hz.h5"
    fi.print_file_info(test_raw_data_file_path)
    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
    print(raw_data.comment)
    print(raw_data.date)
    print(raw_data.clr_date)
    print(raw_data.date_in_clr_ticks)
    print(raw_data.file_guid)
    print(raw_data.mea_name)
    print(raw_data.mea_sn)
    print(raw_data.mea_layout)
    print(raw_data.program_name)
    print(raw_data.program_version) 
    print(raw_data.recordings)
    print(raw_data.recordings[0].analog_streams)
    # Channel raw data:
    draw_raw_data(raw_data.recordings[0].analog_streams[0])
    show_image_plot(raw_data.recordings[0].analog_streams[1].channel_data[:, 0:10000], 1000)
    draw_channel_overlay_in_range(raw_data.recordings[0].analog_streams[0],
                                  raw_data.recordings[0].analog_streams[1], 
                                  list(raw_data.recordings[0].analog_streams[1].channel_infos.keys())[0])
    draw_channel_with_spectrogram(raw_data.recordings[0].analog_streams[1], 
                                  list(raw_data.recordings[0].analog_streams[1].channel_infos.keys())[0])

def plotImage(arr) :
    fig  = pl.figure(figsize=(8,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    imAx = pl.imshow(arr, origin='lower', interpolation='nearest')
    fig.colorbar(imAx, pad=0.01, fraction=0.1, shrink=1.00, aspect=20)
 
def plotHistogram(arr) :
    fig  = pl.figure(figsize=(8,8), dpi=80, facecolor='w',edgecolor='w',frameon=True)
    pl.hist(arr.flatten(), bins=100)
  
def test_frame_raw_data():
    test_raw_frame_data_file_path = "d:\\Programming\\MCSuite\\McsPyDataTools\\McsPyDataTools\\McsPyTests\\TestData\\Sensor200ms.h5"
    #with McsData.RawData(test_raw_frame_data_file_path) as raw_data:
    raw_data = McsPy.McsData.RawData(test_raw_frame_data_file_path)
    print(raw_data.comment)
    print(raw_data.date)
    print(raw_data.clr_date)
    print(raw_data.date_in_clr_ticks)
    print(raw_data.file_guid)
    print(raw_data.mea_id)
    print(raw_data.mea_name)
    print(raw_data.program_name)
    print(raw_data.program_version) 
    print(raw_data.recordings)
    #print(raw_data.recordings[0])
    first_frame = raw_data.recordings[0].frame_streams[0].frame_entity[1].data[:,:,0]
    plotImage(first_frame)
    plotHistogram(first_frame)
    pl.show()

def test_event_raw_data():
    test_raw_data_file_path = ".\\McsPy\\tests\\TestData\\2014-07-09T10-17-35W8 Standard all 500 Hz.h5"
    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
    event_entity = raw_data.recordings[0].event_streams[0].event_entity[0]
    print("Event entity 0 contains: %s events" % event_entity.count)
    all_events = event_entity.get_events()
    print((all_events[0])[0,:])
    all_event_timestamps = event_entity.get_event_timestamps()
    print(all_event_timestamps[0])
    all_event_durations = event_entity.get_event_durations()
    print(all_event_durations[0])

def test_segment_raw_data():
    test_raw_data_file_path = ".\\McsPy\\tests\\TestData\\2014-07-09T10-17-35W8 Standard all 500 Hz.h5"
    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
    first_segment_entity = raw_data.recordings[0].segment_streams[0].segment_entity[0]
    print("Segment entity 0 contains: %s segments" % first_segment_entity.segment_sample_count)
    signal = first_segment_entity.get_segment_in_range(0, flat = True)
    signal_ts = first_segment_entity.get_segment_sample_timestamps(0, flat = True)
    # convert time stamps to second:
    factor = ureg.convert(1, str(signal_ts[1]), "second")
    signal_ts_second = signal_ts[0] * factor
    pl.figure(figsize=(20,12))
    pl.plot(signal_ts_second, signal[0])
    pl.xlabel('time (s)')
    pl.ylabel('voltage (%s)' % signal[1])
    pl.title('Sampled signal segments')
    pl.show()

def test_timestamp_raw_data():
    test_raw_data_file_path = ".\\McsPy\\tests\\TestData\\2014-07-09T10-17-35W8 Standard all 500 Hz.h5"
    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
    first_timestamp_entity = raw_data.recordings[0].timestamp_streams[0].timestamp_entity[0]
    print("Timestamp entity 0 contains: %s timestamps" % first_timestamp_entity.count)
    timestamps = first_timestamp_entity.get_timestamps()
    draw_channel_overlay_in_range_with_events(raw_data.recordings[0].analog_streams[0],
                                              raw_data.recordings[0].analog_streams[1], 
                                              list(raw_data.recordings[0].analog_streams[1].channel_infos.keys())[0],
                                              timestamps[0])

def test_imu_data():
    test_raw_data_file_path = ".\\McsPy\\tests\\TestData\\2017-10-11T13-39-47McsRecording_X981_AccGyro.h5"
    fi.print_file_info_short(test_raw_data_file_path)
    fi.print_file_info(test_raw_data_file_path)
    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)
    draw_raw_data(raw_data.recordings[0].analog_streams[4])
    draw_raw_data(raw_data.recordings[0].analog_streams[5])

def test_opto_stim_data():
    test_raw_data_file_path = ".\\McsPy\\tests\\TestData\\2017-10-11T13-39-47McsRecording_N113_OptoStim.h5"
    fi.print_file_info(test_raw_data_file_path)
    raw_data = McsPy.McsData.RawData(test_raw_data_file_path)

print('McsPy Version: %s' % McsPy.version)
fi.print_dir_file_info(".\\McsPy\\tests\\TestData")
fi.print_dir_file_info(r"/Programming/McsDataManagement/McsPyDataTools/McsPyDataNotebooks/TestData")
#McsPy.McsData.VERBOSE = False
test_channel_raw_data()
##test_frame_raw_data()
#test_event_raw_data()
#test_segment_raw_data()
#test_timestamp_raw_data()
#test_imu_data()
#test_opto_stim_data()