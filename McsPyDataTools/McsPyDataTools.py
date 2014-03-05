import McsPy
import McsData
import matplotlib
import pylab as pl
import numpy as np

from McsPy import ureg, Q_


def show_image_plot(data, aspect_ratio = 10000):
    #matshow(data)
    #imshow(data, interpolation='nearest', cmap='bone', origin='lower')
    pl.figure(figsize=(20,12))
    pl.imshow(data, interpolation='nearest', aspect=aspect_ratio)
    #colorbar(shrink=.92)
    #xticks([]), yticks([])
    pl.title('Heatmap of Wireless Signal (Simulation)')
    pl.show()

def draw_raw_data(stream):
    '''Draw all raw data streams'''
    channels = stream.channel_data
    pl.figure(figsize=(20,12))
    pl.plot(np.transpose(channels))
    pl.title('Signal for Wireless (Simulation)')
    pl.grid()
    pl.show()

def draw_channel_in_range(stream, channel_id):
    ''' Draw one channel of given ID within its original range '''
    time = stream.get_channel_sample_timestamps(channel_id,0,100000)
    signal = stream.get_channel_in_range(channel_id,0,100000)
    pl.figure(figsize=(20,12))
    pl.plot(time[0], signal[0])
    pl.xlabel('time (%s)' % time[1])
    pl.ylabel('voltage (%s)' % signal[1])
    pl.title('Sampled signal')
    pl.show()

def test_channel_raw_data():
    test_raw_data_file_path = "d:\\Programming\\MCSuite\\McsPyDataTools\\McsPyDataTools\\McsPyTests\\TestData\\Experiment.h5"

    raw_data = McsData.RawData(test_raw_data_file_path)
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
    print(raw_data.recordings[0].analog_streams)
    # Channel raw data:
    draw_raw_data(raw_data.recordings[0].analog_streams[0])
    show_image_plot(raw_data.recordings[0].analog_streams[0].channel_data[:, 0:81000], 8500)
    draw_channel_in_range(raw_data.recordings[0].analog_streams[0], raw_data.recordings[0].analog_streams[0].channel_infos.keys()[0])

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
    raw_data = McsData.RawData(test_raw_frame_data_file_path)
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
    test_raw_data_file_path = ".\\McsPyTests\\TestData\\2014-02-27T08-30-03W8SpikeCutoutsAndTimestampsAndRawData.h5"
    raw_data = McsData.RawData(test_raw_data_file_path)
    event_entity = raw_data.recordings[0].event_streams[0].event_entity[0]
    print("Event entity 0 contains: %s events" % event_entity.count)
    all_events = event_entity.get_events()
    print((all_events[0])[0,:])
    all_event_timestamps = event_entity.get_event_timestamps()
    print(all_event_timestamps[0])
    all_event_durations = event_entity.get_event_durations()
    print(all_event_durations[0])

print('McsPy Version: %s' % McsPy.version)
#test_channel_raw_data()
#test_frame_raw_data()
test_event_raw_data()