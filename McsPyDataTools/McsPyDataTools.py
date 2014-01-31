import McsPy
import McsData
import matplotlib
import pylab as pl
import numpy as np

from McsPy import ureg, Q_


raw_data_file_path = "d:\\Programming\\MCSuite\\McsDataTools\\McsDataFileConverter\\bin\\Debug\\Experiment.h5"

def show_image_plot(data, aspect_ratio = 10000):
    #matshow(data)
    #imshow(data, interpolation='nearest', cmap='bone', origin='lower')
    pl.figure(figsize=(20,12))
    pl.imshow(data, interpolation='nearest', aspect=aspect_ratio)
    #colorbar(shrink=.92)
    #xticks([]), yticks([])
    pl.title('Heatmap of Wireless Signal (Simulation)')
    pl.show()

def draw_raw_data(raw_data):
    '''Draw all raw data streams'''
    channels = raw_data.recordings[0]. analog_streams[0].channel_data #channel_data[0, ...]
    pl.figure(figsize=(20,12))
    pl.plot(np.transpose(channels))
    pl.title('Signal for Wireless (Simulation)')
    pl.grid()
    pl.show()

def draw_channel_in_range(raw_data, channel_id):
    ''' Draw one channel of given ID within its original range '''
    time = raw_data.recordings[0].analog_streams[0].get_channel_timepoints(0,0,8100)
    signal = raw_data.recordings[0].analog_streams[0].get_channel_in_range(0,0,8100)
    pl.figure(figsize=(20,12))
    pl.plot(time[0], signal[0])
    pl.xlabel('time (%s)' % time[1])
    pl.ylabel('voltage (%s)' % signal[1])
    pl.title('Sampled signal')
    pl.show()

print('McsPy Version: %s' % McsPy.version)

raw_data = McsData.RawData(raw_data_file_path)
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

draw_raw_data(raw_data)
 
show_image_plot(raw_data.recordings[0]. analog_streams[0].channel_data[0:8, 0:8100], 850)

draw_channel_in_range(raw_data, 0)
