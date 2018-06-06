import sys
sys.path.append(r"../")
sys.executable

# These are the imports of the McsData module
import McsPy.McsData
import McsPy.functions_info
from McsPy import ureg, Q_

import skinematics as skin
from skinematics.imus import IMU_Base

from scipy import constants # for "g"

# matplotlib.pyplot will be used in these examples to generate the plots visualizing the data
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.widgets import Slider
from mpl_toolkits.mplot3d import Axes3D

# numpy is numpy ...
import numpy as np
import McsPy.functions_info as fi

#fi.print_dir_file_info(r"../McsPyDataTools/McsPy/tests/TestData")


acc_gyro_raw_data_file_path = "../McsPyDataTools/McsPy/tests/TestData/2017-10-11T13-39-47McsRecording_X981_AccGyro.h5"
#fi.print_file_info(acc_gyro_raw_data_file_path)


# Load the file in silent mode:

# In[47]:


McsPy.McsData.VERBOSE = False
raw_data = McsPy.McsData.RawData(acc_gyro_raw_data_file_path)


# ## Gyroscope Data<a id='Gyroscope Data'></a>

# In[48]:


gyro_channel = raw_data.recordings[0].analog_streams[4]
print('Channel IDs: %s' % gyro_channel.channel_infos.keys())


# In[49]:


gyro = np.transpose(gyro_channel.channel_data)
gyro.shape


# In[50]:


#plt.figure(figsize=(12,6))
#plt.plot(gyro)
##plt.title('Signal for Wireless (Simulation) / Raw ADC-Values (%s)' % analog_stream_0.label)
#plt.xlabel('Sample Index')
#plt.ylabel('Gyroscope Value')
#plt.grid()

#plt.show()


# Cutout invalid data parts:

# In[51]:


gyro = gyro[0:10000,0:3]
gyro.shape


# In[52]:


time, time_unit = gyro_channel.get_channel_sample_timestamps(148,0,10000)
gyro_x, gyro_x_unit = gyro_channel.get_channel_in_range(148,0,10000)
gyro_y, gyro_y_unit = gyro_channel.get_channel_in_range(149,0,10000)
gyro_z, gyro_z_unit = gyro_channel.get_channel_in_range(150,0,10000)
#plt.figure(figsize=(20,12))
#plt.plot(time, gyro_x)
#plt.plot(time, gyro_y)
#plt.plot(time, gyro_z)
#plt.xlabel('Time (%s)' % time_unit)
#plt.ylabel('Angular Speed (%s)' % gyro_x_unit)
#plt.title('Gyroscope Data')
#plt.show()


# ## Accelerometer Data<a id='Accelerometer Data'></a>

# In[53]:


acc_channel = raw_data.recordings[0].analog_streams[5]
print('Channel IDs: %s' % acc_channel.channel_infos.keys())


# In[54]:


acc = np.transpose(acc_channel.channel_data)
acc.shape


# In[55]:


#plt.figure(figsize=(12,6))
#plt.plot(acc)
##plt.title('Signal for Wireless (Simulation) / Raw ADC-Values (%s)' % analog_stream_0.label)
#plt.xlabel('Sample Index')
#plt.ylabel('Accelerometer Value')
#plt.grid()

#plt.show()


# Cutout invalid data parts:

# In[56]:


acc = acc[0:10000,0:3]
acc.shape


# In[57]:


time, time_unit = acc_channel.get_channel_sample_timestamps(160,0,10000)
acc_x, acc_x_unit = acc_channel.get_channel_in_range(160,0,10000)
acc_y, acc_y_unit = acc_channel.get_channel_in_range(161,0,10000)
acc_z, acc_z_unit = acc_channel.get_channel_in_range(162,0,10000)
#plt.figure(figsize=(20,12))
#plt.plot(time, acc_x)
#plt.plot(time, acc_y)
#plt.plot(time, acc_z)
#plt.xlabel('Time (%s)' % time_unit)
#plt.ylabel('Acceleration (%s)' % acc_x_unit)
#plt.title('Accelerometer Data')
#plt.show()


# <a href='#Top'>Back to index</a>

class McsIMU(IMU_Base):
    """Concrete class based on abstract base class IMU_Base """    
    
    def get_data(self, in_file, in_data):
        '''Get the sampling rate, as well as the recorded data,
        and assign them to the corresponding attributes of "self".
        
        Parameters
        ----------
        in_file : string
                Filename of the data-file
        in_data : 
                Sampling rate (has to be provided!!)
        
        Assigns
        -------
        - rate : rate
        - acc : acceleration
        - omega : angular_velocity
        '''
        
        # The sampling rate has to be provided externally
        rate = in_data['rate']
            
        # Get the data, and label them
        data.columns = ['acc_x', 'acc_y', 'acc_z', 'gyr_x', 'gyr_y', 'gyr_z', 'mag_x', 'mag_y', 'mag_z', 'taccgyr', 'tmag']
            
        # Set the conversion factors by hand, and apply them
        #conversions = {}
        #conversions['time'] = 1/1000000
        #conversions['acc'] = 9.81
        #conversions['gyr'] = np.pi/180   
        #data[:,:3] *= conversions['acc']
        #data[:,3:6] *= conversions['gyr']
        #data[:,6] *= conversions['time']
            
        returnValues = [rate]
        
        # Extract the columns that you want, by name
        #paramList=['acc', 'gyr', 'mag']
        #for param in paramList:
        #    Expression = param + '*'
        #    returnValues.append(data_interp.filter(regex=Expression).values)
        returnValues.append(in_data['acc'])
        returnValues.append(in_data['gyro'])
        self._set_info(*returnValues)

# Set the conversion factors by hand, and apply them
conversions = {}
conversions['time'] = 1/1000000
conversions['acc'] = constants.g
conversions['gyr'] = np.pi/180

acc = np.column_stack((acc_x, acc_y, acc_z)) * conversions['acc']
gyro = np.column_stack((gyro_x, gyro_y, gyro_z)) * conversions['gyr']
time_second = time * conversions['time']

acc_sub = acc[::5,:].copy()
gyro_sub = gyro[::5,:].copy()

initial_orientation = np.array([[1,0,0],
                                [0,1,0],
                                [0,0,1]])
in_data = {"rate" : 2000, "acc" : acc, "omega" : gyro, "mag": None}
in_data_subsampled = {"rate" : 400, "acc" : acc_sub, "omega" : gyro_sub, "mag": None}
#mcs_imu = McsIMU(in_file = None, R_init = initial_orientation, in_data = in_data)
mcs_imu = McsIMU(in_file = None, R_init = initial_orientation, in_data = in_data_subsampled)
# mcs_imu.get_data(None, {'rate': 2000, 'acc': acc, 'gyro': gyro})

def show_result(imu_data):
        fig, axs = plt.subplots(3,1)
        axs[0].plot(imu_data.omega)
        axs[0].set_ylabel('Omega')
        axs[0].set_title(imu_data.q_type)
        axs[1].plot(imu_data.acc)
        axs[1].set_ylabel('Acc')
        axs[2].plot(imu_data.quat[:,1:])
        axs[2].set_ylabel('Quat')
        plt.show()

show_result(mcs_imu)

mcs_imu.q_type = 'analytical'
mcs_imu.calc_position()
pos_data = mcs_imu.pos

fig = plt.figure()
ax = fig.gca(projection='3d')
ax.plot(pos_data[:,0], pos_data[:,1], pos_data[:,2], label='estimated position')
ax.legend()

plt.show()
