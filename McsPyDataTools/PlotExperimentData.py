import sys

import numpy as np
import h5py

FILE = "d:\\Programming\\MCSuite\\McsDataTools\\McsDataFileConverter\\bin\\Debug\\Experiment.h5"
#FILE = "d:\\Programming\\MCSuite\\McsDataTools\\McsDataFileConverter\\bin\\Debug\\Experiment_GZipCompress.h5"
DATASET = "Data/Electrode/Channels"

from pylab import *

def show_image_plot(data):
    #matshow(data)
    #imshow(data, interpolation='nearest', cmap='bone', origin='lower')
    imshow(data, interpolation='nearest', aspect=10000)
    #colorbar(shrink=.92)
    #xticks([]), yticks([])
    show()

def run():
    file = h5py.File(FILE)
    #dset = file["Data/Electrode/Channels"]

    dset = h5py.h5d.open(file.fid, DATASET)
    dcpl = dset.get_create_plist()

    # No NBIT or SCALEOFFSET filter, but there is something new, LZF.
    ddict = {h5py.h5z.FILTER_DEFLATE: "DEFLATE",
             h5py.h5z.FILTER_SHUFFLE: "SHUFFLE",
             h5py.h5z.FILTER_FLETCHER32: "FLETCHER32",
             h5py.h5z.FILTER_SZIP: "SZIP",
             h5py.h5z.FILTER_LZF: "LZF"}

    # Retrieve and print the filter types.
    n = dcpl.get_nfilters()
    for j in range(n):
        filter_type, flags, vals, name = dcpl.get_filter(j)
        print("Filter %d: Type is H5Z_%s" % (j, ddict[filter_type])) 


    # Get the dataspace and allocate an array for reading.  Numpy makes this
    # MUCH easier than C.
    space = dset.get_space()
    dims = space.get_simple_extent_dims()
    channel_data = np.zeros(dims, dtype=np.int32)
    #channel_data = np.zeros(dset.shape,dtype = np.int32)
    dset.read(h5py.h5s.ALL, h5py.h5s.ALL, channel_data)

    channels = channel_data[0, ...]
    plot(channels)
    show()

    show_image_plot(channel_data[0:8, 0:100000])

    #X = np.linspace(-np.pi, np.pi, 256,endpoint=True)
    #C,S = np.cos(X), np.sin(X)

    #plot(X,C)
    #plot(X,S)
    #show()
    file.close()

if __name__ == "__main__":
    run()        
