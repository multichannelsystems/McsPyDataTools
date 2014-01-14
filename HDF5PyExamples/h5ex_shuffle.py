"""
This example shows how to read and write data to a dataset using
the shuffle filter with gzip compression.  The program first checks
if the shuffle and gzip filters are available, then if they are it
writes integers to a dataset using shuffle+gzip, then closes the
file.  Next, it reopens the file, reads back the data, and outputs
the types of filters and the maximum value in the dataset to the
screen.

Tested with:
    Fedora 18:
        HDF5 1.8.9, Python 2.7.3, Numpy 1.7.1, h5py 2.1.3
    Fedora 18:
        HDF5 1.8.9, Python 3.3.0, Numpy 1.7.1, h5py 2.1.3
    Mac OS X 10.6.8:
        HDF5 1.8.10, Python 3.2.5, Numpy 1.7.1, h5py 2.1.3
"""
import sys

import numpy as np
import h5py

FILE = "h5ex_d_shuffle.h5"
DATASET = "DS1"

# Strings are handled very differently between python2 and python3.
if sys.hexversion >= 0x03000000:
    FILE = FILE.encode()
    DATASET = DATASET.encode()

DIM0 = 32
DIM1 = 64
CHUNK0 = 4
CHUNK1 = 8

def run():

    # Check if gzip compression is available and can be used for
    # both compression and decompression.  Normally we do not perform
    # error checking in these examples for the sake of clarity, but
    # in this case we will make an exception because this filter is
    # an optional part of the hdf5 library. 
    if not h5py.h5z.filter_avail(h5py.h5z.FILTER_DEFLATE):
        raise RuntimeError("Gzip filter not available.")

    filter_info = h5py.h5z.get_filter_info(h5py.h5z.FILTER_DEFLATE)
    if ((filter_info & h5py.h5z.FILTER_CONFIG_ENCODE_ENABLED) & 
        (filter_info & h5py.h5z.FILTER_CONFIG_DECODE_ENABLED)):
        msg = "Gzip filter not available for encoding and decoding."
        raise RuntimeError(msg)

    # Similarly, check for availability of the shuffle filter.
    if not h5py.h5z.filter_avail(h5py.h5z.FILTER_SHUFFLE):
        raise RuntimeError("Shuffle filter not available.")

    filter_info = h5py.h5z.get_filter_info(h5py.h5z.FILTER_SHUFFLE)
    if ((filter_info & h5py.h5z.FILTER_CONFIG_ENCODE_ENABLED) & 
        (filter_info & h5py.h5z.FILTER_CONFIG_DECODE_ENABLED)):
        msg = "Shuffle filter not available for encoding and decoding."
        raise RuntimeError(msg)

    # Initialize the data.
    wdata = np.zeros((DIM0, DIM1), dtype=np.int32)
    for i in range(DIM0):
        for j in range(DIM1):
            wdata[i][j] = i * j - j

    # Create a new file using the default properties.
    fid = h5py.h5f.create(FILE)

    # Create the dataspace.  No maximum size parameter needed.
    dims = (DIM0, DIM1)
    space_id = h5py.h5s.create_simple(dims)

    # Create the dataset creation property list, add the fletcher32 filter
    # and set a chunk size.
    dcpl = h5py.h5p.create(h5py.h5p.DATASET_CREATE)
    dcpl.set_shuffle()
    dcpl.set_deflate(9)

    chunk = (CHUNK0, CHUNK1)
    dcpl.set_chunk(chunk)

    # Create the datasets using the dataset creation property list.
    dset = h5py.h5d.create(fid, DATASET, h5py.h5t.STD_I32LE, space_id, dcpl)

    # Write the data to the dataset.
    dset.write(h5py.h5s.ALL, h5py.h5s.ALL, wdata)

    # Close and release resources.
    del dcpl
    del dset
    del space_id
    del fid

    # Reopen the file and dataset using default properties.
    fid = h5py.h5f.open(FILE)
    dset = h5py.h5d.open(fid, DATASET)
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

    rdata = np.zeros((DIM0, DIM1))
    dset.read(h5py.h5s.ALL, h5py.h5s.ALL, rdata)

    # Verify that the dataset was read correctly.
    np.testing.assert_array_equal(rdata, wdata)
    print("Maximum value in DS1 is:  %d" % rdata.max())

if __name__ == "__main__":
    run()        
