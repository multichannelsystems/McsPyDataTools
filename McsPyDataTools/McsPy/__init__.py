"""
    McsPy
    ~~~~~

    McsPy is a Python module/package to read, handle and operate on HDF5-based raw data 
    files converted from recordings of devices of the Multi Channel Systems MCS GmbH. 

    :copyright: (c) 2014 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

#print("McsPy init!")
version = 0.01
supported_mcs_hdf5_versions = (1, 1) # from first to second version number and including this versions

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
