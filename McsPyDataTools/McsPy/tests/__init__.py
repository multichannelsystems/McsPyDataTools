"""
    McsPy
    ~~~~~

    McsPy is a Python module/package to read, handle and operate on HDF5-based raw data
    files converted from recordings of devices of the Multi Channel Systems MCS GmbH.

    :copyright: (c) 2018 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

#print("McsPy init!")
version = "0.3.0"

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

#from McsPy import McsCMOSMEA