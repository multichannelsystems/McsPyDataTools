#print("McsPy init!")
version = 0.01
supported_mcs_hdf5_versions = (1, 1) # from first to second version number and including this versions

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
