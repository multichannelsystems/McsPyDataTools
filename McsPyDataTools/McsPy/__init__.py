"""
    McsPy
    ~~~~~

    McsPy is a Python module/package to read, handle and operate on HDF5-based raw data
    files converted from recordings of devices of the Multi Channel Systems MCS GmbH.

    :copyright: (c) 2017 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

#print("McsPy init!")
version = "0.2.3"

# Supported MCS-HDF5 protocol types and versions:
class McsHdf5Protocols:
    """
    Class of supported MCS-HDF5 protocol types and version ranges

    Entry: (Protocol Type Name => Tuple of supported version range from (including) the first version entry up to (including) the second version entry)
    """
    SUPPORTED_PROTOCOLS = {"RawData" : (1, 3),  # from first to second version number and including this versions
                           "InfoChannel" : (1, 1), # Info-Object Versions
                           "FrameEntityInfo" : (1, 1),
                           "EventEntityInfo" : (1, 1),
                           "SegmentEntityInfo" : (1, 4),
                           "TimeStampEntityInfo" : (1, 1),
                           "AnalogStreamInfoVersion" : (1, 1), # StreamInfo-Object Versions
                           "FrameStreamInfoVersion" : (1, 1),
                           "EventStreamInfoVersion" : (1, 1),
                           "SegmentStreamInfoVersion" : (1, 1),
                           "TimeStampStreamInfoVersion" : (1, 1)}

    @classmethod
    def check_protocol_type_version(self, protocol_type_name, version):
        """
        Check if the given version of a protocol is supported by the implementation

        :param protocol_type_name: name of the protocol that is tested
        :param version: version number that should be checked
        :returns: is true if the given protocol and version is supported
        """
        if protocol_type_name in McsHdf5Protocols.SUPPORTED_PROTOCOLS:
            supported_versions = McsHdf5Protocols.SUPPORTED_PROTOCOLS[protocol_type_name]
            if (version < supported_versions[0]) or (supported_versions[1] < version):
                raise IOError('Given HDF5 file contains \'%s\' type of version %s and supported are only all versions from %s up to %s' % 
                               (protocol_type_name, version, supported_versions[0], supported_versions[1]))
        else:
            raise IOError("The given HDF5 contains a type \'%s\' that is unknown in this implementation!" % protocol_type_name)
        return True

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
