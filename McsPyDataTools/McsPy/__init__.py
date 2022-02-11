"""
    McsPy
    ~~~~~

    McsPy is a Python module/package to read, handle and operate on HDF5-based raw data
    files converted from recordings of devices of the Multi Channel Systems MCS GmbH.

    :copyright: (c) 2022 by Multi Channel Systems MCS GmbH
    :license: see LICENSE for more details
"""

#print("McsPy init!")
version = "0.4.2"

#__all__ = ["CMOSData", "CMOSConvProxy", "RawData", "Recording", "Stream", "AnalogStream", 
#           "Info", "InfoSampledData", "ChannelInfo", "FrameStream", "FrameEntity", "Frame", 
#           "FrameEntityInfo", "EventStream", "EventEntity", "EventEntityInfo", "SegmentStream", 
#           "SegmentEntity", "AverageSegmentTuple", "AverageSegmentEntity", "SegmentEntityInfo",
#           "TimeStampStream", "TimeStampEntity", "TimeStampEntityInfo"]

# Supported MCS-HDF5 protocol types and versions:
class McsHdf5Protocols:
    """
    Class of supported MCS-HDF5 protocol types and version ranges

    Entry: (Protocol Type Name => Tuple of supported version range from (including) the first version entry up to (including) the second version entry)
    """
    SUPPORTED_PROTOCOLS = {"RawData" : (1, 3),  # from first to second version number and including this versions
                           "CMOS_MEA" : (1, 1), #from first to first version
                           "InfoChannel" : (1, 2), # Info-Object Versions
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

# Supported MCS-HDF5 file structure types and versions:
class McsHdf5Types:
    """
    Class of supported MCS-HDF5 file structure types and version ranges

    Entry: (Protocol TypeID => Tuple of supported version range from (including) the first version entry up to (including) the second version entry)
    """
    SUPPORTED_TYPES =  {"RawData" : (1, 3),  # from first to second version number and including this versions
                        "cabb6cdd-47e0-417a-8e04-5664cbbc449b" : {"McsPyClass": "McsCMOSMEAData",   "Tag":  None},                  #CMOSMEA file format, from first to first version
                        "650d88ce-9f24-4b20-ac2b-254defd12761" : {"McsPyClass": "Acquisition",      "Tag":  None},                  #Acquisition group
                        "9217aeb4-59a0-4d7f-bdcd-0371c9fd66eb" : {"McsPyClass": "McsChannelStream", "Tag":  "Channel Stream"},      #Analog Stream group (comprises analog and digital data)
                        "9e8ac9cd-5571-4ee5-bbfa-8e9d9c436daa" : {"McsPyClass": "McsInfo",          "Tag":  "Channel Stream Meta"}, #Analog Stream Meta Dataset
                        "5efe7932-dcfe-49ff-ba53-25accff5d622" : {"McsPyClass": "McsChannelEntity", "Tag":  "Channel Stream Data"}, #Analog Stream Data Dataset
                        "09f288a5-6286-4bed-a05c-02859baea8e3" : {"McsPyClass": "McsEventStream",   "Tag":  "Event Stream"},        #Event Stream group
                        "8f58017a-1279-4d0f-80b0-78f2d80402b4" : {"McsPyClass": "McsInfo",          "Tag":  "Event Stream Meta"},   #Event Meta Dataset
                        "abca7b0c-b6ce-49fa-ad74-a20c352fe4a7" : {"McsPyClass": "McsDataset",       "Tag":  "Event Stream Data"},   #Event Data Dataset
                        "15e5a1fe-df2f-421b-8b60-23eeb2213c45" : {"McsPyClass": "McsSensorStream",  "Tag":  "Sensor Stream"},       #Sensor Stream group, FrameStream
                        "ab2aa189-2e72-4148-a2ef-978119223412" : {"McsPyClass": "McsInfo",          "Tag":  "Sensor Stream Meta"},  #Sensor Meta Dataset
                        "49da47df-f397-4121-b5da-35317a93e705" : {"McsPyClass": "McsSensorEntity",  "Tag":  "Sensor Stream Data"},  #Sensor Data Dataset
                        "35f15fa5-8427-4d07-8460-b77a7e9b7f8d" : {"McsPyClass": "SegmentStream",    "Tag":  "Segment Stream"},      #SegmentStream"
                        "425ce2e0-f1d6-4604-8ab4-6a2facbb2c3e" : {"McsPyClass": None,               "Tag":  "TimeStamp Stream"},    #TimeStampStream
                        "26efe891-c075-409b-94f8-eb3a7dd68c94" : {"McsPyClass": "McsSpikeStream",   "Tag":  "Spike Stream"},        #SpikeStream
                        "e1d7616f-621c-4a26-8f60-a7e63a9030b7" : {"McsPyClass": "McsInfo",          "Tag":  "Spike Stream Meta"},   #SpikeStream Meta Dataset
                        "3e8aaacc-268b-4057-b0bb-45d7dc9ec73b" : {"McsPyClass": "McsSpikeEntity",   "Tag":  "Spike Stream Data"},   #SpikeStream Data Dataset
                        "2f8c246f-9bab-4193-b09e-03aefe17ede0" : {"McsPyClass": "FilterTool",       "Tag":  None},                  #Filter Tool group
                        "c632506d-c961-4a9f-b22b-ac7a56ce3552" : {"McsPyClass": None,               "Tag":  None},                  #Pipe Tool group
                        "941c8edb-78b3-4275-a5b2-6876cbcdeffc" : {"McsPyClass": "NetworkExplorer",  "Tag":  None},                  #STA Explorer group
                        "442b7514-fe3a-4c66-8ae9-4f249ef48f2f" : {"McsPyClass": None,               "Tag":  None},                  #STA Entity Dataset
                        "a95db4a1-d124-4c52-8889-2264fcdb489b" : {"McsPyClass": None,               "Tag":  None},                  #SettingsMapCreatorSpike and SettingsMapCreatorSta Dataset
                        "de316ac6-ad66-4d78-acc4-e3f29bd40991" : {"McsPyClass": None,               "Tag":  None},                  #SettingsVideoControl Dataset
                        "44b29fba-ec5c-48b5-8e0e-02ad9b9ac83a" : {"McsPyClass": None,               "Tag":  None},                  #SettingsStaExplorer Dataset
                        "935a1aa6-4082-482e-9d4d-1ad60d1b1680" : {"McsPyClass": None,               "Tag":  None},                  #SettingsStaCreator Dataset
                        "c6a37148-fa9e-42f2-9d38-eea0434851e2" : {"McsPyClass": "SpikeExplorer",    "Tag":  None},                  #Spike Explorer group
                        "58c92502-516e-46f6-ac50-44e6dd17a3ff" : {"McsPyClass": None,               "Tag":  None},                  #SettingsSpikeDetector Dataset
                        "ef54ef3d-3619-43aa-87ba-dc5f57f7e861" : {"McsPyClass": None,               "Tag":  None},                  #SettingsSpikeExplorer Dataset
                        "1b4e0b8b-6af1-4b55-a685-a6d28a922eb3" : {"McsPyClass": "McsSpikeEntity",   "Tag":  "Spike Data"},                  #SpikeData Dataset
                        "f5dc873b-4aed-4a54-8c19-5743908684bb" : {"McsPyClass": None,               "Tag":  None},                  #SpikePeakActivity Dataset
                        "7263d1b7-f57a-42de-8f51-5d6326d22f2a" : {"McsPyClass": "SpikeSorter",      "Tag":  None},                  #Spike Sorter group
                        "0e5a97df-9de0-4a22-ab8c-54845c1ff3b9" : {"McsPyClass": "SpikeSorterUnitEntity","Tag":  None},              #Spike Sorter Entity group
                        "3fa908a3-fac9-4a80-96a1-310d9bcdf617" : {"McsPyClass": None,               "Tag":  None},                  #ProjectionMatrix Dataset
                        "3533aded-b369-4529-836d-9629eb1a27a8" : {"McsPyClass": None,               "Tag":  None},                  #SettingsPeakDetection Dataset
                        "f20b653e-25fb-4f7a-ae8a-f35044f46720" : {"McsPyClass": None,               "Tag":  None},                  #SettingsPostProcessing Dataset
                        "c7d23018-9006-45fe-942f-c5d0f9cde284" : {"McsPyClass": None,               "Tag":  None},                  #SettingsRoiDetection Dataset
                        "713a9202-87e1-4bfe-ba80-b909a000aae5" : {"McsPyClass": None,               "Tag":  None},                  #SettingsSorterComputing Dataset
                        "62bc7b9f-7eea-4a88-a438-c618067d49f4" : {"McsPyClass": None,               "Tag":  None},                  #SettingsSorterGeneral
                        "9cdcea3f-88aa-40cf-89db-818315a2644a" : {"McsPyClass": "ActivitySummary",  "Tag":  None},                  #Activity Summary group
                       }

    @classmethod
    def get_mcs_class_name(self, typeID):
        """
        Returns the McsPy class name, that corresponds to a given Mcs HDF5 file structure type. The function also checks if the requested class supports 
        the Mcs HDF5 file structure type version

        :param typeID: name of the type that is tested
        :returns: a McsCMOSMEA class if the given type and version is supported
        """
        if not typeID in McsHdf5Types.SUPPORTED_TYPES:
            return None
        class_name = McsHdf5Types.SUPPORTED_TYPES[typeID]['McsPyClass']
        if class_name is None:
            return None
        return getattr(McsCMOSMEA, class_name)

from pint import UnitRegistry
ureg = UnitRegistry()
Q_ = ureg.Quantity
ureg.define('NoUnit = [quantity]')

from McsPy import McsCMOSMEA