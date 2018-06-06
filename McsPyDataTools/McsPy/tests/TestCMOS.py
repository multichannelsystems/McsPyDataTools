import unittest
import McsPy.McsCMOS
import os
import numpy

FILENAME = os.path.join(os.path.dirname(__file__), 'TestData', 'CMOSTestRec.h5')

class TestMcsCMOS(unittest.TestCase):

    def setUp(self):
        self.test_file=McsPy.McsCMOS.CMOSData(FILENAME)

    def test_CMOSData_meta(self):

        #File Meta
        self.assertEqual(self.test_file.meta["McsHdf5ProtocolType"], "RawData")
        self.assertEqual(self.test_file.meta["McsHdf5ProtocolVersion"], 1)

        # Data Meta
        self.assertEqual(self.test_file.meta["ProgramName"].strip(), "CMOS-MEA-Control")
        self.assertEqual(self.test_file.meta["ProgramVersion"].strip(), "0.7.0.0")
        self.assertEqual(self.test_file.meta["MeaName"].strip(), "nMos32?")
        self.assertEqual(self.test_file.meta["MeaLayout"], "")
        self.assertEqual(self.test_file.meta["MeaSN"].strip(), "unknown")
        self.assertEqual(self.test_file.meta["Date"].strip(), "Tuesday, November 04, 2014")
        self.assertEqual(self.test_file.meta["DateInTicks"], 635506934728348929)
        self.assertEqual(self.test_file.meta["FileGUID"], "67ced1bf-c1a7-4a3d-9df3-2e56fd459cbd")
        self.assertEqual(self.test_file.meta["Comment"], "")

        # InfoFrame Meta
        self.assertEqual(self.test_file.meta["FrameID"], 1)
        self.assertEqual(self.test_file.meta["FrameDataID"], 0)
        self.assertEqual(self.test_file.meta["GroupID"], 1)
        self.assertEqual(self.test_file.meta["Label"], "ROI 1")
        self.assertEqual(self.test_file.meta["RawDataType"], "Short")
        self.assertEqual(self.test_file.meta["Unit"], "V")
        self.assertEqual(self.test_file.meta["Exponent"], -9)
        self.assertEqual(self.test_file.meta["ADZero"], 0)
        self.assertEqual(self.test_file.meta["Tick"], 50)
        self.assertEqual(self.test_file.meta["HighPassFilterType"], "")
        self.assertEqual(self.test_file.meta["HighPassFilterCutOffFrequency"], "-1")
        self.assertEqual(self.test_file.meta["HighPassFilterOrder"], -1)
        self.assertEqual(self.test_file.meta["LowPassFilterType"], "")
        self.assertEqual(self.test_file.meta["LowPassFilterCutOffFrequency"], "-1")
        self.assertEqual(self.test_file.meta["LowPassFilterOrder"], -1)
        self.assertEqual(self.test_file.meta["SensorSpacing"], 1)
        self.assertEqual(self.test_file.meta["FrameLeft"], 1)
        self.assertEqual(self.test_file.meta["FrameTop"], 1)
        self.assertEqual(self.test_file.meta["FrameRight"], 65)
        self.assertEqual(self.test_file.meta["FrameBottom"], 65)
        self.assertEqual(self.test_file.meta["ReferenceFrameLeft"], 1)
        self.assertEqual(self.test_file.meta["ReferenceFrameTop"], 1)
        self.assertEqual(self.test_file.meta["ReferenceFrameRight"], 65)
        self.assertEqual(self.test_file.meta["ReferenceFrameBottom"], 65)

    def test_CMOSData_data(self):

        # Dataset Dimensions
        self.assertEqual(self.test_file.raw_data.shape[0],65)
        self.assertEqual(self.test_file.raw_data.shape[1],65)
        self.assertEqual(self.test_file.raw_data.shape[2],2000)

        #Some Random Datapoints
        self.assertEqual(self.test_file.raw_data[56,45,33],10)
        self.assertEqual(self.test_file.raw_data[1,11,203],-3)
        self.assertEqual(self.test_file.raw_data[23,64,870],0)

    def test_CMOSData_conversion_factors(self):
        self.assertEqual(self.test_file.conv_factors[56,45],6456)
        self.assertEqual(self.test_file.conv_factors[1,11],1)
        self.assertEqual(self.test_file.conv_factors[23,64],1)

    def test_CMOSData_conversion_proxy(self):

        #Data Access
        self.assertEqual(self.test_file.conv_data[56,45,33],64560)
        self.assertEqual(self.test_file.conv_data[1,11,203],-3)
        self.assertEqual(self.test_file.conv_data[23,64,870],0)

        #Attribute proxy
        self.assertEqual(self.test_file.conv_data.dtype,numpy.int32)
        self.assertEqual(self.test_file.conv_data[23,64,870].dtype,numpy.int32)

if __name__ == '__main__':
    unittest.main()