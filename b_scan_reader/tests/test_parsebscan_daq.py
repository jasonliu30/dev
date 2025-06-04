import math
import os
import numpy as np
import pytest
from pathlib import Path
from b_scan_reader import BScan_daq, bscan_structure_definitions
from b_scan_reader.BScan_reader import load_bscan
from b_scan_reader.bscan_structure_definitions import *

# Go to the root repo folder
if Path(os.getcwd()) != Path(__file__).parent.parent:
    os.chdir(Path(__file__).parent.parent)

from file_paths import RAW_DATA_ROOT_DIR

FLOAT_RELATIVE_TOLERANCE = 1e-5

input_scan_path = RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2251\L12\BSCAN P2251 L12E 2022-Feb-01 223037 [A5002-5030].daq"


@pytest.mark.regression_test
def test_dscan_header_parsed_correctly():
    """
    Regression test to ensure that the function get_header() reads the file header as expected. An anf file is loaded, and the function get_header() is called. The result is compared to expected hardcoded values.

    This test passes if the header produced by get_header() matches the expected hardcoded values, otherwise this test fails.
    """

    scan = BScan_daq.BScan_daq(input_scan_path)
    header = scan.get_header()

    assert header.AxesLabel == 'Axial'
    assert header.AxialCache == [1167870059, 1167871032, 1167872006, 1167872840, 1167873674, 1167874508, 1167875203,
                                 1167876038, 1167876733, 1167877567, 1167878540, 1167879514, 1167880209, 1167881182,
                                 1167881877, 1167882712, 1167883546, 1167884380, 1167885214, 1167886049, 1167886883,
                                 1167887717, 1167888551, 1167889385, 1167890220, 1167891054, 1167891888, 1167892722,
                                 1167893557, 1167894391, 1167895225, 1167896059, 1167896894, 1167897728, 1167898562,
                                 1167899396, 1167900230, 1167901065, 1167901899, 1167902733, 1167903567, 1167904402,
                                 1167905236, 1167906070, 1167906904, 1167907739, 1167908573, 1167909407, 1167910241,
                                 1167910936, 1167911910, 1167912605, 1167913439, 1167914273, 1167915108, 1167915942,
                                 1167916776, 1167917610, 1167918445, 1167919279, 1167920113, 1167920947, 1167921781,
                                 1167922616, 1167923450, 1167924284, 1167925118, 1167925953, 1167926787, 1167927621,
                                 1167928455, 1167929290, 1167930124, 1167930958, 1167931375, 1167931514]
    assert header.ChannelLabels == ['ID2 NB', 'CW Shear P/E', 'Circ Shear P/C', 'Comb Mat/OD NB', 'CCW Shear P/E',
                                    'BWD Shear P/E', 'ID1 NB', 'FWD Shear P/E', 'Axial Shear P/C', 'OD-focus P/E/ID1',
                                    'Gauging WP/WT', 'Gauging Ref']
    assert header.DataOffset == 15908
    assert header.Day == '1'
    assert header.FooterOffset == 9624198692
    assert header.FrameAxisInfoAvail == 1
    assert header.FrameCount == 76
    assert header.Hour == '22'
    assert header.InspectionHeadId == 'TM12/A556'
    assert header.Minute == '30'
    assert header.Month == 2
    assert header.NumAxes == 1
    assert header.NumExtendedInfo == 14
    assert header.NumMetaData == 0
    assert header.NumScanInfo == 10
    assert header.NumUTChannels == 12
    assert header.Operator == 'Days'
    assert header.Prefix == 'ANDEDAQ'
    assert header.PrimaryAxisInfoAvail == 1
    assert header.ProbeDesc == 'DAQ1|DAQ2|DAQ3|DAQ4'
    assert header.ProbeDescLength == 19
    assert header.ScanDate == '1-Feb-22'
    assert header.ScanTime == '22:30:37'
    assert header.Second == '37'
    assert header.SerialNumber == '12 14 9 13'
    assert header.VersionMajor == 6
    assert header.VersionMinor == 5
    assert header.Year == '22'


@pytest.mark.unit_test
def test_invalid_path_throws_exception():
    """
    Test Overview:
    Failure test to ensure that the software fails as expected when trying to load a file that does not exist.

    Test Requirements:
    This test passes of an exception is raised. This test fails if no exception is raised.

    Test Data:
    The test attempts to load a BScan from path 'garbage_input_path_that_doesnt_exist.daq'
    """

    # TODO: make a specific exception for this
    with pytest.raises(Exception) as _:
        BScan_daq.BScan_daq(Path('garbage_input_path_that_doesnt_exist.daq'))


@pytest.mark.regression_test
def test_daq_nb1_data_parsed_correctly():
    """
    Regression test to ensure that the function read_channel_data() reads the channel data as expected for Probe NB1. A daq file is loaded, and the function read_channel_data() is called. The result is compared to expected hardcoded values.

    This test passes if the data values and length produced by read_channel_data() matches the expected hardcoded values, otherwise this test fails.
    """

    scan = BScan_daq.BScan_daq(input_scan_path)
    data = scan.read_channel_data(Probe.NB1)

    assert (data.data[0, 0, 0] == 1996)
    assert (data.data[1, 2, 42] == 2067)
    assert (data.data[-1, -1, -1] == 2052)

    assert data.data.shape == (76, 3600, 844)
    assert len(data.axes.axial_pos) == 76
    assert len(data.axes.rotary_pos) == 3600
    assert len(data.axes.time_pos) == 845


@pytest.mark.regression_test
def test_daq_nb2_data_parsed_correctly():
    """
    Regression test to ensure that the function read_channel_data() reads the channel data as expected for Probe NB2. A daq file is loaded, and the function read_channel_data() is called. The result is compared to expected hardcoded values.

    This test passes if the data values and length produced by read_channel_data() matches the expected hardcoded values, otherwise this test fails.
    """

    scan = BScan_daq.BScan_daq(input_scan_path)
    data = scan.read_channel_data(Probe.NB2)

    assert (data.data[0, 0, 0] == 2053)
    assert (data.data[1, 2, 42] == 2031)
    assert (data.data[-1, -1, -1] == 2043)

    assert data.data.shape == (76, 3600, 844)
    assert len(data.axes.axial_pos) == 76
    assert len(data.axes.rotary_pos) == 3600
    assert len(data.axes.time_pos) == 845


def check_all_properties_equal(object1, object2, tolerance=FLOAT_RELATIVE_TOLERANCE):
    for prop in filter(lambda s: not s.startswith('__'), dir(object1)):
        attr1 = getattr(object1, prop)
        attr2 = getattr(object2, prop)
        # Check if either property is callable. If one is, it's likely a function. Can't test those with ==
        if not (callable(attr1) and callable(attr2)):
            try:
                assert math.isclose(attr1, attr2, rel_tol=tolerance), "\t".join([str(prop), str(attr1), str(attr2)])
            except:
                assert attr1 == attr2
        else:
            pass
            # print("Debug Ignoring property: ", prop)


@pytest.mark.regression_test
def test_daq_successive_reads_match():
    """
    Regression test to ensure that successive calls of the same functions always return the same results.
    The header, hardware info, and channel data are called multiple times and in different ways.

    This test passes if the values of the header, hardware info, channel data remain unchanged when the functions are called several times. This test fails if any values change.
    """

    scan = BScan_daq.BScan_daq(input_scan_path)

    # For each header type try:
    # 1. Pull the property from the class as a property
    # 2. Call the appropriate accessor

    header1 = scan.header
    header2 = scan.get_header()

    check_all_properties_equal(header1, header2)

    label = Probe.NB1
    data = scan.read_channel_data(label)
    data2 = scan.read_channel_data(label)
    assert np.all(data.data == data2.data)
    assert np.all(data.axes.axial_pos == data2.axes.axial_pos)
    assert np.all(data.axes.rotary_pos == data2.axes.rotary_pos)
    assert np.all(data.axes.time_pos == data2.axes.time_pos)

    assert len(data2.axes.axial_pos) == 76
    assert len(data2.axes.rotary_pos) == 3600
    assert len(data2.axes.time_pos) == 845


@pytest.mark.regression_test
def test_multiple_concurrent_ascan_files():
    """
    Regression test to ensure that loading a .daq BScan file multiple times and reading the channel data multiple times returns the same results.
    A .daq file is loaded, read_channel_data() is called, and the length of the Axial Position axis is checked. This is then repeated.

    This test passes if the length of the Axial Postion axis equals the expected length both times, otherwise this test fails.
    """

    scan = BScan_daq.BScan_daq(input_scan_path)
    label = Probe.NB1
    data = scan.read_channel_data(label)
    assert len(data.axes.axial_pos) == 76

    scan2 = BScan_daq.BScan_daq(input_scan_path)
    data2 = scan2.read_channel_data(label)
    assert len(data2.axes.axial_pos) == 76


file_strings = [
    RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2251\T07\BSCAN P2251 T07E 2022-Feb-05 014348 [A0076-0121].daq",
    RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2251\L06\BSCAN P2251 L06E 2022-Feb-01 145125 [A0076-0121].daq"]
files = [Path(file_st) for file_st in file_strings]
roi = ROI()
roi.axial_start_index = int(0)
roi.axial_end_index = int(50)


@pytest.mark.unit_test
@pytest.mark.parametrize('file', files)
def test_get_probe_subset_without_value_error(file):
    """
    Test Overview:
    This test checks to make sure that the specified file can be loaded, and that read_channel_data_subset() can read the NB1 probe data.

    Test Requirements:
    This test passes if read_channel_data_subset() completes without raising an exception. This test fails if an exception is raised.

    Test Data:
    This test attempts to load a real BScan from one of the following paths (indexed by unit test name suffix):
    Scan Data - Pickering\P2251\T07\BSCAN P2251 T07E 2022-Feb-05 014348 [A0076-0121].daq
    Scan Data - Pickering\P2251\L06\BSCAN P2251 L06E 2022-Feb-01 145125 [A0076-0121].daq
    """
    scan = load_bscan(file)
    try:
        _ = scan.read_channel_data_subset(Probe.NB1, roi).data
        assert True
    except ValueError:
        assert False


@pytest.mark.regression_test
def test_read_all_axial_positions():
    """
    This test checks that get_channel_axes() returns the expected value. The axial position array of the NB1 probe is compared to a hardcoded array.

    This test passes if the value of the axial position array equals the hardcoded expected output. This test fails if the output does not match the expected output.
    """
    expected_output = [4125.8818359375, 4126.2890625, 4126.6962890625, 4127.103515625, 4127.51123046875,
                       4127.91845703125, 4128.32568359375, 4128.73291015625, 4129.140625, 4129.5478515625,
                       4129.955078125, 4130.3623046875, 4130.70166015625, 4131.109375, 4131.5166015625, 4131.923828125,
                       4132.3310546875, 4132.73876953125, 4133.14599609375, 4133.55322265625, 4133.96044921875,
                       4134.3681640625, 4134.775390625, 4135.1826171875, 4135.58984375, 4135.92919921875,
                       4136.3369140625, 4136.744140625, 4137.1513671875, 4137.55859375, 4137.96630859375,
                       4138.37353515625, 4138.78076171875, 4139.18798828125, 4139.595703125, 4140.0029296875,
                       4140.41015625, 4140.74951171875, 4141.15673828125, 4141.564453125, 4141.9716796875,
                       4142.37890625, 4142.7861328125, 4143.19384765625, 4143.60107421875, 4144.00830078125,
                       4144.41552734375, 4144.8232421875, 4145.16259765625, 4145.56982421875, 4145.84130859375,
                       4145.9091796875, 4145.97705078125]

    scan = load_bscan(RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2251\M08\BSCAN P2251 M08E 2022-Feb-04 094550 [A4127-4145].daq")
    assert np.array_equal(scan.get_channel_axes(Probe.NB1).axial_pos, expected_output)


@pytest.mark.regression_test
def test_large_daq_nb1_data_parsed_correctly():
    """
    Regression test to ensure that the function read_channel_data() reads the channel data as expected for Probe NB1. A daq file is loaded, and the function read_channel_data() is called. The result is compared to expected hardcoded values.

    This test passes if the data values and length produced by read_channel_data() matches the expected hardcoded values, otherwise this test fails.
    """
    input_scan_path = Path(RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2251\T07\BSCAN P2251 T07E 2022-Feb-05 022159 [A8619-8790].daq")
    scan = load_bscan(input_scan_path)
    data = scan.read_channel_data(Probe.CPC)
    assert (data.data[0, 0, 0] == 2045)
    assert (data.data[1, 2, 42] == 2040)
    assert (data.data[-1, -1, -1] == 2050)

    assert data.data.shape == (433, 3600, 1200)
    assert len(data.axes.axial_pos) == 433
    assert len(data.axes.rotary_pos) == 3600
    assert len(data.axes.time_pos) == 1200
