import pytest
import os
import math
import numpy as np
from pathlib import Path
from copy import deepcopy
from b_scan_reader import BScan_anf
from b_scan_reader.bscan_structure_definitions import *
from b_scan_reader.BScan_reader import load_bscan

FLOAT_RELATIVE_TOLERANCE = 1e-5

# Go to the root repo folder
if Path(os.getcwd()) != Path(__file__).parent.parent:
    os.chdir(Path(__file__).parent.parent)

from file_paths import RAW_DATA_ROOT_DIR


@pytest.mark.regression_test
def test_dscan_header_parsed_correctly():
    """
    Regression test to ensure that the function get_header() reads the file header as expected. An anf file is loaded, and the function get_header() is called. The result is compared to expected hardcoded values.

    This test passes if the header produced by get_header() matches the expected hardcoded values, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    header = scan.get_header()

    assert header.GeneratingStation == Generating_Station.Pickering_A
    assert header.UnitNumber == 1
    assert header.Year == 20  # todo: should this be 2020?
    assert header.Month == 1
    assert header.Day == 30
    assert header.ChannelNumber == 9
    assert header.ChannelLetter == 'R'
    assert header.ChannelEnd == Channel_End.Inlet
    assert header.ReactorFace == Reactor_Face.East
    assert header.InspectionHeadId == 'PS53/H31H'
    assert header.Operator == 'DAYs            '
    assert header.ScanDate == '30-Jan-20'
    assert header.ScanTime == '10:57:18'


@pytest.mark.regression_test
def test_dscan_extendedHeader_parsed_correctly():
    """
    Regression test to ensure that the function get_header() reads the extended header expected. An anf file is loaded, and the function get_header() is called. The result is compared to expected hardcoded values.

    This test passes if the extended header produced by get_header() matches the expected hardcoded values, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    header = scan.get_header()

    assert header.ScanType == ScanType.Circumferential_B_Scan
    assert header.FirstAxialPosition == 2536
    assert header.FirstRotaryPosition == 1380
    assert header.LastAxialPosition == 2556
    assert header.LastRotaryPosition == 1800
    assert header.FirstChannel == 1
    assert header.LastChannel == 1
    assert header.VersionMajor == 4
    assert header.VersionMinor == 6
    assert header.AxialResolutionCnt == 1
    assert header.AxialStart_mm == 2536
    assert header.AxialEnd_mm == 2556
    assert header.AxialPitch == 1
    assert math.isclose(header.EncoderResolutionAxial, 0.1, rel_tol=FLOAT_RELATIVE_TOLERANCE)
    assert math.isclose(header.AxialIncrementalResolution, 0.1, rel_tol=FLOAT_RELATIVE_TOLERANCE)
    assert header.ScanSensRelNotch == 'NA'
    assert header.PowerCheck == [65535, 65535, 65535, 65535]
    assert header.Gain_db == [-1, -1, -1, -1, -1, -1, -1]
    assert header.Comment == ''
    assert header.ChannelLabels == ['ID1 NB']
    assert header.ChannelThresholds == [-1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, -1]
    assert header.ChannelDataOffsets == [4096]
    assert header.GatesDelay == (-1 * np.ones((5, 5))).astype('int').tolist()
    assert header.GatesRange == (65535 * np.ones((5, 5))).astype('int').tolist()
    assert header.GatesReceiverFreq == [65535, 65535, 65535, 65535, 65535]
    # assert extended_header.channel_labels_2 == [] #TODO: it's ['\x00'] * 12???
    # assert extended_header.footer_offset == 18446744073709500000 #TODO: this changed from previous csv? i think?


# TODO: footer test once we have a file with a footer


@pytest.mark.regression_test
def test_dscan_utex_parsed_correctly():
    """
    Regression test to ensure that the function get_hardware_info() reads the utex information as expected. An anf file is loaded, and the function get_hardware_info() is called. The result is compared to expected hardcoded values.

    This test passes if the utex information produced by get_hardware_info() matches the expected hardcoded values, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    utex = scan.get_hardware_info()

    assert utex.gain == [28, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.pulse_voltage == [200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.pulse_width == [1200, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.low_filter == [0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.high_filter == [10, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.digitizer_rate == [20, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.digitizer_attenuation == [2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.gate_start == [14240, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
    assert utex.gate_width == [8880, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]


@pytest.mark.unit_test
def test_invalid_path_throws_exception():
    """
    Test Overview:
    Failure test to ensure that the software fails as expected when trying to load a file that does not exist.

    Test Requirements:
    This test passes of an exception is raised. This test fails if no exception is raised.

    Test Data:
    The test attempts to load a scan from the path 'garbage_input_path_that_doesnt_exist.anf'
    """

    # TODO: make a specific exception for this
    with pytest.raises(Exception) as e_info:
        scan = BScan_anf.BScan_anf(Path('garbage_input_path_that_doesnt_exist.anf'))


@pytest.mark.regression_test
def test_dscan_data_parsed_correctly():
    """
    Regression test to ensure that the function read_channel_data() reads the channel data as expected. An anf file is loaded, and the function read_channel_data() is called. The result is compared to expected hardcoded values.

    This test passes if the data values and length produced by read_channel_data() matches the expected hardcoded values, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    data = scan.read_channel_data(Probe.NB1)

    assert (data.data[0, 0, 0] == 129)
    assert (data.data[1, 2, 42] == 128)
    assert (data.data[-1, -1, -1] == 127)

    assert data.data.shape == (133, 421, 1110)
    assert len(data.axes.axial_pos) == 133
    assert len(data.axes.rotary_pos) == 421
    assert len(data.axes.time_pos) == 1110


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
def test_dscan_successive_reads_match():
    """
    Regression test to ensure that successive calls of the same functions always return the same results.
    The header, hardware info, and channel data are called multiple times and in different ways.

    This test passes if the values of the header, hardware info, channel data remain unchanged when the functions are called several times. This test fails if any values change.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)

    # For each header type try:
    # 1. Pull the property from the class as a property
    # 2. Call the appropriate accessor
    # 3. Call the private funciton that reads the header from file, creating a deep copy of it.
    # 4. Call the private function a second time.
    # If the deep copy doesn't match, a list may be updating by reference.

    header1 = scan.header
    header2 = scan.get_header()
    header3 = deepcopy(scan._read_header())
    header4 = scan._read_header()
    check_all_properties_equal(header1, header2)
    check_all_properties_equal(header1, header3)
    check_all_properties_equal(header1, header4)

    hw_info1 = scan.hardware_info
    hw_info2 = scan.get_hardware_info()
    hw_info3 = deepcopy(scan._read_hardware_info())
    hw_info4 = scan._read_hardware_info()
    check_all_properties_equal(hw_info1, hw_info2)
    check_all_properties_equal(hw_info1, hw_info3)
    check_all_properties_equal(hw_info1, hw_info4)

    label = Probe.NB1
    data = scan.read_channel_data(label)
    data2 = scan.read_channel_data(label)
    assert np.all(data.data == data2.data)
    assert np.all(data.axes.axial_pos == data2.axes.axial_pos)
    assert np.all(data.axes.rotary_pos == data2.axes.rotary_pos)
    assert np.all(data.axes.time_pos == data2.axes.time_pos)

    assert len(data2.axes.axial_pos) == 133
    assert len(data2.axes.rotary_pos) == 421
    assert len(data2.axes.time_pos) == 1110


@pytest.mark.unit_test
def test_multiple_concurrent_dscan_files():
    """
    Test Overview:
    Test to ensure that loading a D-Type BScan file multiple times and reading the channel data multiple times returns the same results.
    A D-Type anf file is loaded, read_channel_data() is called, and the length of the Axial Position axis is checked. This is then repeated.

    Test Requirements:
    This test passes if the length of the Axial Postion axis equals the expected length both times, otherwise this test fails.

    Test Data:
    The test loads the following BScan file: "Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf"
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1
    data = scan.read_channel_data(label)
    assert len(data.axes.axial_pos) == 133

    scan2 = BScan_anf.BScan_anf(input_scan_path)
    data2 = scan2.read_channel_data(label)
    assert len(data2.axes.axial_pos) == 133


@pytest.mark.unit_test
def test_multiple_concurrent_ascan_files():
    """
    Test Overview:
    Test to ensure that loading a A-Type BScan file multiple times and reading the channel data multiple times returns the same results.
    A D-type anf file is loaded, read_channel_data() is called, and the length of the Axial Position axis is checked. This is then repeated.

    Test Requirements:
    This test passes if the length of the Axial Postion axis equals the expected length both times, otherwise this test fails.

    Test Data:
    The test loads the following BScan file: "Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf"
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1
    data = scan.read_channel_data(label)
    assert len(data.axes.axial_pos) == 132

    scan2 = BScan_anf.BScan_anf(input_scan_path)
    data2 = scan2.read_channel_data(label)
    assert len(data2.axes.axial_pos) == 132


@pytest.mark.regression_test
def test_indexed_rois():
    """
    Regression test to check that regions of interest (ROIs) are created as expected when given various inputs.
    This test checks four pairs of ROIs:
    The first set checks to make sure that an ROI created with no specified inputs is equal to an ROI where every property is explicitly set to None.
    The second set checks to make sure that an ROI created using indices matches an ROI created using the real-world measurements that correspond to those indices.
    The third set check to make sure that an ROI created using indices matches an ROI created using the real-world measurements that are created slightly higher to those indices, to ensure it rounds to the nearest index as expected.
    The fourth set check to make sure that an ROI created using indices matches an ROI created using the real-world measurements that are created slightly lower to those indices, to ensure it rounds to the nearest index as expected.

    This test passes if all four sets of ROIs match as expected. This test fails if the values of any properties in any set do not match.
    """
    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1

    channel = scan.get_channel_info(label)
    axis = scan.get_channel_axes(label)
    # data = scan.read_channel_data(label)

    # Create an ROI and explicitly set all values to none
    roi_None_implicit = BScan_anf.ROI()
    roi_None = BScan_anf.ROI()
    roi_None.axial_start_index = None
    roi_None.axial_end_index = None
    roi_None.rotary_start_index = None
    roi_None.rotary_end_index = None
    roi_None.time_start_index = None
    roi_None.time_end_index = None
    roi_None.axial_start_mm = None
    roi_None.axial_end_mm = None
    roi_None.rotary_start_deg = None
    roi_None.rotary_end_deg = None
    roi_None.time_start_us = None
    roi_None.time_end_us = None

    # TODO: stop checking internal stuff, check output indices
    # Check that calling ROI() with no inputs returns an ROI with nothing set.
    check_all_properties_equal(roi_None_implicit, roi_None)

    roi_indexed = BScan_anf.ROI(axial_start_index=12,
                                axial_end_index=123,
                                rotary_start_index=42,
                                rotary_end_index=49,
                                time_start_index=100,
                                time_end_index=200
                                )

    roi_time_based = BScan_anf.ROI(axial_start_mm=2537.9,
                                   axial_end_mm=2554.6,
                                   rotary_start_deg=142.2,
                                   rotary_end_deg=142.9,
                                   time_start_us=902.24,
                                   time_end_us=1790.24
                                   )

    # Use a lower tolerance for this check.
    # The floating-point accuracy of the axial_pos and rotary_pos value are only accurate to 4 decimal places.
    assert roi_indexed.get_axial_indices(channel, axis) == roi_time_based.get_axial_indices(channel, axis)
    assert roi_indexed.get_rotary_indices(channel, axis) == roi_time_based.get_rotary_indices(channel, axis)
    assert roi_indexed.get_time_indices(channel, axis) == roi_time_based.get_time_indices(channel, axis)

    # check just above the axis positions
    roi_time_based2 = BScan_anf.ROI(axial_start_mm=2537.91,
                                    axial_end_mm=2554.6,
                                    rotary_start_deg=142.21,
                                    rotary_end_deg=142.9,
                                    time_start_us=902.241,
                                    time_end_us=1790.24
                                    )
    assert roi_indexed.get_axial_indices(channel, axis) == roi_time_based2.get_axial_indices(channel, axis)
    assert roi_indexed.get_rotary_indices(channel, axis) == roi_time_based2.get_rotary_indices(channel, axis)
    assert roi_indexed.get_time_indices(channel, axis) == roi_time_based2.get_time_indices(channel, axis)

    # check just under the axis positions
    roi_time_based3 = BScan_anf.ROI(axial_start_mm=2537.89,
                                    axial_end_mm=2554.6,
                                    rotary_start_deg=142.19,
                                    rotary_end_deg=142.9,
                                    time_start_us=902.239,
                                    time_end_us=1790.24
                                    )

    assert roi_indexed.get_axial_indices(channel, axis) == roi_time_based3.get_axial_indices(channel, axis)
    assert roi_indexed.get_rotary_indices(channel, axis) == roi_time_based3.get_rotary_indices(channel, axis)
    assert roi_indexed.get_time_indices(channel, axis) == roi_time_based3.get_time_indices(channel, axis)


@pytest.mark.regression_test
def test_roi_with_same_start_and_end():
    """
    Regression test to ensure calling read_channel_data_subset() with an ROI that was created where the start- and
    end-points for each axis are the same value returns a data subset containing a single data point.
    Two ROIs are tested: one where the start- and end- points for each axis is set using indices, and one where the
    start- and end- points for each axis is set using real-world values.
    read_channel_data_subset() is called with each of the ROIs, and the size of the data subset is checked.

    This test passes if the size of the data subset returend by read_channel_data_subset() is equal to (1, 1, 1), otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1

    # Check using indices
    roi_one_element = BScan_anf.ROI(axial_start_index=10,
                                    axial_end_index=10,
                                    rotary_start_index=25,
                                    rotary_end_index=25,
                                    time_start_index=50,
                                    time_end_index=50
                                    )
    subset_under_test = scan.read_channel_data_subset(label, roi_one_element)
    assert subset_under_test.data.shape == (1, 1, 1)

    # Check using real-world measurements
    roi_one_element = BScan_anf.ROI(axial_start_mm=2540,
                                    axial_end_mm=2540,
                                    rotary_start_deg=150,
                                    rotary_end_deg=150,
                                    time_start_us=1000,
                                    time_end_us=1000
                                    )
    subset_under_test = scan.read_channel_data_subset(label, roi_one_element)
    assert subset_under_test.data.shape == (1, 1, 1)


@pytest.mark.regression_test
def test_data_subset_matches_simple_approach():
    """
    This test checks that read_channel_data_subset() is working as expected. This test creates a data subset by calling read_channel_data_subset() with an ROI created using indices.
    It also creates a second data subset by calling read_channel_data() and then indexing the data array using the same index values.

    This test passes if the shape of the data and length of each axis in the data subset created by read_channel_data_subset() match the data subset that was indexed, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1

    roi = BScan_anf.ROI()
    roi.axial_start_index = 12
    roi.axial_end_index = 123
    roi.rotary_start_index = 42
    roi.rotary_end_index = 49
    roi.time_start_index = 100
    roi.time_end_index = 200

    subset_simple = scan.read_channel_data(label)
    subset_simple.data = subset_simple.data[roi.axial_start_index:roi.axial_end_index + 1,
                         roi.rotary_start_index:roi.rotary_end_index + 1,
                         roi.time_start_index:roi.time_end_index + 1]

    subset_under_test = scan.read_channel_data_subset(label, roi)

    assert subset_simple.data.shape == subset_under_test.data.shape

    assert subset_under_test.data.shape[0] == (roi.axial_end_index - roi.axial_start_index + 1)
    assert subset_under_test.data.shape[1] == (roi.rotary_end_index - roi.rotary_start_index + 1)
    assert subset_under_test.data.shape[2] == (roi.time_end_index - roi.time_start_index + 1)

    assert subset_under_test.axes.axial_pos.shape[0] == (roi.axial_end_index - roi.axial_start_index + 1)
    assert subset_under_test.axes.rotary_pos.shape[0] == (roi.rotary_end_index - roi.rotary_start_index + 1)
    assert subset_under_test.axes.time_pos.shape[0] == (roi.time_end_index - roi.time_start_index + 1)


@pytest.mark.regression_test
def test_data_subset_by_real_world_coordinates_axes_match_roi():
    """
    This test creates an ROI using real-world values and then creates a data subset by calling read_channel_data_subset() using the ROI.

    This test passes if the value of the first and last value of each axis match the start- and end- positions that were used to create the ROI, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1

    roi = BScan_anf.ROI(axial_start_mm=2537.95,
                        axial_end_mm=2554.6,
                        rotary_start_deg=142.2,
                        rotary_end_deg=142.9,
                        time_start_us=902.24,
                        time_end_us=1790.24
                        )

    data = scan.read_channel_data_subset(label, roi)
    axial_tolerance = 0.1
    rotary_rolerance = 0.1
    time_tolerance = 0.1

    print(data.data.shape)

    assert np.isclose(data.axes.axial_pos[0], 2537.95, atol=axial_tolerance)
    assert np.isclose(data.axes.axial_pos[-1], 2554.6, atol=axial_tolerance)

    assert np.isclose(data.axes.rotary_pos[0], 142.2, atol=rotary_rolerance)
    assert np.isclose(data.axes.rotary_pos[-1], 142.9, atol=rotary_rolerance)

    assert np.isclose(data.axes.time_pos[0], 902.24, atol=time_tolerance)
    assert np.isclose(data.axes.time_pos[-1], 1790.24, atol=time_tolerance)


@pytest.mark.regression_test
def test_data_subset_by_roi_without_specifying_channelaxis():
    """
    This test creates an ROI using real-world values, and then creates a data subset using that ROI.

    This test passes if the shape of the data subset matches the expected shape, otherwise this test fails.
    """
    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    label = Probe.NB1

    roi = BScan_anf.ROI(
        axial_start_mm=2537.9,
        axial_end_mm=2554.6,
        rotary_start_deg=142.2,
        rotary_end_deg=142.9,
        time_start_us=902.24,
        time_end_us=1790.24
    )

    subset_data = scan.read_channel_data_subset(label, roi)

    assert subset_data.data.shape == (112, 8, 101)


@pytest.mark.regression_test
def test_data_full_indices_matches_get_data():
    """
    This test checks to make sure that the full channel data is read if read_channel_data_subset() is called without providing an ROI.
    This test calls read_channel_data() and then calls read_channel_data_subset(), where read_channel_data_subset() is called without an ROI.

    This test passes if the shape of output of read_channel_data() matches the shape of the output of read_channel_data_subset(), otherwise this test fails.
    """

    # test a variety of scan types easily
    input_scan_paths = [
        RAW_DATA_ROOT_DIR / Path(
            r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf", )
    ]

    for input_scan_path in input_scan_paths:
        scan = BScan_anf.BScan_anf(input_scan_path)
        label = Probe.NB1

        full_data = scan.read_channel_data(label)
        full_subset = scan.read_channel_data_subset(label)

        assert full_data.data.shape == full_subset.data.shape
        assert full_data.axes.axial_pos.shape == full_subset.axes.axial_pos.shape
        assert full_data.axes.rotary_pos.shape == full_subset.axes.rotary_pos.shape
        assert full_data.axes.time_pos.shape == full_subset.axes.time_pos.shape


@pytest.mark.regression_test
def test_channel_names_as_expected():
    """
    This test checks to make sure that the channel names / probe names (ie: APC, CPC, NB1, NB2) exist in the Type A and Type D scans as expected.

    This test passes if the Type A scan contains: APC, CPC, and NB1 in its label dictionary and list of mapped labels
    AND if the Type D scan contains NB1 and NB2 in its label dictionary and list of mapped labels. This test fails if any of the probes are missing from either file.
    """
    type_a_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    type_d_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan_a = BScan_anf.BScan_anf(type_a_path)
    scan_d = BScan_anf.BScan_anf(type_d_path)

    assert Probe.NB1 in scan_a.label_dict
    assert Probe.CPC in scan_a.label_dict
    assert Probe.APC in scan_a.label_dict

    assert Probe.NB1 in scan_a.mapped_labels
    assert Probe.CPC in scan_a.mapped_labels
    assert Probe.APC in scan_a.mapped_labels

    assert Probe.NB1 in scan_d.label_dict
    assert Probe.NB2 in scan_d.label_dict

    assert Probe.NB1 in scan_d.mapped_labels
    assert Probe.NB2 in scan_d.mapped_labels


@pytest.mark.unit_test
def test_datatypes_are_correct():
    """
    Test Overview:
    This test checks that the attributes of the BScanData class have the correct datatypes after calling read_channel_data() and read_channel_data_subset()

    Test Requirements:
    This test passes if: 1. The datatype of 'data' is np.int16. 2. The datatype of each axis is 'np.float64'. 3. The datatype of a slice is 'np.uint8'.
    This test fails if any datatypes don't match their expected ttype.

    Test Data:
    The test uses the following BScan: "Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf"
    """
    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")

    scan = BScan_anf.BScan_anf(input_scan_path)
    channel = Probe.NB1

    full_data = scan.read_channel_data(channel)
    assert full_data.data.dtype == np.int16
    assert full_data.axes.axial_pos.dtype == np.float64
    assert full_data.axes.rotary_pos.dtype == np.float64
    assert full_data.axes.time_pos.dtype == np.float64

    roi = ROI()
    roi.axial_start_index = 4

    subset_data = scan.read_channel_data_subset(channel, roi)
    assert subset_data.data.dtype == np.int16
    assert subset_data.axes.axial_pos.dtype == np.float64
    assert subset_data.axes.rotary_pos.dtype == np.float64
    assert subset_data.axes.time_pos.dtype == np.float64

    scan_slice = scan._read_slice(scan.get_channel_info(channel), 3)
    assert scan_slice.data.dtype == np.uint8


@pytest.mark.unit_test
def test_read_complete_large_scan():
    """
    Test Overview:
    This test ensures that a very large anf file can be loaded into memory, and the data can be read properly.

    Test Requirements:    
    This test passes if the shape of the data returned by read_channel_data() equals (466, 3600, 1110), otherwise this test fails.

    Test Data:
    This test loads "Scan Data - Darlington\D2011\D06\BSCAN Type A  D-06 Darlington Unit-1 west 20-Feb-2021 111331 [A9025-9165][R0-3599].anf", which is 6.1 GB, and then calls read_channel_data()
    """
    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Darlington\D2011\D06\BSCAN Type A  D-06 Darlington Unit-1 west 20-Feb-2021 111331 [A9025-9165][R0-3599].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    data = scan.read_channel_data(Probe.NB1)
    assert data.data.shape == (466, 3600, 1110)


@pytest.mark.regression_test
def test_get_channel_info_typeA_is_as_expected():
    """
    Regression test to ensure that the function get_channel_info() reads the channel info expected for a Type-A BScan.
    A Type-A anf file is loaded, and the function get_channel_info() is called using the NB1 probe. The result is compared to expected hardcoded values.

    This test passes if the BScanChannel class produced by get_channel_info() contains the expected hardcoded values, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Darlington\D2011\D06\BSCAN Type A  D-06 Darlington Unit-1 west 20-Feb-2021 111331 [A9025-9165][R0-3599].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)

    info = scan.get_channel_info(Probe.NB1)
    assert info.time_range == 1110
    assert info.rotary_range == 3600
    assert info.axial_range == 466

    assert info.slice_databuffer_size_bytes == 3996000
    assert info.slice_total_size_bytes == 3996006
    assert info.slice_offsets.shape == (466,)
    assert info.slice_offsets[0] == 4102
    assert info.slice_offsets[-1] == 1858146892


@pytest.mark.regression_test
def test_get_channel_info_typeD_is_as_expected():
    """
    Regression test to ensure that the function get_channel_info() reads the channel info expected for a Type-D BScan.
    A Type-D anf file is loaded, and the function get_channel_info() is called using the NB1 probe. The result is compared to expected hardcoded values.

    This test passes if the BScanChannel class produced by get_channel_info() contains the expected hardcoded values, otherwise this test fails.
    """

    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)

    info = scan.get_channel_info(Probe.NB1)
    assert info.time_range == 1110
    assert info.rotary_range == 421
    assert info.axial_range == 133

    assert info.slice_databuffer_size_bytes == 467310
    assert info.slice_total_size_bytes == 467316
    assert info.slice_offsets.shape == (133,)
    assert info.slice_offsets[0] == 4102
    assert info.slice_offsets[-1] == 123375526


@pytest.mark.regression_test
def test_typeD_channel_axes():
    """
    This test checks that calling different probes using get_channel_axes() returns BScanChannel objects that have different values for each probe.

    This test calls get_channel_axes() twice, one to read the NB1 probe and once to read the NB2 probe. This test passes if all properties of each class are different from each other, otherwise it fails.
    """
    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)

    axis_nb1 = scan.get_channel_axes(Probe.NB1)
    axis_nb2 = scan.get_channel_axes(Probe.NB2)

    assert not np.any(axis_nb1.axial_pos == axis_nb2.axial_pos)


@pytest.mark.regression_test
def test_roi_errors_using_indices_and_real_world_together():
    """
    Failure test to make sure that an exception is raised when both the index and real-world position of the same axial position of an ROI() are set.
    ROI() classes only allow users to specify the start/end index for each axis, or the start/end position for each axis. Not both.

    This test passes if an exception is raised if the axial_start_mm position of an ROI is set, followed by axial_start_index, and vice versa. This test fails if no exception is raised.
    """
    with pytest.raises(Exception) as e_info:
        roi = ROI()
        roi.axial_start_mm = 7.42
        roi.axial_start_index = 42

    with pytest.raises(Exception) as e_info:
        roi = ROI()
        roi.axial_start_index = 42
        roi.axial_start_mm = 7.42

    # we can set an index to none, then set the real-world version with errors
    roi = ROI()
    roi.axial_start_index = None
    roi.axial_start_mm = 7.42


@pytest.mark.regression_test
def test_roi_errors_using_indices_out_of_range():
    """
    Failure test to make sure that an exception is raised when read_channel_data_subset() is called using an ROI that contains indices that are outside of the file range.

    This test passes if an exception each time read_channel_data_subset() is called with an ROI() that exceeds the file limits. This test fails if no exceptions are raised.
    """
    input_scan_path = RAW_DATA_ROOT_DIR / Path(
        r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east 30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    label = scan.mapped_labels[0]

    with pytest.raises(Exception) as e_info:
        roi = ROI(axial_start_index=-1)
        data = scan.read_channel_data_subset(label, roi)

    with pytest.raises(Exception) as e_info:
        roi = ROI(axial_start_index=99999999)
        data = scan.read_channel_data_subset(label, roi)

    roi = ROI(axial_start_index=2)
    data = scan.read_channel_data_subset(label, roi)

    with pytest.raises(Exception) as e_info:
        roi = ROI(rotary_start_index=-1)
        data = scan.read_channel_data_subset(label, roi)

    with pytest.raises(Exception) as e_info:
        roi = ROI(rotary_start_index=99999999)
        data = scan.read_channel_data_subset(label, roi)

    roi = ROI(rotary_start_index=2)
    data = scan.read_channel_data_subset(label, roi)



test_scan_type_data = [(RAW_DATA_ROOT_DIR /
    r'Scan Data - Darlington\D1341\F11\BSCAN Type A  F-11 Darlington Unit-4 west 17-Feb-2013 074134 [A2891-3073][R0-3599].anf',
         Anf_File_Scan_Type.A),

                       (RAW_DATA_ROOT_DIR /
    r'Scan Data - Darlington\D1341\F11\BSCAN Type D  F-11 Darlington Unit-4 west 17-Feb-2013 074134 [A2891-3073][R0-3599].anf',
                         Anf_File_Scan_Type.D)]


@pytest.mark.regression_test
@pytest.mark.parametrize('scan_type_data', test_scan_type_data)
def test_scan_type(scan_type_data):
    """
    Test to ensure that the scan_type property of the BScan class reflects the actual scan type of the BScan file (Type A or Type D).

    This test passes if the scan_type property matches the actual scan type of the BScan file, and if the axial
    locations match hte expected axial locations of the file. This test fails if the scan_type is incorrect or if the expected
    axial locations are incorrect.
    """
    scan_path = Path(scan_type_data[0])
    expected_anf_scan_type = scan_type_data[1]
    actual_anf_scan_type = load_bscan(scan_path).scan_type
    assert expected_anf_scan_type == actual_anf_scan_type

    scan = load_bscan(Path(scan_path))
    nb1_axial_locations = scan.get_channel_axes(Probe.NB1).axial_pos
    assert nb1_axial_locations[0] == 2891.300048828125
    assert nb1_axial_locations[-1] == 3072.800048828125