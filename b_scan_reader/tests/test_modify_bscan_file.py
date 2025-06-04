import filecmp
import os.path
import shutil
from copy import deepcopy
from pathlib import Path
import numpy as np

import pytest

from b_scan_reader import BScan_reader
from b_scan_reader import bscan_structure_definitions as bssd

from file_paths import RAW_DATA_ROOT_DIR


#these tests (and functionality) aren't needed for autoanalysis
pytest.skip(allow_module_level=True)


# Go to the root repo folder
if Path(os.getcwd()) != Path(__file__).parent.parent:
    os.chdir(Path(__file__).parent.parent)

# Saving files to the D drive because there isn't enough space on the C drive for some larger files
save_dir = Path(r'D:\test_output')
method = 'test_write'
type_a_anf_scan = os.path.join(RAW_DATA_ROOT_DIR, r'Scan Data - Darlington\D1341\F11\BSCAN Type A  F-11 Darlington Unit-4 west ' \
                  r'17-Feb-2013 074134 [A2891-3073][R0-3599].anf')
type_d_anf_scan = os.path.join(RAW_DATA_ROOT_DIR, r'Scan Data - Darlington\D1341\F11\BSCAN Type D  F-11 Darlington Unit-4 west ' \
                  r'17-Feb-2013 074134 [A2891-3073][R0-3599].anf')
daq_scan = os.path.join(RAW_DATA_ROOT_DIR, r"Scan Data - Pickering\P2251\M08\BSCAN P2251 M08E 2022-Feb-04 094550 [A4127-4145].daq")


@pytest.fixture()
def cleanup_scan_output_dir():
    """
    deletes scan_output_dir  - meant to clean up a test output after the test has finished running
    """
    try:
        shutil.rmtree(save_dir)
    except FileNotFoundError:
        pass


def create_new_file(original_file_path: Path, save_dir: Path, method: str, file_type: bssd.BScan_File_Type):
    # Get the name of the original .anf file so that we can use it for our new filename
    filename = original_file_path.stem
    if file_type == bssd.BScan_File_Type.anf:
        extension = ".anf"
    elif file_type == bssd.BScan_File_Type.daq:
        extension = ".daq"
    else:
        raise ValueError(f"Could not identify bscan file type {file_type}")

    # Create a new directory in root to place new files
    export_anf_file_dir = save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Initialize our bscan object so that we know where to put the data in the file from header.ChannelDataOffsets
    copied_bscan_file_path = Path(export_anf_file_dir / Path(filename + '_' + method.lower() + extension))
    if copied_bscan_file_path.exists():
        os.remove(copied_bscan_file_path)
    # Make a copy of the original .anf file so that we can modify the new one
    print("Making a copy of %s..." % filename)
    if not os.path.exists(copied_bscan_file_path):
        shutil.copy(original_file_path, copied_bscan_file_path)
    return copied_bscan_file_path


@pytest.mark.unit_test
def test_type_a_scan_replace_with_one_write(cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for Type A scans. It loads a BScan file, reads the probe data, then writes a new file using the probe data.

    This test passes if the new BScan file matches the original one. This test fails if the original BScan file and new BScan file are different.
    """
    original_file_path = Path(type_a_anf_scan)
    bscan_object = BScan_reader.load_bscan(original_file_path)

    APC = bscan_object.read_channel_data(bssd.Probe.APC, False)
    CPC = bscan_object.read_channel_data(bssd.Probe.CPC, False)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)

    bscan_data_dict = {bssd.Probe.APC: APC, bssd.Probe.CPC: CPC, bssd.Probe.NB1: NB1}
    file_pos = {bssd.Probe.APC: 0, bssd.Probe.CPC: 0, bssd.Probe.NB1: 0}

    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)
    bscan_object.overwrite_data(bscan_data_dict, new_file_path, file_pos)
    assert filecmp.cmp(original_file_path, new_file_path)


@pytest.mark.unit_test
def test_type_a_scan_replace_with_multiple_writes(cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for Type A scans when each probe is replaced individually.
    It loads a BScan file, reads the probe data, then writes a new file using the probe data one probe at a time.

    This test passes if the new BScan file matches the original one. This test fails if the original BScan file and new BScan file are different.
    """
    original_file_path = Path(type_a_anf_scan)
    bscan_object = BScan_reader.load_bscan(original_file_path)

    APC = bscan_object.read_channel_data(bssd.Probe.APC, False)
    CPC = bscan_object.read_channel_data(bssd.Probe.CPC, False)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)

    bscan_objects = {bssd.Probe.APC: APC, bssd.Probe.CPC: CPC, bssd.Probe.NB1: NB1}
    bscan_object_copies = {bssd.Probe.APC: deepcopy(APC), bssd.Probe.CPC: deepcopy(CPC), bssd.Probe.NB1: deepcopy(NB1)}
    file_pos = {bssd.Probe.APC: 0, bssd.Probe.CPC: 0, bssd.Probe.NB1: 0}
    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)

    for frame in range(len(APC.data)):
        for probe in bscan_objects:
            bscan_object_copies[probe].data = bscan_objects[probe].data[frame:frame + 1]
        file_pos = bscan_object.overwrite_data(bscan_object_copies, new_file_path, file_pos)

    assert filecmp.cmp(original_file_path, new_file_path)


@pytest.mark.unit_test
def test_type_d_scan_replace_with_one_write(cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for Type D scans. It loads a BScan file, reads the probe data, then writes a new file using the probe data.

    This test passes if the new BScan file matches the original one. This test fails if the original BScan file and new BScan file are different.
    """
    original_file_path = Path(type_d_anf_scan)
    bscan_object = BScan_reader.load_bscan(original_file_path)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)
    NB2 = bscan_object.read_channel_data(bssd.Probe.NB2, False)
    bscan_data_dict = {bssd.Probe.NB1: NB1, bssd.Probe.NB2: NB2}
    file_pos = {bssd.Probe.NB1: 0, bssd.Probe.NB2: 0}
    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)
    bscan_object.overwrite_data(bscan_data_dict, new_file_path, file_pos)
    assert filecmp.cmp(original_file_path, new_file_path)


@pytest.mark.unit_test
def test_type_d_scan_replace_with_multiple_writes(cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for Type D scans when each probe is replaced individually.
    It loads a BScan file, reads the probe data, then writes a new file using the probe data one probe at a time.

    This test passes if the new BScan file matches the original one. This test fails if the original BScan file and new BScan file are different.
    """
    original_file_path = Path(type_d_anf_scan)
    bscan_object = BScan_reader.load_bscan(original_file_path)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)
    NB2 = bscan_object.read_channel_data(bssd.Probe.NB2, False)
    bscan_data_dict = {bssd.Probe.NB1: NB1, bssd.Probe.NB2: NB2}
    bscan_object_copies = {bssd.Probe.NB1: deepcopy(NB1), bssd.Probe.NB2: deepcopy(NB2)}
    file_pos = {bssd.Probe.NB1: 0, bssd.Probe.NB2: 0}
    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)
    for frame in range(len(NB1.data)):
        for probe in bscan_data_dict:
            bscan_object_copies[probe].data = bscan_data_dict[probe].data[frame:frame + 1]
        file_pos = bscan_object.overwrite_data(bscan_object_copies, new_file_path, file_pos)
    assert filecmp.cmp(original_file_path, new_file_path)


@pytest.mark.unit_test
def test_daq_scan_replace_with_one_write(cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for .daq scans. It loads a BScan file, reads the probe data, then writes a new file using the probe data.

    This test passes if the new BScan file matches the original one. This test fails if the original BScan file and new BScan file are different.
    """
    original_file_path = Path(daq_scan)
    bscan_object = BScan_reader.load_bscan(Path(original_file_path))
    APC = bscan_object.read_channel_data(bssd.Probe.APC, False)
    CPC = bscan_object.read_channel_data(bssd.Probe.CPC, False)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)
    NB2 = bscan_object.read_channel_data(bssd.Probe.NB2, False)
    bscan_data_dict = {bssd.Probe.APC: APC, bssd.Probe.CPC: CPC, bssd.Probe.NB1: NB1, bssd.Probe.NB2: NB2}
    file_pos = {bssd.Probe.APC: 0, bssd.Probe.CPC: 0, bssd.Probe.NB1: 0, bssd.Probe.NB2: 0}
    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)
    bscan_object.overwrite_data(bscan_data_dict, new_file_path, file_pos)
    assert filecmp.cmp(original_file_path, new_file_path)


@pytest.mark.unit_test
def test_daq_scan_replace_with_multiple_writes(cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for .daq scans when each probe is replaced individually.
    It loads a BScan file, reads the probe data, then writes a new file using the probe data one probe at a time.

    This test passes if the new BScan file matches the original one. This test fails if the original BScan file and new BScan file are different.
    """
    original_file_path = Path(daq_scan)
    bscan_object = BScan_reader.load_bscan(original_file_path)
    APC = bscan_object.read_channel_data(bssd.Probe.APC, False)
    CPC = bscan_object.read_channel_data(bssd.Probe.CPC, False)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)
    NB2 = bscan_object.read_channel_data(bssd.Probe.NB2, False)
    bscan_data_dict = {bssd.Probe.APC: APC, bssd.Probe.CPC: CPC, bssd.Probe.NB1: NB1, bssd.Probe.NB2: NB2}
    bscan_object_copies = {bssd.Probe.APC: deepcopy(APC), bssd.Probe.CPC: deepcopy(CPC), bssd.Probe.NB1: deepcopy(NB1),
                           bssd.Probe.NB2: deepcopy(NB2)}
    file_pos = {bssd.Probe.APC: 0, bssd.Probe.CPC: 0, bssd.Probe.NB1: 0, bssd.Probe.NB2: 0}
    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)
    for frame in range(len(NB1.data)):
        for probe in bscan_data_dict:
            bscan_object_copies[probe].data = bscan_data_dict[probe].data[frame:frame + 1]
        file_pos = bscan_object.overwrite_data(bscan_object_copies, new_file_path, file_pos, frame)
    assert filecmp.cmp(original_file_path, new_file_path)


def modify_frame(frame: np.ndarray):
    time_domain_length = frame.shape[0]
    number_of_ascans = frame.shape[1]
    checkerboard = np.indices((time_domain_length, number_of_ascans)).sum(axis=0) % 2
    return frame * checkerboard


scans = [r"\\azu-fsai01\RAW DATA\Scan Data - Pickering\P2251\T07\BSCAN P2251 T07E 2022-Feb-05 020421 [A3065-3090].daq",
         r"\\azu-fsai01\RAW DATA\Scan Data - Pickering\P2251\O05\BSCAN P2251 O05E 2022-Feb-04 203328 [A0076-0121].daq",
         r"\\azu-fsai01\RAW DATA\Scan Data - Pickering\P2251\M11\BSCAN P2251 M11E 2022-Feb-04 144103 [A2370-2550].daq",
         r"\\azu-fsai01\RAW DATA\Scan Data - Pickering\P2251\L12\BSCAN P2251 L12E 2022-Feb-01 223037 [A5002-5030].daq",
         daq_scan]


@pytest.mark.slow  # Takes 5-60mins per file. Total run time for all scans is 1h40m
@pytest.mark.unit_test
@pytest.mark.parametrize('scan', scans)
def test_modified_daq_scan_replace_with_multiple_writes(scan, cleanup_scan_output_dir):
    """
    This test ensures that overwrite_data() works as expected for .daq scans when each probe is replaced individually.
    It loads a BScan file, reads the probe data, modifies it, then writes a new file using the modified probe data one probe at a time.
    It then loads the modified BScan file and checks to make sure the modifeid data matches the expected values.

    This test passes if the modified data written to the BScan file matches the modified data read from the BScan file, otherwise it fails.
    """
    original_file_path = Path(scan)
    bscan_object = BScan_reader.load_bscan(original_file_path)

    APC = bscan_object.read_channel_data(bssd.Probe.APC, False)
    CPC = bscan_object.read_channel_data(bssd.Probe.CPC, False)
    NB1 = bscan_object.read_channel_data(bssd.Probe.NB1, False)
    NB2 = bscan_object.read_channel_data(bssd.Probe.NB2, False)

    bscan_data_dict = {bssd.Probe.APC: APC, bssd.Probe.CPC: CPC, bssd.Probe.NB1: NB1, bssd.Probe.NB2: NB2}
    bscan_object_copies = {bssd.Probe.APC: deepcopy(APC), bssd.Probe.CPC: deepcopy(CPC), bssd.Probe.NB1: deepcopy(NB1),
                           bssd.Probe.NB2: deepcopy(NB2)}
    file_pos = {bssd.Probe.APC: 0, bssd.Probe.CPC: 0, bssd.Probe.NB1: 0, bssd.Probe.NB2: 0}
    new_file_path = create_new_file(original_file_path, save_dir, method, bscan_object.file_type)

    for frame in range(len(NB1.data)):
        for probe in bscan_data_dict:
            new_data = bscan_data_dict[probe].data[frame:frame + 1]
            new_data[0] = modify_frame(new_data[0])
            bscan_object_copies[probe].data = new_data
        file_pos = bscan_object.overwrite_data(bscan_object_copies, new_file_path, file_pos, frame)

    # check if the file matches the data
    modified_bscan_object = BScan_reader.load_bscan(new_file_path)
    for probe in bscan_data_dict:
        for frame in range(len(bscan_data_dict[probe].data)):
            bscan_data_dict[probe].data[frame] = modify_frame(bscan_data_dict[probe].data[frame])
        new_data = modified_bscan_object.read_channel_data(probe, False)
        assert (np.array_equal(new_data.data, bscan_data_dict[probe].data))
