import pytest
import os
import shutil
from pathlib import Path
from b_scan_reader import BScan_daq


#these tests (and functionality) aren't needed for autoanalysis
pytest.skip(allow_module_level=True)


# Go to the root repo folder
if Path(os.getcwd()) != Path(__file__).parent.parent:
    os.chdir(Path(__file__).parent.parent)

input_scan_path = Path(
    r"\\azu-fsai01\RAW DATA\Scan Data - Pickering\P2251\L12\BSCAN P2251 L12E 2022-Feb-01 223037 [A5002-5030].daq")
scan_output_dir = Path("scan_test_output")

print(os.getcwd())


@pytest.fixture()
def cleanup_scan_output_dir():
    """
    deletes scan_output_dir  - meant to clean up a test output after the test has finished running
    """
    yield
    shutil.rmtree(scan_output_dir)
    pass


@pytest.mark.unit_test
def test_daq_export_Only_has_nb1_nb2_apc_cpc_folders(cleanup_scan_output_dir):
    """
    This test exports a .daq BScan to file and checks to make sure all of the expected folders exist.

    This test passes if the APC, CPC, NB1, and NB2 folders all exist. This test fails if any folder is missing.
    """
    scan = BScan_daq.BScan_daq(input_scan_path)
    scan.export_bscan(scan_output_dir)
    scan_folder = os.path.join(scan_output_dir, input_scan_path.stem)
    channel_folders = os.listdir(scan_folder)
    assert channel_folders.sort() == scan.channels_to_export.sort()


@pytest.mark.unit_test
def test_daq_export_has_header_file(cleanup_scan_output_dir):
    """
    This test ensures that the Header file was created by export_bscan()

    This test passes if Header.csv exists in the expected location after running export_bscan(). This test fails if Header.csv does not exist.
    """
    scan = BScan_daq.BScan_daq(input_scan_path)
    scan.export_bscan(scan_output_dir)
    header_file = scan_output_dir / input_scan_path.stem / 'Header.csv'
    print(header_file)
    assert os.path.exists(header_file)


# test a bunch of reference exports against the bscan class's export
ref_folder = "tests/reference_exports/daq Exports"


@pytest.mark.slow  # takes ~9 minutes to run
@pytest.mark.regression_test
@pytest.mark.parametrize("scan_output_folder", os.listdir(ref_folder))
def test_scans_match_reference_files_parameterized(scan_output_folder):
    """
    This test exports a BScan to file, and compares all of the output files to expected reference files.

    This test passes if all of the files exported by export_bscan() match their respective reference files.
    This test fails if any exported files are different from their respective reference files.
    """
    scan = BScan_daq.BScan_daq(input_scan_path)

    scan.export_bscan(scan_output_dir)
    test_scan_folder_path = os.path.join(scan_output_dir, scan_output_folder)

    reference_scan_folder = os.path.join(ref_folder, scan_output_folder)

    for r, d, f in os.walk(reference_scan_folder):
        for file in f:
            sub_folder = r.replace(reference_scan_folder, "").strip('\\')
            ref_path = os.path.join(r, file)
            test_path = os.path.join(os.path.join(test_scan_folder_path, sub_folder), file)

            ref_content = open(ref_path).read()
            test_content = open(test_path).read()
            assert ref_content == test_content

    shutil.rmtree(scan_output_dir)
