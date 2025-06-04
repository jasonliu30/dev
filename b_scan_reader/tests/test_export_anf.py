import pytest
import os
import shutil
from pathlib import Path
from b_scan_reader import BScan_anf
from pathlib import Path


#these tests (and functionality) aren't needed for autoanalysis
pytest.skip(allow_module_level=True)


# Go to the root repo folder
if Path(os.getcwd()) != Path(__file__).parent.parent:
    os.chdir(Path(__file__).parent.parent)

scan_output_dir = Path(r".\scan_test_output")

from file_paths import RAW_DATA_ROOT_DIR


@pytest.fixture()
def cleanup_scan_output_dir():
    """
    deletes scan_output_dir  - meant to clean up a test output after the test has finished running
    """
    yield
    shutil.rmtree(scan_output_dir)


@pytest.mark.unit_test
def test_dscan_export_has_nb2_folder(cleanup_scan_output_dir):
    """
    This test exports a Type-D BScan to file, and checks to make sure that the NB2 folder was created.

    This test passes if the NB2 folder exists, and fails if the NB2 folder does not exist.
    """
    input_scan_path = Path(
        RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2011\R09\BSCAN Type D  R-09 Pickering A Unit-1 east "
        r"30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    scan_basename = input_scan_path.stem
    scan.export_bscan(scan_output_dir)
    nb2_folder = os.path.join(scan_output_dir, scan_basename, 'NB2')
    assert os.path.exists(nb2_folder)


@pytest.mark.unit_test
def test_ascan_export_has_no_nb2_folder(cleanup_scan_output_dir):
    """
    This test exports a Type-A BScan to file, and checks to make sure that the NB2 folder was not created.

    This test passes if the NB2 folder does not exist, and fails if the NB2 folder exists.
    """
    input_scan_path = Path(
        RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east "
        r"30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    scan_basename = os.path.basename(os.path.splitext(input_scan_path)[0])
    scan.export_bscan(scan_output_dir)
    nb2_folder = os.path.join(scan_output_dir, scan_basename, 'NB2')
    assert not os.path.exists(nb2_folder)


@pytest.mark.unit_test
def test_dscan_export_has_header_file(cleanup_scan_output_dir):
    """
    This test ensures that the Header file was created by export_bscan()

    This test passes if Header.csv exists in the expected location after running export_bscan(). This test fails if Header.csv does not exist.
    """
    input_scan_path = Path(
        RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2011\R09\BSCAN Type A  R-09 Pickering A Unit-1 east "
        r"30-Jan-2020 105718 [A2536-2556][R1380-1800].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    scan_basename = os.path.basename(os.path.splitext(input_scan_path)[0])
    scan.export_bscan(scan_output_dir)
    header_file = os.path.join(scan_output_dir, scan_basename, 'Header.csv')
    assert os.path.exists(header_file)


@pytest.mark.slow  # Takes ~18 minutes to run
@pytest.mark.unit_test
def test_export_large_scan(cleanup_scan_output_dir):
    """
    This test exports a large BScan file, and checks to make sure that the header file and NB1 files were created correctly.

    This test passes if Header.csv and NB1_1.csv exist. This test fails if either file do not exist.
    """
    input_scan_path = Path(
        RAW_DATA_ROOT_DIR / r"Scan Data - Darlington\D2011\D06\BSCAN Type A  D-06 Darlington Unit-1 west "
        r"20-Feb-2021 111331 [A9025-9165][R0-3599].anf")
    scan = BScan_anf.BScan_anf(input_scan_path)
    scan.export_bscan(scan_output_dir)
    scan_basename = os.path.basename(os.path.splitext(input_scan_path)[0])
    header_file = os.path.join(scan_output_dir, scan_basename, 'header.csv')
    nb1_first_file = os.path.join(scan_output_dir, scan_basename, 'NB1', 'NB1_1.csv')
    assert os.path.exists(header_file)
    assert os.path.exists(nb1_first_file)


# test a bunch of reference exports against the bscan class's export
ref_folder = "tests/reference_exports/anf Exports"
base_path = Path(RAW_DATA_ROOT_DIR / r"Scan Data - Pickering\P2011\R09")


@pytest.mark.unit_test
@pytest.mark.parametrize("scan_output_folder", os.listdir(ref_folder))
def test_scans_match_reference_files_parameterized(scan_output_folder):
    """
    This test exports a BScan to file, and compares all of the output files to expected reference files.

    This test passes if all of the files exported by export_bscan() match their respective reference files.
    This test fails if any exported files are different from their respective reference files.
    """
    scan = BScan_anf.BScan_anf(base_path / (scan_output_folder + '.anf'))

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
