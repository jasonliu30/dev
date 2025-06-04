import argparse
import os.path

import BScan
import BScan_anf
import BScan_daq
import BScan_csv

if __name__ == "__main__":  # pragma: no cover
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "input_file",
        help="Path to the BScan file or directory. Expected extension: '.anf','.daq' or a directory from a BScan export"
    )
    parser.add_argument(
        "output_folder",
        help="Root location to save the BScan Export. \
            A folder will be created in this path with the same name as the BScan File."
    )
    args = parser.parse_args()

    # Parse Arguments
    b_scan_file_or_dir = args.input_file
    output_folder = args.output_folder

    # Check that the input file and output path exist
    if not os.path.exists(b_scan_file_or_dir):
        exception_text = "Input File Not Found: " + b_scan_file_or_dir
        raise FileNotFoundError(exception_text)

    file_extension = os.path.splitext(b_scan_file_or_dir)[1]
    if file_extension == '.anf':
        b_scan = BScan_anf.BScan_anf(b_scan_file_or_dir)
    elif file_extension == '.daq':
        b_scan = BScan_daq.BScan_daq(b_scan_file_or_dir)
    elif os.path.isdir(b_scan_file_or_dir):
        b_scan = BScan_csv.BScan_csv(b_scan_file_or_dir)
    else:
        exception_text = b_scan_file_or_dir + " is not a .daq or .anf file."
        raise FileNotFoundError(exception_text)

    # Create the new folder within the output directory
    if not os.path.isdir(output_folder):
        try:
            os.mkdir(output_folder)
        except OSError:
            output_folder = os.path.join(os.getcwd(), output_folder)
            os.mkdir(output_folder)

    print("Opening " + b_scan_file_or_dir + '...')
    
    b_scan.export_bscan(output_folder)
