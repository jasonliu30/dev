import argparse
import os.path
import shutil

import numpy as np
import pandas as pd
from numpy import genfromtxt
import tqdm
from b_scan_reader import BScan_anf
import re

def export_to_anf(original_anf_file_path: str, csv_dict: dict,save_dir: str,method: str):
    #get the type of the .anf file
    type=re.search(r'(?<=Type )\w+', original_anf_file_path).group(0)

    # Get the name of the original .anf file so that we can use it for our new filename
    _, anf_filename = os.path.split(original_anf_file_path)
    # Remove the file extension so we can use it to name our new file
    anf_filename_ext_removed, _ = anf_filename.split('.')
    # Create a new directory in root to place new files
    export_anf_file_dir = save_dir
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    # Initialize our bscan object so that we know where to put the data in the file from header.ChannelDataOffsets
    original_anf_file = BScan_anf.BScan_anf(original_anf_file_path)
    modified_file_path = os.path.join(export_anf_file_dir, anf_filename_ext_removed + '_'+method.lower()+'.anf')
    # Make a copy of the original .anf file so that we can modify the new one
    print("Making a copy of %s..." % anf_filename)
    shutil.copy(original_anf_file_path, modified_file_path)
    anf_export = open(modified_file_path, 'r+b')
    if type=='D':
        assert 'nb1' in csv_dict.keys(), "Error: nb1 csvs not found"
        assert 'nb2' in csv_dict.keys(), "Error: nb2 csvs not found"

        all_frames=[]
        for file in range(len(csv_dict['nb2'])):
            all_frames.append(csv_dict['nb1'][file])
            all_frames.append(csv_dict['nb2'][file])
        data_offset=original_anf_file.header.ChannelDataOffsets[0]
        # Add an offset of 6 for the time range (2 bytes), rotary range (2 bytes) and the axial range (2 bytes) at the
        # start of each probe
        anf_export.seek(data_offset + 6)
        # Find the number of frames for this probe
        number_of_frames = len(all_frames)
        pbar = tqdm.tqdm(range(number_of_frames), ascii=True)
        for i in pbar:
            pbar.set_description_str("Importing NB1 and NB2 data into .anf file")
            # Create the csv file path name
            frame_path = all_frames[i]
            # Read the csv file in as a pd dataframe
            data = pd.read_csv(frame_path, header=None, dtype=np.uint8)
            # Move the file position ahead another 6 bytes for the axial (4 bytes) and the rotary (2 bytes) positions
            # in between each frame
            anf_export.seek(anf_export.tell() + 6)
            # Convert the 2D numpy array into a 1D array of bytes
            byte_data = data.to_records(index=False).tobytes()
            #Write the 1D array of bytes to file
            anf_export.write(byte_data)
        anf_export.close()
    else:
        for probe, data_offset in zip(original_anf_file.mapped_labels, original_anf_file.header.ChannelDataOffsets):
            assert probe.lower() in csv_dict.keys(), "Error: "+probe+" csvs not found"
            # Add an offset of 6 for the time range (2 bytes), rotary range (2 bytes) and the axial range (2 bytes) at the
            # start of each probe
            anf_export.seek(data_offset+6)
            # Find the number of frames for this probe
            number_of_frames = len(csv_dict[probe.lower()])
            pbar = tqdm.tqdm(range(number_of_frames), ascii=True)
            for i in pbar:
                pbar.set_description_str("Importing %s.csv data into .anf file" % probe)
                # Create the csv file path name
                frame_path = csv_dict[probe.lower()][i]
                # Read the csv file in as a pd dataframe
                data = pd.read_csv(frame_path, header=None, dtype=np.uint8)
                # Move the file position ahead another 6 bytes for the axial (4 bytes) and the rotary (2 bytes) positions
                # in between each frame
                anf_export.seek(anf_export.tell() + 6)
                # Convert the 2D numpy array into a 1D array of bytes
                byte_data = data.to_records(index=False).tobytes()
                # Write the 1D array of bytes to file
                anf_export.write(byte_data)
        anf_export.close()



if __name__ == "__main__":  # pragma: no cover
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "original_anf_file",
        help="Path to the original .anf file that was exported"
    )
    parser.add_argument(
        "csv_dict",
        help="dict of lists where each probe should be a key, and each list has the file directory of each frame"
    )
    parser.add_argument(
        "save_dir",
        help="directory to save results in"
    )
    parser.add_argument(
        "method",
        help="extension to add at the end of .anf file"
    )
    args = parser.parse_args()

    # Parse Arguments
    anf_file = args.original_anf_file
    csv_dict = args.csv_dict
    save_dir=args.save_dir
    method=args.method

    # Check that the input file and output path exist
    if not os.path.exists(anf_file):
        exception_text = "Input File Not Found: " + anf_file
        raise FileNotFoundError(exception_text)

    export_to_anf(anf_file, csv_dict,save_dir,method)
