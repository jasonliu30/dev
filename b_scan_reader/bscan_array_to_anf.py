import argparse
import os.path
import shutil

import numpy as np
import pandas as pd
from pathlib import Path
import tqdm
from b_scan_reader import BScan_anf
import re

def export_to_anf(original_anf_file_path: Path, Bscans: dict,save_dir: str,method: str,file_pos:dict):
    #get the type of the .anf file
    type=re.search(r'(?<=Type )\w+', original_anf_file_path.__str__()).group(0)

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
    if not os.path.exists(modified_file_path):
        shutil.copy(original_anf_file_path, modified_file_path)
    anf_export = open(modified_file_path, 'r+b')
    if type=='D':
        assert 'nb1' in Bscans.keys(), "Error: nb1 array not found"
        assert 'nb2' in Bscans.keys(), "Error: nb2 array not found"
        assert len(Bscans['nb1'])>=len(Bscans['nb2']), "Error, Length of NB1 array does not match NB2"
        all_frames=[]
        for frame in range(len(Bscans['nb2'])):
            all_frames.append(Bscans['nb1'][frame])
            all_frames.append(Bscans['nb2'][frame])
        data_offset=original_anf_file.header.ChannelDataOffsets[0]
        # Add an offset of 6 for the time range (2 bytes), rotary range (2 bytes) and the axial range (2 bytes) at the
        # start of each probe
        if file_pos['nb2']==0:
            anf_export.seek(data_offset + 6)
        else:
            anf_export.seek(file_pos['nb2'])

        # Find the number of frames for this probe
        number_of_frames = len(all_frames)
        pbar = tqdm.tqdm(range(number_of_frames), ascii=True)
        for i in pbar:
            pbar.set_description_str("Importing NB1 and NB2 array into .anf file")

            # Read the csv file in as a pd dataframe
            data = pd.DataFrame(all_frames[i],dtype=np.uint8)
            # Move the file position ahead another 6 bytes for the axial (4 bytes) and the rotary (2 bytes) positions
            # in between each frame
            anf_export.seek(anf_export.tell() + 6)
            # Convert the 2D numpy array into a 1D array of bytes
            byte_data = data.to_records(index=False).tobytes()
            #Write the 1D array of bytes to file
            anf_export.write(byte_data)
        file_pos['nb2']=anf_export.tell()
        anf_export.close()
    else:
        for probe, data_offset in zip(original_anf_file.mapped_labels, original_anf_file.header.ChannelDataOffsets):
                        
            assert probe.lower() in Bscans.keys(), "Error: "+probe+" array not found"

            # Add an offset of 6 for the time range (2 bytes), rotary range (2 bytes) and the axial range (2 bytes) at the
            # start of each probe
            if file_pos[probe.lower()]==0:
                anf_export.seek(data_offset+6)
            else:
                anf_export.seek(file_pos[probe.lower()])

            # Find the number of frames for this probe
            number_of_frames = len(Bscans[probe.lower()])
            pbar = tqdm.tqdm(range(number_of_frames), ascii=True)
            for i in pbar:
                pbar.set_description_str("Importing %s array data into .anf file" % probe)
   
                # Read the array in as a pd dataframe
                data = pd.DataFrame(Bscans[probe.lower()][i],dtype=np.uint8)
                # Move the file position ahead another 6 bytes for the axial (4 bytes) and the rotary (2 bytes) positions
                # in between each frame
                anf_export.seek(anf_export.tell() + 6)
                # Convert the 2D numpy array into a 1D array of bytes
                byte_data = data.to_records(index=False).tobytes()
                # Write the 1D array of bytes to file
                anf_export.write(byte_data)
            file_pos[probe.lower()]=anf_export.tell()
        anf_export.close()
    return file_pos



if __name__ == "__main__":  # pragma: no cover
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "original_anf_file",
        help="Path to the original .anf file that was exported"
    )
    parser.add_argument(
        "Bscans",
        help="dict of arrays of bscans"
    )
    parser.add_argument(
        "save_dir",
        help="directory to save results in"
    )
    parser.add_argument(
        "method",
        help="extension to add at the end of .anf file"
    )

    parser.add_argument(
        "file_pos",
        help="dict of the position of each probe, only nb2 is used for type D, nb1, cpc and apc are used for type A"
    )
    args = parser.parse_args()

    # Parse Arguments
    anf_file = (args.original_anf_file)
    Bscans = args.Bscans
    save_dir=args.save_dir
    method=args.method
    file_pos=args.file_pos
    # Check that the input file and output path exist
    if not os.path.exists(anf_file):
        exception_text = "Input File Not Found: " + anf_file
        raise FileNotFoundError(exception_text)

    export_to_anf(anf_file, Bscans,save_dir,method,file_pos)
