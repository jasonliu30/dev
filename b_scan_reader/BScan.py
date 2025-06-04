import os
from tqdm.auto import tqdm
import file_saving

from bscan_structure_definitions import *
import bscan_structure_definitions as bscan
from pathlib import Path
import warnings


class BScan(object):

    def __init__(self, bscan_file_or_csv_dir: Path):
        """Open a .anf or .daq file and returns a BScan class that can be used to interact with the file.
        The BScan file will remain open until BScan.close() is called

        Args: bscan_file_or_csv_dir (str): Path to the .anf file, .daq file or the a directory which contains the csv
        export of a bscan file which consists of a Header.csv and folders for each probe type. File or directory must
        exist.

        Returns:
            b_scan (BScan): BScan class that can be used to interact with.
        """
        self.file_type = BScan_File_Type.NA
        self.endian = 'little'
        self.scan_folder_name = bscan_file_or_csv_dir.stem
        self.ROTARY_INDEX_TO_DEGREES_FACTOR = 0.1
        # raise exception if file doesn't exist
        if not os.path.exists(bscan_file_or_csv_dir):
            exception_text = "Input File Not Found: " + str(bscan_file_or_csv_dir)
            raise FileNotFoundError(exception_text)

            # Set File Path and Filename
        self.file_path = bscan_file_or_csv_dir
        self.filename = bscan_file_or_csv_dir.stem

        # Open the File

        self.f = open(bscan_file_or_csv_dir, "rb")
        self.header = self._read_header()
        self.hardware_info = self._read_hardware_info()
        mapped_labels = []
        label_dict = {}
        # Put together a list of all the available mapped labels
        # Also make a dictionary so we can easily go from 'NB1' to 'ID1 NB'
        self.channels_to_export = [bscan.Probe.NB2, bscan.Probe.CPC, bscan.Probe.NB1, bscan.Probe.APC]
        for label in self.header.ChannelLabels:
            mapped_label = self.map_channel_label(label)
            for map_label in mapped_label:
                if map_label not in self.channels_to_export:
                    continue
                # NOTE - i need better terms for these...
                mapped_labels.append(map_label)
                label_dict[map_label] = label
        self.mapped_labels = mapped_labels
        self.label_dict = label_dict

    def __del__(self):
        try:
            self.f.close()
        except:
            pass

    # this method is over-ridden by children
    def _read_header(self):
        return ScanHeaderData

    def get_channel_info(self, channel: Probe) -> BScanChannel:
        return BScanChannel()

    def get_channel_axes(self, channel, roi: ROI = ROI()) -> BScanAxis:
        return BScanAxis()

    def _read_slice(self, bscan_channel: BScanChannel, slice) -> BScanSlice:
        return BScanSlice()

    # this method only gets over-ridden by the anf implementation
    def map_channel_label(self, b_scan_channel_label: str) -> list[Probe]:
        if b_scan_channel_label == 'ID1 NB':
            return [Probe.NB1]
        elif b_scan_channel_label == 'ID2 NB':
            return [Probe.NB2]
        elif b_scan_channel_label == 'Circ Shear P/C':
            return [Probe.CPC]
        elif b_scan_channel_label == 'Axial Shear P/C':
            return [Probe.APC]
        else:
            return [Probe.NA]

    def _read_hardware_info(self) -> UTEXInfo:
        """Returns the ScanHardwareInfo.
        """
        return UTEXInfo()

    def read_channel_data(self, channel: str, use_progress_bar=False, dtype=np.int16) -> BScanData:
        """Reads every single slice from the bscan file for the specified channel.

        Args:
            channel (str): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
            use_progress_bar (bool, optional): Whether or not to draw a progres bar. Defaults to False. A progress bar is recommended for very large files (>1GB).
            dtype (numerical datatype): The data type of the array to return

        Returns:
            BScanData: A class containing all of the scan data for the specified channel.
        """
        # is overridden by child implementations
        pass

    def get_header(self) -> ScanHeaderData():
        """Returns the ScanHeaderData.
        """
        return self.header

    def export_bscan(self, output_dir: Path):
        """Export the currently loaded bscan file to an output directory.

        Args:
            output_dir (str): Root directory to export the header & csv files to.
        """

        # Make output directories if needed
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        output_dir = os.path.join(output_dir, self.scan_folder_name)
        if not os.path.exists(output_dir):
            os.mkdir(output_dir)

        header_row = ["Frame"]
        for channel_label in self.mapped_labels:
            header_row.append(channel_label.name + ".AxialLoc")
            header_row.append(channel_label.name + ".CircLoc")
        header_text = [', '.join(header_row)]

        nb1_axes = self.get_channel_axes(Probe.NB1)
        for i in range(nb1_axes.axial_pos.shape[0]):
            header_row = str(i + 1) + ", "
            for channel_label in self.mapped_labels:
                scan_slice = self.get_channel_axes(channel_label)
                header_row += str(scan_slice.axial_pos[i]) + "," + str(scan_slice.rotary_pos[i]) + ","

            header_text.append(header_row)

        print('Writing Header.csv...')
        self.save_header_file(self.get_header(), output_dir, header_text)
        print('Done')

        # Loop through all of the channel_labels
        for channel_label in self.mapped_labels:

            # Map the channel label.
            mapped_channel_labels = channel_label.name

            # Initialize the b_scan_channel variable using the first label in mapped_channel_labels
            b_scan_channel = self.get_channel_info(channel_label)
            pbar = tqdm(range(b_scan_channel.axial_range), ascii=True)

            for i in pbar:
                save_dir = output_dir / Path(mapped_channel_labels)

                slice = self._read_slice(b_scan_channel, i)

                pbar.set_description_str("Saving " + mapped_channel_labels + ".csv output files")

                file_saving.save_bscan_data(slice.data, save_dir, channel_label, int(i))


    def get_header_text(self):
        """Get the header of the currently loaded BScan file in string-format.
        """
        header_row = ["Frame"]
        for channel_label in self.mapped_labels:
            label=str(channel_label)
            header_row.append(label + ".AxialLoc")
            header_row.append(label + ".CircLoc")
        header_text = [', '.join(header_row)]

        nb1_axes = self.get_channel_axes(Probe.NB1)
        for i in range(nb1_axes.axial_pos.shape[0]):
            header_row = str(i + 1) + ", "
            for channel_label in self.mapped_labels:
                # Map the channel label.
                b_scan_channel = self.get_channel_info(channel_label)
                scan_slice = self._read_slice(b_scan_channel, i)
                header_row += str(scan_slice.axial_pos) + ", " + str(scan_slice.rotary_pos) + ","

            header_text.append(header_row)

        return header_text

    def set_string_bytes(self, string: str, length: int):
        if len(string) == 0:
            return bytes(length)
        if len(string.encode('utf-8')) < length:
            str2 = ('{0: <' + str(length) + '}').format(string)  # adds needed space to the end of str
            return str2.encode('utf-8')
        else:
            return string.encode('utf-8')

    def close(self):
        """Close the file and release it from memory.
        """
        self.f.close()

    def save_header_file(self, header_info: bscan.ScanHeaderData, file_location: str, position):
        pass

    def get_scan_info(self):
        """

        Returns:
            scan_start (float): start location of the scan
            scan_end (float): end location of the scan
            station_name: name of the station the scan was taken in
        """
        scan_start = 0.0
        scan_end = 0.0
        station_name = 'None'
        return scan_start, scan_end, station_name

def rm_white_space(string: str):
    return string.rstrip(b'\x00').decode('utf-8')
