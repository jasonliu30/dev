import os
import struct
import BScan as Bscan
import re
import tqdm
import argparse
from bscan_structure_definitions import *
import file_saving
import pandas as pd


class BScan_csv(Bscan.BScan):
    def __init__(self, csv_dir: str):
        """Open a dirctory containing the csv files exported from a csv file return a BScan class that can be used to
        interact with.

        Args:
            Path to directory containing exported csv files

        Returns:
            b_scan (BScan): BScan class that can be used to interact with.
        """
        self.scan_dir = csv_dir
        _, self.scan_folder_name = os.path.split(csv_dir)
        try:
            self.scan_type = Anf_File_Scan_Type[re.search(r'(?<=Type )\w+', str(self.scan_folder_name)).group(0)]
        except:
            raise ValueError("Scan Type Not Found")
        self.header_file_path = os.path.join(csv_dir, "Header.csv")
        if not os.path.exists(self.header_file_path):
            raise FileNotFoundError("Header file not found")
        self.DATA_DTYPE = np.uint8
        super().__init__(self.header_file_path)
        # Read the various headers

    def _read_header(self) -> ScanHeaderData_anf:
        '''
        Reads the Scan Header Data.

        This function returns the ScanHeaderData, and stores it internally as a class variable.
        '''

        # Init empty ScanHeaderData class
        header_data = ScanHeaderData_anf()

        # Seek to the beginning of the bscan file
        self.raw_header_csv_contents = self.f.readlines()
        self.header_csv_contents = []
        for line in self.raw_header_csv_contents:
            stripped_string = line.rstrip(b'\r\n').decode("utf-8")
            if stripped_string:
                self.header_csv_contents.append(stripped_string)

        header_data.GeneratingStation = Generating_Station[self.header_csv_contents[1].replace(' ', '_')]
        header_data.UnitNumber = int(self.header_csv_contents[3])
        header_data.Year = int(self.header_csv_contents[5])
        header_data.Month = int(self.header_csv_contents[7])
        header_data.Day = int(self.header_csv_contents[9])
        header_data.ChannelNumber = int(self.header_csv_contents[11])
        header_data.ChannelLetter = self.header_csv_contents[13]
        header_data.ChannelEnd = Channel_End[self.header_csv_contents[15]]
        header_data.ReactorFace = Reactor_Face[self.header_csv_contents[17]]
        header_data.InspectionHeadId = self.header_csv_contents[19]
        header_data.ScanDate = self.header_csv_contents[22]
        header_data.ScanTime = self.header_csv_contents[24]
        header_data.ScanType = ScanType[self.header_csv_contents[26].replace(' ', '_')]
        header_data.FirstAxialPosition = int(self.header_csv_contents[28])
        header_data.FirstRotaryPosition = int(self.header_csv_contents[30])
        header_data.LastAxialPosition = int(self.header_csv_contents[32])
        header_data.LastRotaryPosition = int(self.header_csv_contents[34])
        header_data.FirstChannel = int(self.header_csv_contents[36])
        header_data.LastChannel = int(self.header_csv_contents[38])
        header_data.VersionMajor = int(self.header_csv_contents[40])
        header_data.VersionMinor = int(self.header_csv_contents[42])
        header_data.AxialResolutionCnt = int(self.header_csv_contents[44])
        header_data.AxialStart_mm = float(self.header_csv_contents[46])
        header_data.AxialEnd_mm = float(self.header_csv_contents[48])
        header_data.AxialPitch = int(self.header_csv_contents[50])
        header_data.PowerCheck = [int(x) for x in (self.header_csv_contents[52]).split(', ')]
        header_data.Gain_db = [int(x) for x in (self.header_csv_contents[54]).split(', ')]
        header_data.ChannelLabels = self.header_csv_contents[57].split(', ')
        header_data.ChannelThresholds = [np.int16(x) for x in (self.header_csv_contents[59]).split(', ')]
        header_data.ChannelDataOffsets = [int(x) for x in (self.header_csv_contents[61]).split(', ')]
        header_data.GatesDelay.append([int(x) for x in (self.header_csv_contents[63]).split(', ')])
        header_data.GatesDelay.append([int(x) for x in (self.header_csv_contents[65]).split(', ')])
        header_data.GatesDelay.append([int(x) for x in (self.header_csv_contents[66]).split(', ')])
        header_data.GatesDelay.append([int(x) for x in (self.header_csv_contents[64]).split(', ')])
        header_data.GatesDelay.append([int(x) for x in (self.header_csv_contents[67]).split(', ')])
        header_data.GatesRange.append([int(x) for x in (self.header_csv_contents[69]).split(', ')])
        header_data.GatesRange.append([int(x) for x in (self.header_csv_contents[70]).split(', ')])
        header_data.GatesRange.append([int(x) for x in (self.header_csv_contents[71]).split(', ')])
        header_data.GatesRange.append([int(x) for x in (self.header_csv_contents[72]).split(', ')])
        header_data.GatesRange.append([int(x) for x in (self.header_csv_contents[73]).split(', ')])
        header_data.GatesReceiverFreq = [int(x) for x in (self.header_csv_contents[75]).split(', ')]
        header_data.EncoderResolutionAxial = float(self.header_csv_contents[77])
        header_data.AxialIncrementalResolution = float(self.header_csv_contents[79])
        header_data.ScanSensRelNotch = self.header_csv_contents[81]
        header_data.FooterOffset = int(self.header_csv_contents[83])

        return header_data

    def get_channel_info(self, channel: str) -> BScanChannel:
        """Gets metadata about the requested channel.

        Args:
            channel (str): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).

        Returns:
            BScanChannel: Class containing metadata for the requested channel.
        """
        # Initialize the class
        bscan_channel = BScanChannel()

        # Set the label name. Label is the short-form, such as NB1, NB2, CPC, or APC
        bscan_channel.label = channel

        # Set the Channel Index and Channel Label.
        # Different Labels may correspond to the came channel_label, such as 'NB1' and 'NB2' both being part of the 'ID1 NB' label in D-Scans.
        bscan_channel.channel_label = self.label_dict[channel]
        bscan_channel.channel_index = self.header.ChannelLabels.index(self.label_dict[channel])

        # Get the data offset and seek to the correct position
        bscan_channel.data_offset = self.header.ChannelDataOffsets[bscan_channel.channel_index]
        probe_folder = os.path.join(self.scan_dir, bscan_channel.label)
        first_data_file = open(os.path.join(probe_folder, bscan_channel.label + '_1.csv'))
        first_data_file_contents = first_data_file.readlines()
        first_data_file.close()

        # Time domain of A scan within the frame
        bscan_channel.time_range = len(first_data_file_contents[0].split(','))

        # Number of A-scans in the frame
        bscan_channel.rotary_range = len(first_data_file_contents)

        # Number of Frames
        bscan_channel.axial_range = len(os.listdir(probe_folder))

        # adjust the length for interleaved type D scans
        if self.label_dict[channel] == 'ID1 NB' and self.scan_type == Anf_File_Scan_Type.D:
            bscan_channel.axial_range = bscan_channel.axial_range // 2
            slice_i_step = 2
            if channel == 'NB1':
                slice_i_start = 0
            elif channel == 'NB2':
                slice_i_start = 1
            else:
                raise Exception("Unknown channel for ID1 NB: '%s'" % channel)
        else:
            slice_i_step = 1
            slice_i_start = 0

        return bscan_channel

    def get_channel_axes(self, channel, roi: ROI = ROI()) -> BScanAxis:
        """Gets axis information about the selected channel.

        Args:
            channel (str): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
            roi (ROI, optional): ROI describing the axis limits. Defaults to ROI().

        Returns:
            axis (BScanAxis): Class containing axis information about the selecetd channel
        """

        # Get information about the selected channel
        channel_info = self.get_channel_info(channel)

        # Initialize empty BScan Axis class
        full_axis = BScanAxis()

        utex_info = self.get_hardware_info()[channel_info.channel_index]

        full_axis.axial_pos = np.zeros(channel_info.axial_range)
        full_axis.rotary_pos = np.zeros(channel_info.rotary_range, dtype=int)
        assert (
                       self.header.LastRotaryPosition - self.header.FirstRotaryPosition + 1) == channel_info.rotary_range

        gate_start_us = utex_info.gate_start / 1000.0
        gate_width_us = utex_info.gate_width / 1000.0
        digitizer_rate_mhz = utex_info.digitizer_rate * 6.25

        full_axis.time_pos = np.arange(0, channel_info.time_range) * gate_width_us + gate_start_us

        for i, frame_info in enumerate(self.header_csv_contents[107:]):
            full_axis.axial_pos[i] = float((frame_info.split(',')[self.mapped_labels.index(channel) * 2 + 1]))
            full_axis.rotary_pos[i] = int((frame_info.split(',')[self.mapped_labels.index(channel) * 2 + 2]))

        axis = full_axis

        # crop the axes according to the specified ROI
        axial_indices = roi.get_axial_indices(channel_info, full_axis)
        axis.axial_pos = axis.axial_pos[slice(axial_indices[0], axial_indices[1] + 1)]
        rotary_indices = roi.get_rotary_indices(channel_info, full_axis)
        axis.rotary_pos = axis.rotary_pos[slice(rotary_indices[0], rotary_indices[1] + 1)]
        time_indices = roi.get_time_indices(channel_info, full_axis)
        axis.time_pos = axis.time_pos[slice(time_indices[0], time_indices[1] + 1)]

        return axis

    def _read_slice(self, bscan_channel: BScanChannel, slice) -> BScanSlice:
        """Reads a single slice from the BScan File.

        Args:
            channel (BScanChannel): BScanChannel class for the desired channel.
            slice (int): Slice Number.

        Returns:
            channel_data: Class containing the data for the specified slice
        """

        # Initialize the data array
        bscan_slice = BScanSlice()
        bscan_slice.data = np.empty([bscan_channel.rotary_range, bscan_channel.time_range], dtype=self.DATA_DTYPE)

        # Read the axial position and add it to the axial_pos list
        scan_data = self.get_channel_axes(bscan_channel.label)
        bscan_slice.axial_pos = scan_data.axial_pos[slice]

        # Read rotary position and add it to the rotary_pos list
        bscan_slice.rotary_pos = scan_data.rotary_pos[slice]

        # Read the channel data in
        df = pd.read_csv(
            os.path.join(self.scan_dir, bscan_channel.label, bscan_channel.label + '_' + str(slice + 1) + '.csv'),
            header=None)
        bscan_slice.data = df.to_numpy(dtype=self.DATA_DTYPE)

        return bscan_slice

    def map_channel_label(self, b_scan_channel_label: str):
        if b_scan_channel_label == 'ID1 NB' or b_scan_channel_label == 'ID2 NB':
            if self.scan_type == Anf_File_Scan_Type.D:
                return ['NB1', 'NB2']
            else:
                return ['NB1']
        else:
            return super().map_channel_label(b_scan_channel_label)

    def _read_hardware_info(self) -> UTEXInfo:
        # Init empty Hardware Info class
        hardware_info = UTEXInfo()
        utex_array = []
        for row in self.header_csv_contents[89:101]:
            utex_array.append([int(value) for value in row.split(', ')])
        hardware_info.gain = [gain[0] for gain in utex_array]
        hardware_info.pulse_voltage = [pulse_voltage[1] for pulse_voltage in utex_array]
        hardware_info.pulse_width = [pulse_width[2] for pulse_width in utex_array]
        hardware_info.high_filter = [high_filter[3] for high_filter in utex_array]
        hardware_info.low_filter = [low_filter[4] for low_filter in utex_array]
        hardware_info.digitizer_rate = [digitizer_rate[5] for digitizer_rate in utex_array]
        hardware_info.digitizer_attenuation = [digitizer_attenuation[6] for digitizer_attenuation in utex_array]
        hardware_info.gate_start = [int(gate_start) for gate_start in self.header_csv_contents[102].split(', ')]
        hardware_info.gate_width = [int(gate_start) for gate_start in self.header_csv_contents[104].split(', ')]

        return hardware_info

    def get_hardware_info(self) -> UTEXInfo:
        """Returns the UTEXInfo / Hardware Info.
        """
        return self.hardware_info

        # TODO: remove this method altogether for the _subset() version. or make this call that

    def read_channel_data(self, channel: str, use_progress_bar=False) -> BScanData:
        """Reads every single slice from the bscan file for the specified channel.

        Args:
            channel (str): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
            use_progress_bar (bool, optional): Whether or not to draw a progres bar. Defaults to False. A progress bar is recommended for very large files (>1GB).

        Returns:
            BScanData: A class containing all of the scan data for the specified channel.
        """

        result = BScanData()
        result.axes = self.get_channel_axes(channel)

        # Get information for the selected channel
        b_scan_channel = self.get_channel_info(channel)

        # Initialize the data array
        result.data = np.empty([b_scan_channel.axial_range, b_scan_channel.rotary_range, b_scan_channel.time_range],
                               dtype=self.DATA_DTYPE)

        if use_progress_bar:
            pbar = tqdm.tqdm(total=b_scan_channel.axial_range, ascii=True)

        for i in range(b_scan_channel.axial_range):
            result.data[i] = self._read_slice(b_scan_channel, i).data

            if use_progress_bar:
                pbar.update()

        if use_progress_bar:
            pbar.close()

        return result

    def read_channel_data_subset(self, channel: str, roi: ROI = ROI()) -> BScanData:
        """Reads a subset of data from the bscan file for the specified channel.

        Args:
            channel (str): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
            roi (ROI, optional): ROI describing the axis limits that define the subset. Defaults to ROI().

        Returns:
            result (BScanData): A class containing the data subset for the specified channel.
        """

        # Get information for the selected channel
        channel_info = self.get_channel_info(channel)
        full_axis = self.get_channel_axes(channel)

        axial_start_index, axial_end_index = roi.get_axial_indices(channel_info, full_axis)
        rotary_start_index, rotary_end_index = roi.get_rotary_indices(channel_info, full_axis)
        time_start_index, time_end_index = roi.get_time_indices(channel_info, full_axis)

        result = BScanData()
        result.axes = self.get_channel_axes(channel, roi)

        # Initialize the data array
        result.data = np.empty([axial_end_index - axial_start_index + 1,
                                rotary_end_index - rotary_start_index + 1,
                                time_end_index - time_start_index + 1],
                               dtype=self.DATA_DTYPE)

        for slice_i in range(axial_start_index, axial_end_index + 1):
            # use the same _read_slice() since it can't efficiently parse rotary and time anyways
            # TODO: dig in further and see if _read_slice() can in fact be optimized a bit for this case
            bscan_slice = self._read_slice(channel_info, slice_i)
            result.data[slice_i - axial_start_index] = bscan_slice.data[rotary_start_index: rotary_end_index + 1,
                                                       time_start_index: time_end_index + 1]

        return result

    # Overriding parent method
    def save_header_file(self, header_info, file_location, position):
        hardware_info = self.hardware_info
        utex_info = [
            *zip(
                *[hardware_info.gain, hardware_info.pulse_voltage, hardware_info.pulse_width, hardware_info.high_filter,
                  hardware_info.low_filter, hardware_info.digitizer_rate, hardware_info.digitizer_attenuation])]
        utex_info.insert(0, ['Gain.dB', 'PulseVoltage.V', 'PulseWidth.100ns', 'HighFilter.MHz', 'LowFilter.MHz',
                             'DigitizationRate.MHZ', 'DigitizationAttenuation.dB'])
        utex_string = ''
        for row in utex_info:
            row = [str(i) for i in row]
            utex_string += (', '.join(row) + '\n')
        formatted_header_data = "Station\n" + header_info.GeneratingStation.name.replace("_", " ") + "\n\n" + \
                                "Unit\n" + str(header_info.UnitNumber) + "\n\n" + \
                                "Year\n" + str(header_info.Year) + "\n\n" + \
                                "Month\n" + str(header_info.Month) + "\n\n" + \
                                "Day\n" + str(header_info.Day) + "\n\n" + \
                                "ChannelNumber\n" + str(header_info.ChannelNumber) + "\n\n" + \
                                "ChannelLetter\n" + header_info.ChannelLetter + "\n\n" + \
                                "Channel End\n" + str(header_info.ChannelEnd.name) + "\n\n" + \
                                "Reactor Face\n" + str(header_info.ReactorFace.name) + "\n\n" + \
                                "InspectionHead\n" + str(header_info.InspectionHeadId) + "\n\n" + \
                                "OperatorName\n" + str(header_info.Operator) + "\n\n" + \
                                "Date\n" + str(header_info.ScanDate) + "\n\n" + \
                                "Time\n" + str(header_info.ScanTime) + "\n\n" + \
                                "Scan Type\n" + str(header_info.ScanType.name.replace("_", " ")) + "\n\n" + \
                                "AxialStart\n" + str(header_info.FirstAxialPosition) + "\n\n" + \
                                "RotaryStart\n" + str(header_info.FirstRotaryPosition) + "\n\n" + \
                                "AxialEnd\n" + str(header_info.LastAxialPosition) + "\n\n" + \
                                "RotaryEnd\n" + str(header_info.LastRotaryPosition) + "\n\n" + \
                                "FirstChannel\n" + str(header_info.FirstChannel) + "\n\n" + \
                                "LastChannel\n" + str(header_info.LastChannel) + "\n\n" + \
                                "VersionMajor\n" + str(header_info.VersionMajor) + "\n\n" + \
                                "VersionMinor\n" + str(header_info.VersionMinor) + "\n\n" + \
                                "AxialIncrRes\n" + str(header_info.AxialResolutionCnt) + "\n\n" + \
                                "AxialStartPos.mm\n" + str(header_info.AxialStart_mm) + "\n\n" + \
                                "AxialEndPos.mm\n" + str(header_info.AxialEnd_mm) + "\n\n" + \
                                "AxialPitch.mm\n" + str(header_info.AxialPitch) + "\n\n" + \
                                "PowerCheck.dB\n" + ', '.join([str(pwr) for pwr in header_info.PowerCheck]) + "\n\n" + \
                                "Gain.dB\n" + ', '.join([str(pwr) for pwr in header_info.Gain_db]) + "\n\n" + \
                                "OperatorComment\n" + header_info.Comment + "\n\n" + \
                                "ChannelLabels\n" + ', '.join(header_info.ChannelLabels) + "\n\n" + \
                                "ChannelThresholds\n" + ', '.join(
            [str(thresh) for thresh in header_info.ChannelThresholds]) + "\n\n" + \
                                "BScanChannelOffsets\n" + ', '.join(
            [str(thresh) for thresh in header_info.ChannelDataOffsets]) + "\n\n" + \
                                "GateDelays\n" + file_saving.format_2d_str_array(header_info.GatesDelay) + "\n\n" + \
                                "GateRanges\n" + file_saving.format_2d_str_array(header_info.GatesRange) + "\n\n" + \
                                "ReceiverFreq\n" + ', '.join(
            [str(freq) for freq in header_info.GatesReceiverFreq]) + "\n\n" + \
                                "AxialRes.mmPerBit\n" + str(header_info.EncoderResolutionAxial) + "\n\n" + \
                                "AxialIncRes.mmPerBit\n" + str(header_info.AxialIncrementalResolution) + "\n\n" + \
                                "ScanSensivitity.dB\n" + str(header_info.ScanSensRelNotch) + "\n\n" + \
                                "FileFooterOffset\n" + str(header_info.FooterOffset) + "\n\n" + \
                                "ANDE.Channel.Labels\n" + "\n\n" + \
                                "File.Flip.Ind\n" + str(0) + "\n\n" + \
                                "UTEX\n" + utex_string + "\n\n" + \
                                "GateStart.ns\n" + ', '.join(
            [str(freq) for freq in hardware_info.gate_start]) + "\n\n" + \
                                "GateWidth.ns\n" + ', '.join(
            [str(freq) for freq in hardware_info.gate_width]) + "\n\n" + \
                                "Locations\n" + '\n'.join(position)
        header_file = open(os.path.join(file_location, 'Header.csv'), 'w')
        header_file.write(formatted_header_data)
        header_file.close()
        # TODO - axial_pitch is read as u8 but the sample header.csv has it as a float
        # TODO - should the gain, channel threshold, gate delays gate ranges, receiver freq be coerced to -1 from 65535?


def rm_white_space(string: str):
    return string.rstrip(b'\x00').decode('utf-8')


def remove_zeroes_emptystrgs(input_list: list[any]):
    """Removes all zeroes and empty strings from an input list

    Args:
        input_list: List of strings, or list of integers.

    Returns:
        filtered_list: List with zeroes and empty strings removed.
    """
    filtered_list = input_list.copy()
    if all(isinstance(x, int) for x in filtered_list):
        try:
            while True:
                filtered_list.remove(0)
        except ValueError:
            return filtered_list
    if all(isinstance(x, str) for x in filtered_list):
        try:
            while True:
                filtered_list.remove('')
        except ValueError:
            return filtered_list
    else:
        return filtered_list


def filter_nan(number: float):
    return "NA" if math.isnan(number) else str(number)


if __name__ == "__main__":  # pragma: no cover
    # Parse arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "bscan_file_or_dir",
        help="Path to the BScan export directory containing .csv files"
    )
    args = parser.parse_args()

    # Parse Arguments
    bscan_file_or_dir = args.bscan_file_or_dir
    BScan_csv(bscan_file_or_dir)
