import filecmp
import io
import tqdm

import os
import struct

import pandas as pd

import BScan as Bscan
import re
from tqdm.auto import tqdm
import argparse
from bscan_structure_definitions import *
import file_saving
from pathlib import Path


class BScan_anf(Bscan.BScan):
    def __init__(self, anf_bscan_file: Path):
        """Open a .anf file and returns a BScan class that can be used to interact with the file.
        The BScan file will remain open until BScan.close() is called

        Args:
            anf_bscan_file (Path): Path to the .anf file. File must exist.

        Returns:
            b_scan (BScan): BScan class that can be used to interact with and read from the .anf file.
        """
        self.scan_format = BScan_File_Type.anf
        self.header = ScanHeaderData_anf()
        self.file_positions_and_lengths = ScanHeaderData_anf()
        self.file_positions_and_lengths.GeneratingStation = (1, 0)

        _, filename = os.path.split(anf_bscan_file)
        self.scan_folder_name, _ = filename.split('.')
        try:
            self.scan_type = Anf_File_Scan_Type[re.search(r'(?<=Type )\w+', str(anf_bscan_file.stem)).group(0)]
        except:
            raise ValueError("Scan Type Not Found")
        self.DATA_DTYPE = np.uint8
        super().__init__(anf_bscan_file)
        self.file_type = BScan_File_Type.anf
        # Read the various headers

    def _read_header(self) -> ScanHeaderData_anf:
        '''
        Reads the Scan Header Data.

        This function returns the ScanHeaderData, and stores it internally as a class variable.
        '''
        def fix_channel_labels():
            '''
            If none of the expected channel_labels were found, try to interpret the channel names.

            This function replaces channel_labels with the expected channel label.

            For example "[8] ID1 NB" will be replaced with "ID1 NB"
            '''
            channel_labels = []
            for channel_label in header_data.ChannelLabels:
                if 'ID1 NB' in channel_label:
                    channel_labels.append('ID1 NB')
                elif 'ID2 NB' in channel_label:
                    channel_labels.append('ID2 NB')
                elif 'Circ Shear P/C' in channel_label:
                    channel_labels.append('Circ Shear P/C')
                elif 'Axial Shear P/C' in channel_label:
                    channel_labels.append('Axial Shear P/C')
                elif 'ID NB' in channel_label:
                    channel_labels.append('ID1 NB')
                else:
                    channel_labels.append(channel_label)
            header_data.ChannelLabels = channel_labels

        # Init empty ScanHeaderData class
        header_data = ScanHeaderData_anf()

        # Seek to the beginning of the bscan file
        self.f.seek(0)
        header_data.GeneratingStation = Generating_Station(int.from_bytes(self.f.read(1), self.endian))
        header_data.UnitNumber = int.from_bytes(self.f.read(1), self.endian)
        header_data.Year = int.from_bytes(self.f.read(1), self.endian)
        header_data.Month = int.from_bytes(self.f.read(1), self.endian)
        header_data.Day = int.from_bytes(self.f.read(1), self.endian)
        header_data.ChannelNumber = int.from_bytes(self.f.read(1), self.endian)
        header_data.ChannelLetter = bytes.decode(self.f.read(1))
        header_data.ChannelEnd = Channel_End(int.from_bytes(self.f.read(1), self.endian))
        header_data.ReactorFace = Reactor_Face(int.from_bytes(self.f.read(1), self.endian))
        header_data.InspectionHeadId = rm_white_space(self.f.read(10))
        header_data.Operator = rm_white_space(self.f.read(22))
        header_data.ScanDate = rm_white_space(self.f.read(10))
        header_data.ScanTime = rm_white_space(self.f.read(10))
        # Seek to the Extended Header Position
        # Extended Header Data starts at Byte 70
        self.f.seek(70)
        header_data.ScanType = ScanType(int.from_bytes(self.f.read(1), self.endian))
        header_data.System = int.from_bytes(self.f.read(1), self.endian)
        header_data.FirstAxialPosition = int.from_bytes(self.f.read(2), self.endian)
        header_data.FirstRotaryPosition = int.from_bytes(self.f.read(2), self.endian)
        header_data.LastAxialPosition = int.from_bytes(self.f.read(2), self.endian)
        header_data.LastRotaryPosition = int.from_bytes(self.f.read(2), self.endian)
        header_data.FirstChannel = int.from_bytes(self.f.read(1), self.endian)
        header_data.LastChannel = int.from_bytes(self.f.read(1), self.endian)
        header_data.VersionMajor = int.from_bytes(self.f.read(1), self.endian)
        header_data.VersionMinor = int.from_bytes(self.f.read(1), self.endian)
        header_data.AxialResolutionCnt = int.from_bytes(self.f.read(2), self.endian)
        [header_data.AxialStart_mm] = struct.unpack('f', self.f.read(4))
        [header_data.AxialEnd_mm] = struct.unpack('f', self.f.read(4))
        header_data.AxialPitch = int.from_bytes(self.f.read(1), self.endian)
        _ = self.f.read(1)  # unused
        header_data.PowerCheck = [int.from_bytes(self.f.read(2), self.endian) for b in range(4)]
        header_data.Gain_db = [np.int16(int.from_bytes(self.f.read(2), self.endian)) for b in range(7)]
        header_data.Comment = rm_white_space(self.f.read(60))
        _ = self.f.read(10)  # unused
        header_data.UnfilteredChannelLabels = [rm_white_space(self.f.read(21)) for b in range(16)]
        header_data.ChannelLabels = remove_zeroes_emptystrgs(header_data.UnfilteredChannelLabels)
        if not bool(set(header_data.ChannelLabels) & {'ID1 NB', 'ID2 NB', 'Circ Shear P/C', 'Axial Shear P/C'}):
            fix_channel_labels()
        header_data.ChannelThresholds = [np.int16(int.from_bytes(self.f.read(2), self.endian)) for b in range(16)]
        header_data.ChannelDataOffsets = remove_zeroes_emptystrgs(
            [int.from_bytes(self.f.read(8), self.endian) for b in range(8)])
        header_data.GatesDelay = [[np.int16(int.from_bytes(self.f.read(2), self.endian)) for b in range(5)] for d
                                  in range(5)]
        header_data.GatesRange = [[int.from_bytes(self.f.read(2), self.endian) for b in range(5)] for d in
                                  range(5)]
        header_data.GatesReceiverFreq = [int.from_bytes(self.f.read(2), self.endian) for b in range(5)]
        [header_data.EncoderResolutionAxial] = struct.unpack('f', self.f.read(4))
        [header_data.AxialIncrementalResolution] = struct.unpack('f', self.f.read(4))
        header_data.ScanSensRelNotch = filter_nan(struct.unpack('f', self.f.read(4))[0])
        header_data.FooterOffset = int.from_bytes(self.f.read(8), self.endian)
        header_data.ChannelLabels2 = [None] * 4  # Initialize the list and update the indices.
        for i in range(4):
            header_data.ChannelLabels2[i] = self.f.read(12)

        _ = self.f.read(2)  # unused

        return header_data


    def get_channel_info(self, channel: Probe) -> BScanChannel:
        """Gets metadata about the requested channel.

        Args:
            channel (object): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).

        Returns:
            BScanChannel: Class containing metadata for the requested channel.
        """

        # Initialize the class
        bscan_channel = BScanChannel()

        # Set the label name. Label is the short-form, such as NB1, NB2, CPC, or APC
        bscan_channel.label = channel.name

        # Set the Channel Index and Channel Label.
        # Different Labels may correspond to the came channel_label, such as 'NB1' and 'NB2' both being part of the 'ID1 NB' label in D-Scans.
        bscan_channel.channel_label = self.label_dict[channel]
        bscan_channel.channel_index = self.header.ChannelLabels.index(self.label_dict[channel])

        # Get the data offset and seek to the correct position
        bscan_channel.data_offset = self.header.ChannelDataOffsets[bscan_channel.channel_index]
        self.f.seek(bscan_channel.data_offset)

        # Time domain of A scan within the frame
        bscan_channel.time_range = int.from_bytes(self.f.read(2), self.endian)
        # Number of A-scans in the frame
        bscan_channel.rotary_range = int.from_bytes(self.f.read(2), self.endian)
        # Number of Frames
        bscan_channel.axial_range = int.from_bytes(self.f.read(2), self.endian)

        # adjust the length for interleaved type D scans
        if self.label_dict[channel] == 'ID1 NB' and self.scan_type == Anf_File_Scan_Type.D:
            bscan_channel.axial_range = bscan_channel.axial_range // 2
            slice_i_step = 2
            if channel == Probe.NB1:
                slice_i_start = 0
            elif channel == Probe.NB2:
                slice_i_start = 1
            else:
                raise Exception("Unknown channel for ID1 NB: '%s'" % channel)
        else:
            slice_i_step = 1
            slice_i_start = 0

        # Use the range to calculate the total size of a slice, in bytes
        bscan_channel.slice_databuffer_size_bytes = bscan_channel.time_range * bscan_channel.rotary_range
        bscan_channel.slice_total_size_bytes = bscan_channel.slice_databuffer_size_bytes + 6  # buffer + 4 axialpos + 2rotarypos

        # Calculate the data offsets for each slice start (in bytes)
        bscan_channel.slice_offsets = np.empty(bscan_channel.axial_range, dtype=np.int64)
        for i in range(bscan_channel.axial_range):
            bscan_channel.slice_offsets[i] = bscan_channel.data_offset \
                                             + 6 \
                                             + (i * slice_i_step + slice_i_start) * (
                                                 bscan_channel.slice_total_size_bytes)

        return bscan_channel

    def get_channel_axes(self, channel: Probe, roi: ROI = ROI()) -> BScanAxis:
        """Gets axis information about the selected channel.

        Args:
            channel (object): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
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
        # assert (
        #                self.header.LastRotaryPosition - self.header.FirstRotaryPosition + 1) == channel_info.rotary_range

        full_axis.rotary_pos = self.ROTARY_INDEX_TO_DEGREES_FACTOR * (
                np.arange(0, channel_info.rotary_range) + self.header.FirstRotaryPosition)
        full_axis.rotary_pos = np.remainder(full_axis.rotary_pos,
                                            360.0)  # Get the rotary_position array to loop back to 0 if it crosses 360.0

        gate_start_us = utex_info.gate_start / 1000.0
        gate_width_us = utex_info.gate_width / 1000.0
        digitizer_rate_mhz = utex_info.digitizer_rate * 6.25

        full_axis.time_pos = np.arange(0, channel_info.time_range) * gate_width_us + gate_start_us

        for i, slice_offset in enumerate(channel_info.slice_offsets):
            self.f.seek(slice_offset)
            raw_axial_pos = self.f.read(4)
            axial_pos = struct.unpack('f', raw_axial_pos)[0]
            full_axis.axial_pos[i] = axial_pos

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

        # Get number bytes in each slice
        bytes_to_read = bscan_channel.slice_databuffer_size_bytes

        # Seek to the appropriate position for the target slice
        self.f.seek(bscan_channel.slice_offsets[slice])

        # Read the axial position and add it to the axial_pos list
        axial_pos = struct.unpack('f', self.f.read(4))[0]
        bscan_slice.axial_pos = axial_pos

        # Read rotary position and add it to the rotary_pos list
        rot_pos = int.from_bytes(self.f.read(2), self.endian)
        bscan_slice.rotary_pos = rot_pos

        # Read the channel data in
        bscan_slice.data = np.frombuffer(self.f.read(bytes_to_read), np.uint8, bytes_to_read).reshape(
            (bscan_channel.rotary_range, bscan_channel.time_range))

        return bscan_slice

    def map_channel_label(self, b_scan_channel_label: str) -> list[Probe]:
        if b_scan_channel_label == 'ID1 NB' or b_scan_channel_label == 'ID2 NB':
            if self.scan_type == Anf_File_Scan_Type.D:
                return [Probe.NB1, Probe.NB2]
            else:
                return super().map_channel_label(b_scan_channel_label)
        else:
            return super().map_channel_label(b_scan_channel_label)

    def _read_hardware_info(self) -> UTEXInfo:
        # Init empty Hardware Info class
        hardware_info = UTEXInfo()

        # Seek to the Hardware Info position
        # Hardware Info starts at byte 800
        self.f.seek(800)

        for data_offset in range(12):
            hardware_info.gain[data_offset] = int.from_bytes(self.f.read(1), self.endian)
            hardware_info.pulse_voltage[data_offset] = int.from_bytes(self.f.read(2), self.endian)
            hardware_info.pulse_width[data_offset] = int.from_bytes(self.f.read(2), self.endian)
            hardware_info.low_filter[data_offset] = int.from_bytes(self.f.read(1), self.endian)
            hardware_info.high_filter[data_offset] = int.from_bytes(self.f.read(1), self.endian)

        for data_offset in range(12):
            hardware_info.digitizer_rate[data_offset] = int.from_bytes(self.f.read(1), self.endian)

        for data_offset in range(12):
            hardware_info.digitizer_attenuation[data_offset] = int.from_bytes(self.f.read(1), self.endian)

        for data_offset in range(20):
            hardware_info.gate_start[data_offset] = int.from_bytes(self.f.read(2), self.endian)

        for data_offset in range(20):
            hardware_info.gate_width[data_offset] = int.from_bytes(self.f.read(2), self.endian)

        return hardware_info

    def get_hardware_info(self) -> UTEXInfo:
        """Returns the UTEXInfo / Hardware Info.
        """
        return self.hardware_info

        # TODO: remove this method altogether for the _subset() version. or make this call that

    def read_channel_data(self, channel: Probe, use_progress_bar=False, dtype=np.int16) -> BScanData:
        """Reads every single slice from the bscan file for the specified channel.

        Args:
            channel (object): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
            use_progress_bar (bool, optional): Whether or not to draw a progres bar. Defaults to False. A progress bar is recommended for very large files (>1GB).
            dtype (numerical datatype): The data type of the array to return

        Returns:
            BScanData: A class containing all of the scan data for the specified channel.
        """
        result = BScanData()
        result.axes = self.get_channel_axes(channel)
        result.probe = channel

        # Get information for the selected channel
        b_scan_channel = self.get_channel_info(channel)

        # Initialize the data array
        result.data = np.empty([b_scan_channel.axial_range, b_scan_channel.rotary_range, b_scan_channel.time_range],
                               dtype=dtype)

        if use_progress_bar:
            pbar = tqdm(total=b_scan_channel.axial_range, ascii=True)

        for i in range(b_scan_channel.axial_range):
            result.data[i] = self._read_slice(b_scan_channel, i).data

            if use_progress_bar:
                pbar.update()

        if use_progress_bar:
            pbar.close()

        return result

    def read_channel_data_subset(self, channel: Probe, roi: ROI = ROI(), dtype=np.int16) -> BScanData:
        """Reads a subset of data from the bscan file for the specified channel.

        Args:
            channel (object): Label of the requested channel. The label is expected to be in the short-form (ie: NB1, NB2, CPC, or APC).
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
                               dtype=dtype)

        for slice_i in range(axial_start_index, axial_end_index + 1):
            # use the same _read_slice() since it can't efficiently parse rotary and time anyways
            # TODO: dig in further and see if _read_slice() can in fact be optimized a bit for this case
            bscan_slice = self._read_slice(channel_info, slice_i)
            result.data[slice_i - axial_start_index] = bscan_slice.data[rotary_start_index: rotary_end_index + 1,
                                                       time_start_index: time_end_index + 1]

        return result

    # Overriding parent method
    def save_header_file(self, header_info: ScanHeaderData_anf, file_location, position):
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

    def get_scan_info(self):
        scan_start = self.header.AxialStart_mm
        scan_end = self.header.AxialEnd_mm
        station_name = self.header.GeneratingStation
        return scan_start, scan_end, station_name

    def overwrite_data(self, new_data: dict, new_file_path: Path, file_pos: dict):
        new_file_ref = open(new_file_path, 'r+b')
        # get the type of the .anf file
        if self.scan_type == Anf_File_Scan_Type.D:
            assert Probe.NB1 in new_data.keys(), "Error: nb1 array not found"
            assert Probe.NB2 in new_data.keys(), "Error: nb2 array not found"

            all_nb1_data = new_data[Probe.NB1].data
            all_nb2_data = new_data[Probe.NB2].data
            assert len(all_nb1_data) >= len(all_nb2_data), "Error, Length of NB1 array does not match NB2"
            data_offset = self.header.ChannelDataOffsets[0]

            # Add an offset of 6 for the time range (2 bytes), rotary range (2 bytes) and the axial range (2 bytes) at the
            # start of each probe
            if file_pos[Probe.NB1] == 0:
                new_file_ref.seek(data_offset + 6)
            else:
                new_file_ref.seek(file_pos[Probe.NB2])


            # Find the number of frames for this probe
            number_of_frames = len(all_nb1_data)
            pbar = range(number_of_frames)

            for i in pbar:
                # Read the csv file in as a pd dataframe
                nb1_data = pd.DataFrame(all_nb1_data[i], dtype=np.uint8)
                nb2_data = pd.DataFrame(all_nb2_data[i], dtype=np.uint8)
                # Move the file position ahead another 6 bytes for the axial (4 bytes) and the rotary (2 bytes) positions
                # in between each frame
                # Convert the 2D numpy array into a 1D array of bytes
                nb1_byte_data = nb1_data.to_records(index=False).tobytes()
                nb2_byte_data = nb2_data.to_records(index=False).tobytes()
                # in between each frame
                new_file_ref.seek(new_file_ref.tell() + 6)
                # Write the 1D array of bytes to file
                new_file_ref.write(nb1_byte_data)
                file_pos[Probe.NB1] = new_file_ref.tell()

                # in between each frame
                new_file_ref.seek(new_file_ref.tell() + 6)
                new_file_ref.write(nb2_byte_data)
                file_pos[Probe.NB2] = new_file_ref.tell()

            new_file_ref.close()

        else:
            for probe, data_offset in zip(self.mapped_labels, self.header.ChannelDataOffsets):

                assert probe in new_data.keys(), "Error: " + probe + " array not found"

                # Add an offset of 6 for the time range (2 bytes), rotary range (2 bytes) and the axial range (2 bytes) at the
                # start of each probe
                b_scan_data = new_data[probe].data
                if file_pos[probe] == 0:
                    new_file_ref.seek(data_offset + 6)
                else:
                    new_file_ref.seek(file_pos[probe])

                # Find the number of frames for this probe
                number_of_frames = len(b_scan_data)
                pbar = tqdm(range(number_of_frames), ascii=True)
                for i in pbar:
                    pbar.set_description_str("Importing %s array data into .anf file" % probe.name)

                    # Read the array in as a pd dataframe
                    data = pd.DataFrame(b_scan_data[i], dtype=self.DATA_DTYPE)
                    # Move the file position ahead another 6 bytes for the axial (4 bytes) and the rotary (2 bytes) positions
                    # in between each frame
                    new_file_ref.seek(new_file_ref.tell() + 6)
                    # Convert the 2D numpy array into a 1D array of bytes
                    byte_data = data.to_records(index=False).tobytes()
                    # Write the 1D array of bytes to file
                    new_file_ref.write(byte_data)
                file_pos[probe] = new_file_ref.tell()
            new_file_ref.close()
        return file_pos


def rm_white_space(string: str):
    return string.split(b'\x00')[0].decode('ascii')


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
        help="Path to the BScan File. Expected extension: '.anf' or '.daq'"
    )
    args = parser.parse_args()

    # Parse Arguments
    bscan_file_or_dir = args.bscan_file_or_dir
    BScan_anf(bscan_file_or_dir)
