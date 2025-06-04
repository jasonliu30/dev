import filecmp
import io
import os
import re
import struct
import BScan as Bscan
import calendar
import tqdm
from tqdm.auto import tqdm
from bscan_structure_definitions import *
from pathlib import Path
import pandas as pd


class BScan_daq(Bscan.BScan):
    def __init__(self, bscan_file_or_csv_dir: Path):
        """Open a .daq file and returns a BScan class that can be used to interact with the file.
        The BScan file will remain open until BScan.close() is called

        Args:
            bscan_file_or_csv_dir (str): Path to the .daq file. File must exist.

        Returns:
            b_scan (BScan): BScan class that can be used to interact with and read from the .daq file.
        """
        self.scan_format = BScan_File_Type.daq
        self.header = ScanHeaderData_daq()
        _, filename = os.path.split(bscan_file_or_csv_dir)
        self.scan_folder_name, _ = filename.split('.')
        super().__init__(bscan_file_or_csv_dir)
        # Not sure if it's guaranteed this will be in scan info, the documentation says that DAQ files use 0.1
        # resolution, so use that as a default
        self.step_size = 0.1
        self.file_type = BScan_File_Type.daq


        for information in self.header.ScanInfo:
            if information.key == 'Step Size':
                self.step_size = float(information.val)

        self.DATA_DTYPE = np.uint16

        self.data_offsets = np.zeros((self.header.NumUTChannels, self.header.FrameCount), dtype=np.uint64)
        self.a_scan_metadata_size = 26
        frame_metadata_size = 32
        self.ut_data_point_size = 2

        for i, ut_channel in enumerate(self.header.ChannelDataOffsets):
            byte_offset = 0
            for j in range(self.header.NumUTChannels):
                read_offset = ut_channel + byte_offset
                self.data_offsets[j][i] = read_offset
                # The last four bytes before the a-scan data is the a-scan length
                self.f.seek(read_offset + frame_metadata_size - 4)
                # a-scan values are U16, so each value is two bytes
                a_scan_data_size = int.from_bytes(self.f.read(4), self.endian) * self.ut_data_point_size
                total_a_scan_size = (a_scan_data_size + self.a_scan_metadata_size) * 3600 + frame_metadata_size
                byte_offset += total_a_scan_size

    def _read_header(self) -> ScanHeaderData_daq:
        '''
        Reads the Scan Header Data.

        This function returns the ScanHeaderData, and stores it internally as a class variable.
        '''

        # Init empty ScanHeaderData class
        header_data = ScanHeaderData_daq()

        # Seek to the beginning of the bscan file
        self.f.seek(0)
        header_data.Prefix = bytes.decode(self.f.read(7))
        header_data.VersionMajor = int.from_bytes(self.f.read(2), self.endian)
        header_data.VersionMinor = int.from_bytes(self.f.read(2), self.endian)

        fileversion = header_data.VersionMajor + header_data.VersionMinor / 10
        operator_string_length = int.from_bytes(self.f.read(1), self.endian)
        header_data.Operator = bytes.decode(self.f.read(operator_string_length))
        tool_desc_length = int.from_bytes(self.f.read(1), self.endian)
        header_data.InspectionHeadId = bytes.decode(self.f.read(tool_desc_length))
        serial_no_length = int.from_bytes(self.f.read(1), self.endian)
        header_data.SerialNumber = bytes.decode(self.f.read(serial_no_length))
        header_data.Year = str(int.from_bytes(self.f.read(4), self.endian))[
                           2:]  # only want the last two digits of the year
        header_data.Month = int.from_bytes(self.f.read(4), self.endian)
        header_data.Day = str(int.from_bytes(self.f.read(4), self.endian))
        header_data.ScanDate = header_data.Day + '-' + calendar.month_abbr[header_data.Month] + '-' + header_data.Year
        header_data.Hour = str(int.from_bytes(self.f.read(4), self.endian))
        header_data.Minute = str(int.from_bytes(self.f.read(4), self.endian))
        header_data.Second = str(int.from_bytes(self.f.read(4), self.endian))
        header_data.ScanTime = header_data.Hour + ':' + header_data.Minute + ':' + header_data.Second
        header_data.SampleFreqMHz = int.from_bytes(self.f.read(4), self.endian)
        header_data.SampleRes = int.from_bytes(self.f.read(4), self.endian)
        header_data.FrameCount = int.from_bytes(self.f.read(4), self.endian)
        header_data.FooterOffset = int.from_bytes(self.f.read(8), self.endian)
        header_data.NumUTChannels = int.from_bytes(self.f.read(4), self.endian)
        header_data.UTChannelInfo = [UT_Channel_daq() for _ in range(header_data.NumUTChannels)]
        header_data.ChannelLabels = ['' for _ in range(header_data.NumUTChannels)]
        for chan_num in range(header_data.NumUTChannels):
            channel_info = UT_Channel_daq()
            channel_name_length = int.from_bytes(self.f.read(1), self.endian)
            _, channel_info.channel_name = bytes.decode(self.f.read(channel_name_length)).split(' ', 1)
            header_data.ChannelLabels[chan_num] = channel_info.channel_name
            channel_info.eMode = int.from_bytes(self.f.read(4), self.endian)
            channel_info.num_RX_elements = int.from_bytes(self.f.read(4), self.endian)
            channel_info.num_TX_elements = int.from_bytes(self.f.read(4), self.endian)
            channel_info.sampling_freq_MHz = int.from_bytes(self.f.read(4), self.endian)
            channel_info.calibrated = int.from_bytes(self.f.read(1), self.endian)
            analog_filter_name_length = int.from_bytes(self.f.read(1), self.endian)
            channel_info.analog_filter_name = bytes.decode(self.f.read(analog_filter_name_length))
            channel_info.low_filter_MHz = int.from_bytes(self.f.read(4), self.endian)
            channel_info.high_filter_MHz = int.from_bytes(self.f.read(4), self.endian)
            channel_info.zero_offset_dp = int.from_bytes(self.f.read(4), self.endian)
            channel_info.acq_delay = int.from_bytes(self.f.read(4), self.endian)
            channel_info.insp_delay = int.from_bytes(self.f.read(4), self.endian)
            channel_info.range_dp = int.from_bytes(self.f.read(4), self.endian)
            if fileversion > 6.5:
                _gain_db = struct.unpack('f', self.f.read(4))[0]
            else:
                _gain_db = (self.f.read(16))
            channel_info.pulse_voltage_V = int.from_bytes(self.f.read(2), self.endian)
            channel_info.pulse_width_ns = int.from_bytes(self.f.read(2), self.endian)
            channel_info.tx = int.from_bytes(self.f.read(2), self.endian)
            channel_info.rx = int.from_bytes(self.f.read(2), self.endian)
            channel_info.b_echo_trigger_mode = int.from_bytes(self.f.read(1), self.endian)
            channel_info.b_rectified = int.from_bytes(self.f.read(1), self.endian)
            channel_info.aperture = int.from_bytes(self.f.read(2), self.endian)
            channel_info.averaging = int.from_bytes(self.f.read(2), self.endian)
            channel_info.half_matrix_capture = int.from_bytes(self.f.read(1), self.endian)
            channel_info.fmc_circular_aperature = int.from_bytes(self.f.read(1), self.endian)
            gate_label_length = int.from_bytes(self.f.read(1), self.endian)
            channel_info.gate_info.Gate_label = bytes.decode(self.f.read(gate_label_length))
            channel_info.gate_info.gate_offset = int.from_bytes(self.f.read(2), self.endian)
            channel_info.gate_info.gate_length = int.from_bytes(self.f.read(2), self.endian)
            channel_info.gate_info.gate_threshold = int.from_bytes(self.f.read(2), self.endian)
            channel_info.gate_info.params = int.from_bytes(self.f.read(2), self.endian)
            channel_info.num_gates = int.from_bytes(self.f.read(4), self.endian)
            channel_info.gate_data = ""
            for i in range(channel_info.num_gates):
                gate_data = Interface_Gate_Info()
                _gate_label_length = int.from_bytes(self.f.read(1), self.endian)
                gate_data.Gate_label = bytes.decode(self.f.read(_gate_label_length))
                gate_data.gate_offset = int.from_bytes(self.f.read(2), self.endian)
                gate_data.gate_length = int.from_bytes(self.f.read(2), self.endian)
                gate_data.gate_threshold = int.from_bytes(self.f.read(2), self.endian)
                gate_data.params = int.from_bytes(self.f.read(2), self.endian)
                channel_info.gate_data += str(gate_data)
            channel_info.has_primary_dac = int.from_bytes(self.f.read(1), self.endian)
            channel_info.has_secondary_dac = int.from_bytes(self.f.read(1), self.endian)
            header_data.UTChannelInfo[chan_num] = channel_info
        header_data.NumAxes = int.from_bytes(self.f.read(4), self.endian)
        axes_label_length = int.from_bytes(self.f.read(1), self.endian)
        header_data.AxesLabel = bytes.decode(self.f.read(axes_label_length))
        header_data.NumMetaData = int.from_bytes(self.f.read(4), self.endian)
        header_data.PrimaryAxisInfoAvail = int.from_bytes(self.f.read(1), self.endian)
        if header_data.PrimaryAxisInfoAvail:
            header_data.PrimaryAxisInfo.axis_type = int.from_bytes(self.f.read(4), self.endian)
            header_data.PrimaryAxisInfo.axis_index = int.from_bytes(self.f.read(4), self.endian)
            header_data.PrimaryAxisInfo.min_pos = self.f.read(16)
            header_data.PrimaryAxisInfo.max_pos = self.f.read(16)
            header_data.PrimaryAxisInfo.num_waves = int.from_bytes(self.f.read(4), self.endian)
            header_data.PrimaryAxisInfo.step = self.f.read(16)
            header_data.PrimaryAxisInfo.units_length = int.from_bytes(self.f.read(1), self.endian)
            header_data.PrimaryAxisInfo.units = int.from_bytes(self.f.read(header_data.PrimaryAxisInfo.units_length),
                                                               self.endian)
        header_data.FrameAxisInfoAvail = int.from_bytes(self.f.read(1), self.endian)
        if header_data.FrameAxisInfoAvail:
            header_data.FrameAxisInfo.axis_type = int.from_bytes(self.f.read(4), self.endian)
            header_data.FrameAxisInfo.axis_index = int.from_bytes(self.f.read(4), self.endian)
            header_data.FrameAxisInfo.min_pos = self.f.read(16)
            header_data.FrameAxisInfo.max_pos = self.f.read(16)
            header_data.FrameAxisInfo.num_waves = int.from_bytes(self.f.read(4), self.endian)
            header_data.FrameAxisInfo.step = self.f.read(16)
            header_data.FrameAxisInfo.units_length = int.from_bytes(self.f.read(1), self.endian)
            header_data.FrameAxisInfo.units = int.from_bytes(self.f.read(header_data.PrimaryAxisInfo.units_length),
                                                             self.endian)
        header_data.ProbeInfoAvail = int.from_bytes(self.f.read(1), self.endian)
        header_data.ProbeDescLength = int.from_bytes(self.f.read(1), self.endian)
        header_data.ProbeDesc = bytes.decode(self.f.read(header_data.ProbeDescLength))
        header_data.NumProbes = int.from_bytes(self.f.read(2), self.endian)
        header_data.ProbeTypes = int.from_bytes(self.f.read(4), self.endian)
        # initialize probe info array
        header_data.ProbeInfo = [PROBE_INFO_daq() for _ in range(header_data.NumProbes)]
        for probe in header_data.ProbeInfo:
            probe.probe_desc_len = int.from_bytes(self.f.read(1), self.endian)
            probe.probe_desc = bytes.decode(self.f.read(probe.probe_desc_len))
            probe.freq_MHz = struct.unpack('f', self.f.read(4))
            probe.angle_deg = struct.unpack('f', self.f.read(4))
            probe.position = struct.unpack('f', self.f.read(4))
        header_data.NumScanInfo = int.from_bytes(self.f.read(4), self.endian)
        header_data.ScanInfo = [Scan_Info() for _ in range(header_data.NumScanInfo)]
        for i, _ in enumerate(header_data.ScanInfo):
            scan_info_data = Scan_Info()
            scan_info_data.key_len = int.from_bytes(self.f.read(1), self.endian)
            scan_info_data.key = bytes.decode(self.f.read(scan_info_data.key_len))
            scan_info_data.val_len = int.from_bytes(self.f.read(1), self.endian)
            scan_info_data.val = bytes.decode(self.f.read(scan_info_data.val_len))
            header_data.ScanInfo[i] = scan_info_data
        header_data.NumExtendedInfo = int.from_bytes(self.f.read(4), self.endian)
        header_data.ExtendedInfo = [Scan_Info() for _ in range(header_data.NumExtendedInfo)]
        for i, _ in enumerate(header_data.ExtendedInfo):
            scan_info_data = Scan_Info()
            scan_info_data.key_len = int.from_bytes(self.f.read(1), self.endian)
            scan_info_data.key = bytes.decode(self.f.read(scan_info_data.key_len))
            scan_info_data.val_len = int.from_bytes(self.f.read(1), self.endian)
            scan_info_data.val = bytes.decode(self.f.read(scan_info_data.val_len))
            header_data.ExtendedInfo[i] = scan_info_data
            # convert to FC specific parameters
            if scan_info_data.key == "Campaign":
                header_data.Campaign = scan_info_data.val
            elif scan_info_data.key == "Station":
                if scan_info_data.val == "Pickering B":
                    header_data.GeneratingStation = Generating_Station.Pickering_B
                elif scan_info_data.val == "Pickering A":
                    header_data.GeneratingStation = Generating_Station.Pickering_A
                elif scan_info_data.val == "Darlington A":
                    header_data.GeneratingStation = Generating_Station.Darlington_A
                elif scan_info_data.val == "Darlington":
                    header_data.GeneratingStation = Generating_Station.Darlington_A
            elif scan_info_data.key == "Unit":
                header_data.UnitNumber = int(scan_info_data.val)
            elif scan_info_data.key == "Channel":
                header_data.ChannelNumber = int(scan_info_data.val[1:])
                header_data.ChannelLetter = scan_info_data.val[0]
            elif scan_info_data.key == "Axial Pitch":
                header_data.AxialPitch = float(scan_info_data.val)
            elif scan_info_data.key == "Axial Start":
                header_data.FirstAxialPosition = float(scan_info_data.val)
            elif scan_info_data.key == "Axial End":
                header_data.LastAxialPosition = float(scan_info_data.val)
            elif scan_info_data.key == "Rotary Start":
                header_data.FirstRotaryPosition = float(scan_info_data.val)
            elif scan_info_data.key == "Rotary End":
                header_data.LastRotaryPosition = float(scan_info_data.val)

        header_data.DataOffset = int.from_bytes(self.f.read(8), self.endian)
        header_data.ChannelDataOffsets = [-1 for _ in range(header_data.FrameCount)]
        for offset_cache in range(header_data.FrameCount):
            header_data.ChannelDataOffsets[offset_cache] = int.from_bytes(self.f.read(8), self.endian)
        header_data.AxialCache = [-1 for _ in range(header_data.FrameCount)]
        for axial_cache in range(header_data.FrameCount):
            header_data.AxialCache[axial_cache] = int.from_bytes(self.f.read(4), self.endian)
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
        # Different Labels may correspond to the came channel_label, such as 'NB1' and 'NB2' both being part of the
        # 'ID1 NB' label in D-Scans.
        bscan_channel.channel_label = self.label_dict[channel]
        bscan_channel.channel_index = self.header.ChannelLabels.index(self.label_dict[channel])
        b_scan_channel_of_interest = self.header.UTChannelInfo[bscan_channel.channel_index]
        # axial range
        bscan_channel.axial_range = self.header.FrameCount

        # rotary range
        bscan_channel.rotary_range = b_scan_channel_of_interest.num_RX_elements

        # time range
        bscan_channel.time_range = b_scan_channel_of_interest.range_dp

        # Use the range to calculate the total size of a slice, in bytes
        bscan_channel.slice_databuffer_size_bytes = b_scan_channel_of_interest.range_dp * self.ut_data_point_size
        bscan_channel.slice_total_size_bytes = bscan_channel.slice_databuffer_size_bytes + self.a_scan_metadata_size
        # Calculate the data offsets for each slice start (in bytes)
        bscan_channel.slice_offsets = self.data_offsets[bscan_channel.channel_index]

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

        full_axis.axial_pos = np.zeros(channel_info.axial_range)
        position_offset = 12
        for i, slice_offset in enumerate(channel_info.slice_offsets):
            self.f.seek(int(slice_offset) + position_offset)
            axial_pos = struct.unpack('f', self.f.read(4))[0]
            full_axis.axial_pos[i] = axial_pos
        axis = full_axis

        # crop the axes according to the specified ROI
        axial_indices = roi.get_axial_indices(channel_info, full_axis)
        axis.axial_pos = axis.axial_pos[slice(axial_indices[0], axial_indices[1] + 1)]
        rotary_indices = roi.get_rotary_indices(channel_info, full_axis)
        axis.rotary_pos = np.arange(0, channel_info.rotary_range / 10, self.step_size)
        time_indices = roi.get_time_indices(channel_info, full_axis)
        axis.time_pos = np.arange(0, channel_info.time_range)

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

        # Seek to the appropriate position for the target slice
        self.f.seek(int(bscan_channel.slice_offsets[slice]) + 12)

        # Read the axial position and add it to the axial_pos list
        axial_pos = struct.unpack('f', self.f.read(4))[0]
        bscan_slice.axial_pos = axial_pos

        # Read rotary position and add it to the rotary_pos list
        self.f.read(12)
        # rot_pos = int.from_bytes(self.f.read(2), self.endian)
        # bscan_slice.rotary_pos = rot_pos

        # Get number bytes in each slice
        bytes_to_read = int.from_bytes(self.f.read(4), self.endian)
        bscan_slice.data = np.empty([bscan_channel.rotary_range, bytes_to_read], dtype=self.DATA_DTYPE)

        # Read the channel data in
        for axial_data in range(bscan_channel.rotary_range):
            self.f.read(26)
            try:
                read_data = np.frombuffer(self.f.read(bytes_to_read * 2), self.DATA_DTYPE,
                                          bytes_to_read)
                bscan_slice.data[axial_data] = read_data
            except:
                pass
        return bscan_slice

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
        result.data = np.empty([b_scan_channel.axial_range, b_scan_channel.rotary_range,
                                self._read_slice(b_scan_channel, 1).data.shape[1]],
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
        rotary_start_index, _ = roi.get_rotary_indices(channel_info, full_axis)
        time_start_index, _ = roi.get_time_indices(channel_info, full_axis)

        # The UTChannel info time array size and the actual size of the time domain do not agree in all cases. The
        # number of data points in the slice is considered correct.
        rotary_end_index, time_end_index = self._read_slice(channel_info, 0).data.shape

        result = BScanData()
        result.axes = self.get_channel_axes(channel, roi)

        # Initialize the data array
        result.data = np.empty([axial_end_index - axial_start_index + 1,
                                rotary_end_index - rotary_start_index,
                                time_end_index - time_start_index],
                               dtype=dtype)

        for slice_i in range(axial_start_index, axial_end_index + 1):
            # use the same _read_slice() since it can't efficiently parse rotary and time anyways
            # TODO: dig in further and see if _read_slice() can in fact be optimized a bit for this case
            bscan_slice = self._read_slice(channel_info, slice_i)
            result.data[slice_i - axial_start_index] = bscan_slice.data[rotary_start_index: rotary_end_index + 1,
                                                       time_start_index: time_end_index + 1]

        return result

    def save_header_file(self, header_info: ScanHeaderData_daq, file_location: str, header_text):
        formatted_header_data = ''
        handled_types = [str, int, list]
        for prop in filter(lambda s: not s.startswith('__'), dir(header_info)):
            attribute = getattr(header_info, prop)
            attribute_type = type(attribute)
            # assert attribute_type in handled_types
            formatted_header_data += prop + '\n'
            if attribute_type == int or attribute_type == str:
                formatted_header_data += str(getattr(header_info, prop))
            elif attribute_type == list:
                formatted_header_data += ', '.join([str(element) for element in attribute])
            formatted_header_data += "\n\n"
        formatted_header_data += "Locations\n" + '\n'.join(header_text)
        header_file = open(os.path.join(file_location, 'Header.csv'), 'w')
        header_file.write(formatted_header_data)
        header_file.close()

    def get_scan_info(self):
        scan_start = float(self.header.ExtendedInfo[6].val)
        scan_end = float(self.header.ExtendedInfo[7].val)
        station_name = Generating_Station[self.header.ExtendedInfo[1].val.replace(" ", "_")]
        return scan_start, scan_end, station_name

    def overwrite_data(self, new_data: dict, new_file_path: Path, file_pos: dict, frame_start_number=0):
        new_file_ref = open(new_file_path, 'r+b')
        for probe, data_offset in zip(self.mapped_labels, self.header.ChannelDataOffsets):
            frame = frame_start_number
            assert probe in new_data.keys(), "Error: " + probe + " array not found"
            # get the channel info in order to get the slice offsets
            bscan_channel = self.get_channel_info(probe)
            # go to the next slice offset
            new_file_ref.seek(file_pos[probe])
            # get the data from the probe we are iterating over
            b_scan_data = new_data[probe].data

            number_of_frames = b_scan_data.shape[0]
            number_of_ascans = b_scan_data.shape[1]
            pbar = range(number_of_frames)

            for i in pbar:
                # Skip over axis meta-data which we do not care about
                new_file_ref.seek(int(bscan_channel.slice_offsets[frame]) + 32)
                for y in range(number_of_ascans):
                    # Read the array in as a pd dataframe
                    data = pd.DataFrame(b_scan_data[i][y], dtype=self.DATA_DTYPE)
                    # the daq files have meta data after every a-scan. we need to skip over this data every single a-scan
                    new_file_ref.seek(new_file_ref.tell() + 26)
                    # Convert the 1D numpy array into a 1D array of bytes
                    byte_data = data.to_records(index=False).tobytes()
                    # Write the 1D array of bytes to file
                    new_file_ref.write(byte_data)
                frame += 1
            file_pos[probe] = new_file_ref.tell()
        new_file_ref.close()
        return file_pos


