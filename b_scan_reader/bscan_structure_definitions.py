from enum import IntEnum
import math
import numpy as np
from collections.abc import Sequence


class BScan_File_Type(IntEnum):
    NA = 0
    anf = 1
    daq = 2


class Anf_File_Scan_Type(IntEnum):
    NA = 0
    A = 1
    D = 2


class Channel_End(IntEnum):
    NA = 0
    Inlet = 1
    Outlet = 2


class Probe(IntEnum):
    NA = 0
    APC = 1
    CPC = 2
    NB1 = 3
    NB2 = 4

    def __str__(self):
        return str(self.name)


class WaveGroup(IntEnum):
    NA = -1
    G2 = 0
    G3 = 1
    G4 = 2
    G5 = 3


class Reactor_Face(IntEnum):
    NA = 0
    East = 1
    West = 2
    North = 3
    South = 4


class ScanType(IntEnum):
    NA = 0
    Helical_C_Scan = 2,
    Detailed_Axial_C_Scan = 3,
    Detailed_Circ_C_Scan = 4,
    Circumferential_B_Scan = 5,
    Axial_B_Scan = 6,
    On_Channel_Cal_1 = 10,
    On_Channel_Cal_2 = 11,
    Circumferential_B_Scan_Cal = 12,
    Pipe_B_Scan = 13


class Generating_Station(IntEnum):
    NA = 0
    Pickering_A = 1
    Pickering_B = 2
    Bruce_A = 3
    Bruce_B = 4
    Darlington_A = 5
    Darlington_B = 6
    Point_Lepreau = 7
    Gentilly = 8
    Wolsung = 9
    Cordoba = 10
    Cermavoda = 11


class PROBE_INFO_daq:
    def __str__(self):
        string = 'Probe Description: ' + self.probe_desc + ' ' + \
                 'Frequency: ' + str(self.freq_MHz[0]) + ' ' + \
                 'Angle: ' + str(self.angle_deg[0]) + ' ' + \
                 'Position: ' + str(self.position[0])
        return string

    probe_desc_len = -1
    probe_desc = ''
    freq_MHz = 0.0
    angle_deg = 0.0
    position = 0.0


class Interface_Gate_Info:
    def __init__(self):
        self.Gate_label = ''
        self.gate_offset = -1
        self.gate_length = -1
        self.gate_threshold = -1
        self.params = -1

    def __str__(self):
        string = "GateLabel = " + str(self.Gate_label) + ' ' + \
                 "GateOffset = " + str(self.gate_offset) + ' ' + \
                 "GateLength = " + str(self.gate_length) + ' ' + \
                 "GateThreshold = " + str(self.gate_threshold) + ' ' + \
                 "Params = " + str(self.params) + ' '
        return string


class gate_data:
    gate_label = ''
    gate_offset = -1
    gate_length = -1
    gate_threshold = -1
    params = -1


class UT_Channel_daq:
    def __init__(self):
        self.channel_name = ''
        self.eMode = -1
        self.num_RX_elements = -1
        self.num_TX_elements = -1
        self.sampling_freq_MHz = -1
        self.calibrated = -1
        self.analog_filter_name = -1
        self.low_filter_MHz = -1.0
        self.high_filter_MHz = -1.0
        self.zero_offset_dp = -1
        self.acq_delay = -1
        self.insp_delay = -1
        self.range_dp = -1
        self.pulse_voltage_V = -1
        self.pulse_width_ns = -1
        self.tx = -1
        self.rx = -1
        self.b_echo_trigger_mode = -1
        self.b_rectified = -1
        self.aperture = -1
        self.averaging = -1
        self.half_matrix_capture = -1
        self.fmc_circular_aperature = -1
        self.gate_info = Interface_Gate_Info()
        self.num_gates = -1
        self.gate_data = [Interface_Gate_Info()]
        self.has_primary_dac = -1
        self.has_secondary_dac = -1

    def __str__(self):
        dictionary = self.__dict__
        string = ''
        for item in dictionary:
            string += item + ": " + str(dictionary[item]) + ' '
        return string


class Scan_Info:
    def __str__(self):
        return self.key + ": " + self.val + '\t'

    key_len = -1
    key = ''
    val_len = -1
    val = ''


class Axis_Info:
    axis_type = -1
    axis_index = -1
    min_pos = -1
    max_pos = -1
    num_waves = -1
    step = -1
    units_length = -1
    units = ''


class ScanHeaderData:
    VersionMajor = -1
    VersionMinor = -1
    Operator = ''
    InspectionHeadId = ''
    Year = -1
    Month = -1
    Day = -1
    FooterOffset = None
    ScanTime = ''
    ChannelLabels = []
    UnfilteredChannelLabels = []
    ChannelDataOffsets = []
    ScanDate = ''


class ScanHeaderData_anf(ScanHeaderData):
    GeneratingStation = Generating_Station.NA
    UnitNumber = -1
    ChannelNumber = -1
    ChannelLetter = ''
    ChannelEnd = Channel_End.NA
    ReactorFace = Reactor_Face.NA
    FirstRotaryPosition = -1
    LastRotaryPosition = -1
    AxialPitch = -1
    FirstChannel = -1
    LastChannel = -1
    EncoderResolutionAxial = -1
    AxialResolutionCnt = -1
    ScanType = ScanType.NA
    AxialStart_mm = -1
    AxialEnd_mm = -1
    FirstAxialPosition = -1
    LastAxialPosition = -1
    AxialIncrementalResolution = -1
    ScanSensRelNotch = -1
    PowerCheck = []
    Gain_db = []
    Comment = ''
    ChannelThresholds = []
    GatesDelay = []
    GatesRange = []
    GatesReceiverFreq = []
    System = -1
    ChannelLabels2 = []


class ScanHeaderData_daq(ScanHeaderData):
    AxesLabel = ''
    AxialCache = []
    DataOffset = -1
    ExtendedInfo = [Scan_Info]
    FrameAxisInfo = Axis_Info
    FrameAxisInfoAvail = -1
    FrameCount = -1
    NumAxes = -1
    NumProbes = -1
    NumUTChannels = -1
    NumExtendedInfo = -1
    NumMetaData = -1
    NumScanInfo = -1
    Prefix = ''
    PrimaryAxisInfo = Axis_Info
    PrimaryAxisInfoAvail = -1
    ProbeDesc = ''
    ProbeInfo = [PROBE_INFO_daq]
    ProbeInfoAvail = -1
    ProbeDescLength = -1
    ProbeTypes = -1
    SampleFreq = -1
    ScanInfo = [Scan_Info]
    SerialNumber = ''
    SampleFreqMHz = -1
    SampleRes = -1
    UTChannelInfo = [UT_Channel_daq]


class FooterEntryTypes(IntEnum):
    TrackChangeRecord = 0
    SoftwareGains = 1
    AnalystComments = 2


class FooterEntryItem:
    header_offset = -1
    entry_type = -1
    data = None


class FooterChangeTrackRecord:
    number_change_parameters = -1
    change_track_records = []


class SingleChangeTrackRecord:
    axial_position = -1
    date_time = ''
    parameter_name = ''
    parameter_value_before_change = ''
    parameter_value_after_change = ''


class FooterSoftwareGains:
    gains = []


class FooterAnalystComments:
    length = -1
    comment_string = ''


class ScanFooterData:
    item_count = -1
    footer_items = []

    def __init__(self):
        self.item_count = -1
        self.footer_items = []


class UTEXInfoChannel:
    gain = -1
    pulse_voltage = -1
    pulse_width = -1
    low_filter = -1
    high_filter = -1
    digitizer_rate = -1
    digitizer_attenuation = -1
    gate_start = -1
    gate_width = -1


class UTEXInfo(Sequence):
    gain = [-1 for i in range(12)]
    pulse_voltage = [-1 for i in range(12)]
    pulse_width = [-1 for i in range(12)]
    low_filter = [-1 for i in range(12)]
    high_filter = [-1 for i in range(12)]
    digitizer_rate = [-1 for i in range(12)]
    digitizer_attenuation = [-1 for i in range(12)]
    gate_start = [-1 for i in range(20)]
    gate_width = [-1 for i in range(20)]

    def __init__(self):
        self.gain = [-1 for i in range(12)]
        self.pulse_voltage = [-1 for i in range(12)]
        self.pulse_width = [-1 for i in range(12)]
        self.low_filter = [-1 for i in range(12)]
        self.high_filter = [-1 for i in range(12)]
        self.digitizer_rate = [-1 for i in range(12)]
        self.digitizer_attenuation = [-1 for i in range(12)]
        self.gate_start = [-1 for i in range(20)]
        self.gate_width = [-1 for i in range(20)]
        super().__init__()

    def __getitem__(self, index) -> UTEXInfoChannel:
        """Get the UTEX Hardware info for a specific channel
    
        Args:
            index (int): Channel index to get
    
        Returns:
            channel_utex (UTEXInfoChannel): Class with the same field names as UTEXInfo, but every field is a scalar.
        """
        channel_utex = UTEXInfoChannel()
        channel_utex.gain = self.gain[index]
        channel_utex.pulse_voltage = self.pulse_voltage[index]
        channel_utex.pulse_width = self.pulse_width[index]
        channel_utex.low_filter = self.low_filter[index]
        channel_utex.high_filter = self.high_filter[index]
        channel_utex.digitizer_rate = self.digitizer_rate[index]
        channel_utex.digitizer_attenuation = self.digitizer_attenuation[index]
        channel_utex.gate_start = self.gate_start[index]
        channel_utex.gate_width = self.gate_width[index]
        return channel_utex

    def __len__(self):
        return len(self.gain)


class BScanSlice:
    axial_pos = -1
    rotary_pos = -1
    data = np.empty([0, 0])

    def __init__(self):
        self.axial_pos = -1
        self.rotary_pos = -1
        self.data = np.empty([0, 0])


class BScanChannel:
    """Class containing metadata about a channel.

    Attributes:\n
    - label = aka 'Mapped Label'. Short-form channel label, such as NB1, NB2, CPC, or APC.\n
    - channel_label = Full channel label, such as 'ID1 NB' or 'Axial Shear P/C'.\n
    - channel_index = Index of the channel in the file.\n
    - data_offset = Start position of the channel in the file, in bytes.\n
    - slice_size_bytes = Size, in bytes, of a single slice. This does not include the 6 bytes for axial & rotary positions at the start of the slice.\n
    - time_range = Number of elements in the Time axis.\n
    - rotary_range = Number of elements in the Rotary axis.\n
    - axial_range = Number fo elements in the Axial axis.
    """
    label = None
    channel_label = None
    channel_index = None
    data_offset = None
    slice_databuffer_size_bytes = None
    slice_total_size_bytes = None
    time_range = None
    rotary_range = None
    axial_range = None
    slice_offsets = None


class BScanAxis:
    axial_pos = np.empty(0)
    rotary_pos = np.empty(0)
    time_pos = np.empty(0)
    axial_data_index = -1
    rotary_data_index = -1
    time_data_index = -1

    def __init__(self):
        self.axial_pos = np.empty(0)
        self.rotary_pos = np.empty(0)
        self.time_pos = np.empty(0)
        self.axial_data_index = -1
        self.rotary_data_index = -1
        self.time_data_index = -1

    def rotary_angle_to_index(self, rotary_target: float):
        """Calculates nearest index of the rotary position.

        Args:
            rotary_target (float): Rotary position in degrees.

        Raises:
            ValueError: Requested rotary position is outside of the files range.

        Returns:
            index (int): Index of the rotary position.
        """
        if self.rotary_pos[0] < self.rotary_pos[-1]:
            # Rotary Start is less than Rotary Stop. Rotary doesn't cross 360.0 Assume the rotary_pos array is sorted from low-> high.
            assert self.rotary_pos[0] <= rotary_target <= self.rotary_pos[
                -1], "ValueError: Requested position is not in the axis range."

        else:
            # Rotary Start is greater than Rotary Stop. Rotary crosses 360.0. rotary_pos is not sorted from low -> high.
            # Check that the target position is within an acceptable range. Return the index of the position closest to rotary_target
            assert (self.rotary_pos[0] <= rotary_target <= 360.0) or (0.0 <= rotary_target <= self.rotary_pos[
                -1]), "ValueError: Requested position is not in the axis range."

        index = np.argmin(abs(self.rotary_pos - rotary_target))
        return index

    def rotary_range_indices(self, rotary_start: float, rotary_stop: float):
        """Calculates Rotary Start and Rotary Stop positons.

        Args:
            rotary_start (float): Start position in degrees.
            rotary_stop (float): Stop position in degrees.

        Raises:
            ValueError: Requested angles are not in the file range.

        Returns:
            start_index (int): Index of the start position
            stop_index (int): Index of the stop position
        """

        start_index = self.rotary_angle_to_index(rotary_start)
        stop_index = self.rotary_angle_to_index(rotary_stop)

        return start_index, stop_index

    def axial_pos_to_index(self, axial_target: float):
        """Calculates nearest index of the axial position.

        Args:
            axial_target (float): Axial position in mm

        Raises:
            ValueError: Requested position is not in the files range.

        Returns:
            index (int): Index of the requested position
        """
        # Round to 3 decimal places to compensate for floating-point errors.

        # this line will frequently be off by one, but should be much faster than the line below for large arrays. if performance requires this, just fix the following case:
        # Ex: searching for 9.8 in [9.7999999999, 9.9] will return different from [9.800000001, 9.9] even through it probably shouldn't
        # index = np.searchsorted(self.axial_pos, axial_target)

        index = np.argmin(np.abs(self.axial_pos - axial_target))
        return index

    def axial_range_indices(self, axial_start: float, axial_stop: float):
        """Calculates Axial Start and Axial Stop positons.

        Args:
            axial_start (float): Start position in mm.
            axial_stop (float): Stop position in mm.

        Raises:
            ValueError: Requested positions are not in the file range.

        Returns:
            start_index (int): Index of the start position
            stop_index (int): Index of the stop position
        """

        start_index = self.axial_pos_to_index(axial_start)
        stop_index = self.axial_pos_to_index(axial_stop)

        return start_index, stop_index

    def time_to_index(self, time_target: float):
        """Calculates nearest index of the requested time position.

        Args:
            time_target (float): Time position in µs

        Raises:
            ValueError: Requested position is not in the files range.

        Returns:
            index (int): Index of the requested position
        """
        assert self.time_pos.min() <= time_target <= self.time_pos.max(), "ValueError: Requested position is not in the axis range."

        # see comment on axial_pos_to_index() if you're changing this
        # index = np.searchsorted(self.time_pos, time_target)
        index = np.argmin(np.abs(self.time_pos - time_target))
        return index

    def time_range_indices(self, time_start: float, time_stop: float):
        """Calculates Time Start and Time Stop positon indexes. The returned indices will be the index closest to the specied value.

        Args:
            axial_start (float): Time position in µs.
            axial_stop (float): Time position in µs.

        Raises:
            ValueError: Requested positions are not in the file range.

        Returns:
            start_index (int): Index of the start position
            stop_index (int): Index of the stop position
        """

        start_index = self.time_to_index(time_start)
        stop_index = self.time_to_index(time_stop)

        return start_index, stop_index


class BScanData:
    def __init__(self):
        self.data = np.array(3, int)
        self.axes = BScanAxis()
        self.probe = Probe(0)


def double_set_checker_decorator_factory(other_bounds_name):
    """verifies that the decorated setter function does not write the property if 
    self.other_bounds_name has already been set. This prevents a user from setting
    roi.axial_start_index and roi.axial_start_mm for example
    """

    def decorator(function):
        def set_wrapper(*args, **kwargs):
            other_bounds_value = getattr(args[0], other_bounds_name)
            if args[1] is not None and other_bounds_value is not None:
                # trying to set the same roi bound in 2 different ways
                raise Exception("Cannot set property '%s' when '%s' has already been set to '%s'" % (
                    function.__name__, other_bounds_name, other_bounds_value))
            return function(*args, **kwargs)

        return set_wrapper

    return decorator


class ROI:
    _axial_start_index = None
    _axial_end_index = None
    _rotary_start_index = None
    _rotary_end_index = None
    _time_start_index = None
    _time_end_index = None

    _axial_start_mm = None
    _axial_end_mm = None
    _rotary_start_deg = None
    _rotary_end_deg = None
    _time_start_us = None
    _time_end_us = None

    def __init__(self, **kwargs):
        # Loop through any arguments that may have been specified.
        for key, value in kwargs.items():
            try:
                # Only let users set properties that already exist.
                getattr(self, key)
                setattr(self, key, value)
            except:
                # An invalid property was set, throw an error.
                raise ValueError("Unexpected key name: '%s'" % key)

    @property
    def axial_start_index(self):
        return self._axial_start_index

    @axial_start_index.setter
    @double_set_checker_decorator_factory("axial_start_mm")
    def axial_start_index(self, value):
        self._axial_start_index = value

    @property
    def axial_end_index(self):
        return self._axial_end_index

    @axial_end_index.setter
    @double_set_checker_decorator_factory("axial_end_mm")
    def axial_end_index(self, value):
        self._axial_end_index = value

    @property
    def axial_start_mm(self):
        return self._axial_start_mm

    @axial_start_mm.setter
    @double_set_checker_decorator_factory("axial_start_index")
    def axial_start_mm(self, value):
        self._axial_start_mm = value

    @property
    def axial_end_mm(self):
        return self._axial_end_mm

    @axial_end_mm.setter
    @double_set_checker_decorator_factory("axial_end_index")
    def axial_end_mm(self, value):
        self._axial_end_mm = value

    @property
    def rotary_start_index(self):
        return self._rotary_start_index

    @rotary_start_index.setter
    @double_set_checker_decorator_factory("rotary_start_deg")
    def rotary_start_index(self, value):
        self._rotary_start_index = value

    @property
    def rotary_end_index(self):
        return self._rotary_end_index

    @rotary_end_index.setter
    @double_set_checker_decorator_factory("rotary_end_deg")
    def rotary_end_index(self, value):
        self._rotary_end_index = value

    @property
    def rotary_start_deg(self):
        return self._rotary_start_deg

    @rotary_start_deg.setter
    @double_set_checker_decorator_factory("rotary_start_index")
    def rotary_start_deg(self, value):
        self._rotary_start_deg = value

    @property
    def rotary_end_deg(self):
        return self._rotary_end_deg

    @rotary_end_deg.setter
    @double_set_checker_decorator_factory("rotary_end_index")
    def rotary_end_deg(self, value):
        self._rotary_end_deg = value

    @property
    def time_start_index(self):
        return self._time_start_index

    @time_start_index.setter
    @double_set_checker_decorator_factory("time_start_us")
    def time_start_index(self, value):
        self._time_start_index = value

    @property
    def time_end_index(self):
        return self._time_end_index

    @time_end_index.setter
    @double_set_checker_decorator_factory("time_end_us")
    def time_end_index(self, value):
        self._time_end_index = value

    @property
    def time_start_us(self):
        return self._time_start_us

    @time_start_us.setter
    @double_set_checker_decorator_factory("time_start_index")
    def time_start_us(self, value):
        self._time_start_us = value

    @property
    def time_end_us(self):
        return self._time_end_us

    @time_end_us.setter
    @double_set_checker_decorator_factory("time_end_index")
    def time_end_us(self, value):
        self._time_end_us = value

    def get_axial_indices(self, channel: BScanChannel, axis: BScanAxis):
        if self.axial_start_index is not None:
            assert 0 <= self.axial_start_index < channel.axial_range, "Axial Index out of range"
            start = self.axial_start_index
        elif self.axial_start_mm is not None:
            start = axis.axial_pos_to_index(self.axial_start_mm)
        else:
            start = 0

        if self.axial_end_index is not None:
            assert 0 <= self.axial_end_index < channel.axial_range, "Axial Index out of range"
            end = self.axial_end_index
        elif self.axial_end_mm is not None:
            end = axis.axial_pos_to_index(self.axial_end_mm)
        else:
            end = channel.axial_range - 1

        assert start <= end, "Axial start must be <= axial end"

        return start, end

    def get_rotary_indices(self, channel: BScanChannel, axis: BScanAxis):
        if self.rotary_start_index is not None:
            assert 0 <= self.rotary_start_index < channel.rotary_range, "Rotary Index out of range"
            start = self.rotary_start_index
        elif self.rotary_start_deg is not None:
            start = axis.rotary_angle_to_index(self.rotary_start_deg)
        else:
            start = 0

        if self.rotary_end_index is not None:
            assert 0 <= self.rotary_end_index < channel.rotary_range, "Rotary Index out of range"
            end = self.rotary_end_index
        elif self.rotary_end_deg is not None:
            end = axis.rotary_angle_to_index(self.rotary_end_deg)
        else:
            end = channel.rotary_range - 1

        assert start <= end, "Rotary start must be <= rotary end"

        return start, end

    def get_time_indices(self, channel: BScanChannel, axis: BScanAxis):
        if self.time_start_index is not None:
            assert 0 <= self.time_start_index < channel.time_range, "Time Index out of range"
            start = self.time_start_index
        elif self.time_start_us is not None:
            start = axis.time_to_index(self.time_start_us)
        else:
            start = 0

        if self.time_end_index is not None:
            assert 0 <= self.time_end_index < channel.time_range, "Time Index out of range"
            end = self.time_end_index
        elif self.time_end_us is not None:
            end = axis.time_to_index(self.time_end_us)
        else:
            end = channel.time_range - 1

        assert start <= end, "Time start must be <= time end"

        return start, end
