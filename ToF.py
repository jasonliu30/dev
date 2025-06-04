import numpy as np
from scipy.signal import savgol_filter
import numba
import pycorrelate_norm
from tqdm.auto import trange
from b_scan_reader.bscan_structure_definitions import BScanData
from utils.logger_init import create_dual_loggers

# create loggers
dual_logger, _  = create_dual_loggers()

wg_dtype = np.dtype([('start_index', 'i2'), ('stop_index', 'i2'), ('peak_index', 'i2'), ('peak_value', 'i2'), ('average_value', 'f4'), ('area_under_curve', 'f4')])


def generate_cscans(probe_data, bboxes, config):
    """
    Generate C-Scan representations based on the current probe and the provided bounding boxes.

    Args:
        probe_data (BScanData): Object for the current probe.
        bboxes (list[np.ndarray]): Contains start & stop locations of each G4 location after imputation.
        config (Config): Configuration parameters.

    Returns:
        dict: Contains generated C_Scans representations.
    """

    # Convert probe data to np.int16 so that the median value can be calculated
    probe_data.data = probe_data.data.astype(np.int16)

    scan_data_placeholder = np.zeros((len(probe_data.data), len(probe_data.data[0]), 4), 
                                     dtype=np.int16)

    # Map probe names to configurations
    config_map = {
        'APC': ('apc_whole_scan', 'apc'),
        'CPC': ('cpc_whole_scan', 'cpc'),
        'NB1': ('nb1_whole_scan', 'nb1'),
        'NB2': ('nb2_whole_scan', 'nb2'),
    }

    # Select the appropriate configuration
    whole_scan_config_name, probe_config_name = config_map[probe_data.probe.name]
    whole_scan_config = getattr(config.tof.probe_config, whole_scan_config_name)
    probe_config = getattr(config.tof.probe_config, probe_config_name)

    # Calculate G4 results
    dual_logger.info("  - Calculating Time of Flight...")
    tof_results = time_of_flight(probe_data, probe_config, config.tof, bboxes)

    # Calculate Whole-scan results
    dual_logger.info("  - Finding Lags in Whole Scan...")
    full_scan_lags, full_scan_cors = find_lags_in_scans(
                                    probe_data.data,
                                    tof_results['reference_locs'],
                                    scan_data_placeholder,
                                    whole_scan_config.lag_range,
                                    whole_scan_config.lag_tracking,
                                    whole_scan_config.min_points,
                                    whole_scan_config.use_wavegroups,
                                    whole_scan_config.wiggle,
                                    whole_scan_config.cor_tracking,
                                    whole_scan_config.lag_start,
                                    whole_scan_config.median_value
                                    )
    
    tof_results.update({
            'lags_whole': full_scan_lags,
            'cors_whole': full_scan_cors,
            'probe': probe_data.probe.name
        })

    return tof_results


def get_bounds_and_wavegroups(b_scan_data: BScanData, probe_config, bboxes, use_g2_g3=False):
    # Prepare variables

    probe_data = b_scan_data.data
    wavegroups = np.zeros((probe_data.shape[0], probe_data.shape[1]), dtype=wg_dtype)
    total_TOF = np.zeros((probe_data.shape[0], probe_data.shape[1]), dtype=np.int16)

    # Convert bounding boxes to a more useful format.
    bboxes_bounds = prepare_bounding_boxes(bboxes, probe_data.shape[0], use_g2_g3)

    # Find peak locations
    for frame in trange(len(probe_data), desc='Finding Peak Locations', ascii=True):
        wavegroups[frame] = detect_wavegroups_scan(probe_data[frame], bboxes_bounds[frame], **probe_config.__dict__)


    if not probe_config.use_wavegroups and bboxes != []:
        bounds = bboxes_bounds
    else:
        bounds = wavegroups

    return bounds, wavegroups


def get_lags_and_cors(b_scan_data: BScanData, reference_locs, wavegroups, probe_config):
    lags, cors = find_lags_in_scans(b_scan_data.data, reference_locs, wavegroups,
                                                probe_config.lag_range, probe_config.lag_tracking, probe_config.min_points,
                                                probe_config.use_wavegroups,
                                                probe_config.wiggle, probe_config.cor_tracking, probe_config.lag_start,
                                                probe_config.median_value)

    return lags, cors


def get_time_of_flights(wavegroups, reference_locs, lags, cors, probe_config, tof_config):
    total_TOF = np.zeros((wavegroups.shape[0], wavegroups.shape[1]), dtype=np.int16)

    for frame in range(len(wavegroups)):
        total_TOF[frame][0] = wavegroups[frame][reference_locs[frame]][2] + lags[frame][0]

    total_indications = []
    for frame in trange(len(lags), desc='Detecting Indications', ascii=True):
        indications = detect_indications(lags[frame], cors[frame], tof_config)
        total_indications.append(indications)

    # converting correlation lags into TOF
    for frame in range(len(total_TOF)):
        for loc in range(1, len(total_TOF[frame])):
            total_TOF[frame][loc] = total_TOF[frame][0] + lags[frame][loc]
    flattened_TOF = np.array([flatten_lags_indications(total_TOF[frame], total_indications[frame]) for frame in range(len(total_TOF))])

    return total_TOF * probe_config.conversion_factor, flattened_TOF*probe_config.conversion_factor


def time_of_flight(b_scan_data: BScanData, probe_config, tof_config, bboxes, use_g2_g3=False):

    """
    Main Time of Flight script.

    results dictionary contains:
        - ToF
        - ToF_flat
        - cors
        - lags
        - wavegroups
        - amp
        - average_amplitudes
        - area_under_curve
        - reference_locs

    Args:
        b_scan_data: Probe Data in BScanData class format
        tof_config: Dictionary of setttings, taken from ToF_config.yaml
        bboxes (list[np.ndarray]): List of bounding boxes
        use_g2_g3 (bool): If True, searches for G2 & G3 instead of G4

    Returns:
        dict: results dictionary
    """
    # Prepare variables
    # probe_config = ProbeConfig(tof_config, b_scan_data.probe)
    # probe_config = tof_config.probe_config
    probe_data = b_scan_data.data
    wavegroups = np.zeros((probe_data.shape[0], probe_data.shape[1]), dtype=wg_dtype)
    total_TOF = np.zeros((probe_data.shape[0], probe_data.shape[1]), dtype=np.int16)

    # Convert bounding boxes to a more useful format.
    bboxes_bounds = prepare_bounding_boxes(bboxes, probe_data.shape[0], use_g2_g3)

    # Find peak locations
    for frame in range(len(probe_data)):
        wavegroups[frame] = detect_wavegroups_scan(probe_data[frame], bboxes_bounds[frame], **probe_config.__dict__)

    # Start cross correlation calculations
    reference_locs = select_reference_scans(wavegroups)

    if not probe_config.use_wavegroups and bboxes != []:
        bounds = bboxes_bounds
    else:
        bounds = wavegroups

    # print("Finding Lags in scans...")
    # start_time = time.time()
    lags, cors = find_lags_in_scans(probe_data, reference_locs, bounds,
                                    probe_config.lag_range, probe_config.lag_tracking, probe_config.min_points, probe_config.use_wavegroups,
                                    probe_config.wiggle, probe_config.cor_tracking, probe_config.lag_start, probe_config.median_value)
    # print(f"Elapsed Time: {time.time() - start_time:.1f} seconds")

    for frame in range(len(wavegroups)):
        total_TOF[frame][0] = wavegroups[frame][reference_locs[frame]][2] + lags[frame][0]

    total_indications = []
    for frame in range(len(lags)):
        indications = detect_indications(lags[frame], cors[frame], tof_config)
        total_indications.append(indications)

    # converting correlation lags into TOF
    for frame in range(len(total_TOF)):
        for loc in range(1, len(total_TOF[frame])):
            total_TOF[frame][loc] = total_TOF[frame][0] + lags[frame][loc]
    flattened_TOF = [flatten_lags_indications(total_TOF[frame], total_indications[frame]) for frame in range(len(total_TOF))]

    # Gather results for outputting
    results = {'tof': total_TOF * probe_config.conversion_factor,
               'ToF_flat': np.array(flattened_TOF) * probe_config.conversion_factor,
               'cors': cors,
               'lags': lags,
               'wavegroups': wavegroups,
               'amp': wavegroups['peak_value'],
               'average_amplitudes': wavegroups['average_value'],
               'area_under_curve': wavegroups['area_under_curve'],
               'reference_locs': reference_locs}

    return results


def detect_wavegroups_scan(frame, bbox_bounds, number_of_wavegroups=2, amplitude_threshold=10,
                           wavegroup_width=200, max_peak_distances=0, median_value=128, **kwargs):
    """Iterates through locations in a frame to find wave group.

    Optionally detects several wavegroups in every scan, based on number_of_wavegroups, and chooses the best one.

    Args:
        frame (np.ndarray): Bscan to detect wave groups in
        bbox_bounds(np.ndarray): Bounds to detect wavegroups in, in the shape (number of scans, 2)
             Each scan should have a beginning and end value. If both are 0 the whole scan is used
        number_of_wavegroups (int): Maximum number of wavegroups to detect
        amplitude_threshold (int): Minimum peak amplitude threshold. If no peaks exceed this threshold, no wavegroups are returned.
        wavegroup_width (int): expected width of each wavegroup, used for all wavegroups identified
        max_peak_distances (int): If multiple wavegroups are detected, but some wavegroups are further away than
            max_peak_distance from the peak with the highest amplitude, those far-away peaks are ignored
        median_value (int): Value to be subtracted from frame to convert it to a median scan
        **kwargs (dict): Unused inputs from the settings yaml

    Returns
        wavegroups: list of lists, the outermost list has 1 entry for each scan.
                    each scan has a list of wavegroups that is length 0-2, based on the number of wave groups detected
                    each wavegroup has 4 values, starting point, ending point, peak location, and amplitude of the peak
    """

    wavegroups = [np.array([(bbox_bounds[0], bbox_bounds[1], 0, 0, 0, 0)], dtype=wg_dtype)] * len(frame)
    prev_valid_peak = np.array([(0, 0, 0, 0, 0, 0)], dtype=wg_dtype)[0]

    ##################
    # Peak Detection #
    ##################
    for a_scan_i in range(len(frame)):
        # Find peaks in every A Scan in the frame
        wave_locations = find_wavegroups_in_frame(frame[a_scan_i],
                                                  prev_valid_peak[2],
                                                  bbox_bounds,
                                                  number_of_wavegroups,
                                                  amplitude_threshold,
                                                  wavegroup_width,
                                                  max_peak_distances,
                                                  median_value)
        if len(wave_locations) == 0:
            # No peaks detected. Return bbox_bounds as the start & stop
            wave_locations = np.array([(bbox_bounds[0], bbox_bounds[1], 0, 0, 0, 0)], dtype=wg_dtype)
        elif len(wave_locations) == 1:
            # Only one wave location was found, keep track of it
            prev_valid_peak = wave_locations[0]
        wavegroups[a_scan_i] = wave_locations

    ################
    # Post Process #
    ################
    for a_scan_i in reversed(range(len(frame))):
        # Go backwards through the wavegroups, choosing the best wavegroup from each a_scan
        if len(wavegroups[a_scan_i]) == 1:
            # If only one peak was detected, use it and remember it
            wavegroups[a_scan_i] = wavegroups[a_scan_i][0]
            prev_valid_peak = wavegroups[a_scan_i]
        elif len(wavegroups[a_scan_i]) > 1:
            # If multiple peak are detected, choose the peak closest to the previously valid peak
            distances = [np.abs(prev_valid_peak[2] - wg[2]) for wg in wavegroups[a_scan_i]]
            wavegroups[a_scan_i] = wavegroups[a_scan_i][np.argmin(distances)]

    return np.asarray(wavegroups)


def prepare_bounding_boxes(bboxes, num_frames, use_g2_g3):
    """Adjusts the ML bounding boxes (if necessary) and converts it to a 2D numpy array.

    If `use_g2_g3` is True, this function will return the area of the BScan before the G4 wavegroup. That is, the part
    of the BScan that contains the G2 and G3 wavegroups.

    If `use_g2_g3` is False, it leaves the bounding boxes unchanged and simply converts it to a numpy array.

    Args:
        bboxes (list[np.ndarray]): Contains the start & stop locations of each G4 wavegroup detection
        num_frames (int): Number of frames in the BScan file
        use_g2_g3 (bool): If True, uses the start of the G4 wavegroup detection as the end of the prepared bounding box

    Returns:
        np.ndarray: list of prepared bounding boxes, which either represent G4 or G2 / G3
    """
    bounding_boxes = np.zeros((num_frames, 2), dtype=np.int16)
    if bboxes is not None and len(bboxes) == num_frames:
        for frame in range(len(bounding_boxes)):
            if len(bboxes[frame].shape) == 1:
                bbox = bboxes[frame]
            else:
                bbox = [np.min(bboxes[frame], axis=0)[0], np.max(bboxes[frame], axis=0)[1]]

            if use_g2_g3:
                if any(np.isnan(bbox)):
                    bounding_boxes[frame] = [0, 0]
                else:
                    bounding_boxes[frame][0] = 0
                    bounding_boxes[frame][1] = bbox[0]
            else:
                if any(np.isnan(bbox)):
                    bounding_boxes[frame] = [0, 0]
                else:
                    bounding_boxes[frame] = bbox
    return bounding_boxes


def find_wavegroups_in_frame(a_scan, previous_peak_posn, scan_bounds, number_of_wavegroups=2,
                             amplitude_threshold=0, wavegroup_width=300,
                             max_peak_distances=100, median_value=128):
    """This function looks through the A Scan and searches for the wavegroup location

    Args:
        a_scan (np.ndarray): Current A Scan
        previous_peak_posn (int): Peak location in the previous A Scan. Used for outlier rejection.
        scan_bounds (np.ndarray): Start & Stop location of the peak detection search bounds
        number_of_wavegroups (int): Max number of wavegroups to look for.
        amplitude_threshold (int): Minimum peak amplitude threshold. If no peaks exceed this threshold, no wavegroups are returned.
        wavegroup_width (int): expected width of each wavegroup, used for all wavegroups identified
        max_peak_distances (int): Ignore secondary peaks further away from the primary peak
        median_value (int): Value to subtract from the A Scan to convert it to a median scan

    Returns:
        np.ndarray: wavegroup_locations of type wg_dtype. Columns are: ['start_index', 'stop_index', 'peak_index', 'peak_value', 'average_value', 'area_under_curve']
    """
    ########
    # Initialize data
    ########
    wavegroup_locations = np.zeros(0, dtype=wg_dtype)

    median_scan = np.abs(a_scan - median_value)

    if scan_bounds[1] != 0:
        a_scan_subset = a_scan[scan_bounds[0]:scan_bounds[1]]
        median_scan_subset = median_scan[scan_bounds[0]:scan_bounds[1]].copy()
    else:
        a_scan_subset = a_scan
        median_scan_subset = median_scan.copy()

    # Convert width to half-width, to it can easily be added/subtracted from positions
    wavegroup_width = int(wavegroup_width / 2)

    ########
    # Wavegroup detection
    ########
    for i in range(number_of_wavegroups):
        peak_posn = np.argmax(median_scan_subset)


        # Take subset of the scan based on the wavegroup widths. Check to make sure the wavegroup width doesn't exceed the scan indices.
        #reverse the subset so we can efficiently grab the last-biggest value using np.argmax() - which defaults to the first instance of multiple equally large values
        scan = a_scan_subset[
               min(len(a_scan_subset), peak_posn + wavegroup_width + 1)-1:max(0, peak_posn - wavegroup_width):-1]

        peak_index = len(scan) - np.argmax(scan)

        # Shift the peak_posn index if necessary. pos_peak refers to the index in `scan`, which might be a subset of `a_scan`
        peak_posn = peak_index + max(0, peak_posn - wavegroup_width)

        if median_scan_subset[peak_posn] < amplitude_threshold:
            # No peaks were detected above the threshold. End detection.
            pass

        elif i > 0 and 0 < max_peak_distances < np.abs(wavegroup_locations[2] - peak_posn):
            # The other peak detected in the current A Scan is too far away from the initial detection
            pass

        elif i > 0 and abs(wavegroup_locations[i - 1][2] - previous_peak_posn) < abs(peak_posn + scan_bounds[0] - previous_peak_posn) and previous_peak_posn > 0:
            # The other peak detected in the current frame is closer to the previous A Scan. Ignore this peak.
            pass

        else:
            # A wavegroup was detected. Add it to the wavegroup list
            wavegroup_start_sub = peak_posn - wavegroup_width if peak_posn - wavegroup_width > 0 else 0
            wavegroup_end_sub = peak_posn + wavegroup_width if peak_posn + wavegroup_width < median_scan_subset.shape[0] - 1 else median_scan_subset.shape[0] - 1

            # Set the wavegroup area to 0 for the detection of the next wavegroup to prevent it from being detected twice
            median_scan_subset[wavegroup_start_sub:wavegroup_end_sub] = 0

            # Shift positions based on the scan_bounds
            peak_posn = peak_posn + scan_bounds[0]
            peak_value = median_scan[peak_posn]
            # wavegroup_start = wavegroup_start_sub + scan_bounds[0]
            # wavegroup_end = wavegroup_end_sub + scan_bounds[0]
            wavegroup_start = max(peak_posn - wavegroup_width, 0)
            wavegroup_end = min(peak_posn + wavegroup_width,  median_scan.shape[0] - 1)

            average_amplitude = np.mean(median_scan[wavegroup_start:wavegroup_end])
            area_under_curve = np.trapz(median_scan[wavegroup_start:wavegroup_end])

            wavegroup = np.array((wavegroup_start, wavegroup_end, peak_posn, peak_value, average_amplitude, area_under_curve), dtype=wg_dtype)
            wavegroup_locations = np.append(wavegroup_locations, wavegroup)

            amplitude_threshold = max(amplitude_threshold, peak_value // 2)

    return wavegroup_locations


@numba.jit(parallel=True)
def select_reference_scans(wavegroups, n_row=10):
    """Selects a reference scan to use in a frame using wave groups.
    If there are n_row locations in a row with a wave group, the first scan in that sequence is used.
    This is done to prevent using an anomalous wave group as reference, and to use a scan that has a valid wave group as a reference.

    Args:
        wavegroups (np.ndarray): (number of frames, number of scans,4)
        n_row (int): integer number, this many wavegroups in a row is needed to consider a wavegroup to be a valid one

    Returns:
        np.ndarray: reference_scans: reference scan locations, 1 integer for each frame
    """

    reference_scans = np.zeros(len(wavegroups), dtype=np.uint16)
    first = 0
    for frame in numba.prange(len(wavegroups)):
        count = 0
        for loc in range(len(wavegroups[frame])):
            if wavegroups[frame][loc][2] != 0:
                if count == 0:
                    first = loc
                count = count + 1
                if count == n_row:
                    break
            else:
                first = 0
                count = 0
        reference_scans[frame] = first
    return reference_scans


@numba.jit(nopython=True, parallel=True)
def calc_positive_lags(ref_A_scan, A_scan, min_lag, max_lag, scale):
    cors_scan = pycorrelate_norm.ucorrelate(ref_A_scan, A_scan, maxlag=(max_lag - min_lag + 1),
                                            scale=scale, start=min_lag)
    max_index = np.argmax(cors_scan)
    shift = min_lag + max_index
    x_axis = np.zeros(len(cors_scan), dtype=np.int16)
    for i in range(len(cors_scan)):
        x_axis[i] = min_lag + i

    return cors_scan, x_axis, shift, cors_scan[max_index]


@numba.jit(nopython=True, parallel=True)
def calc_negative_lags(ref_A_scan, A_scan, min_lag, max_lag, scale):
    cors_scan = pycorrelate_norm.ucorrelate(A_scan, ref_A_scan, maxlag=(np.abs(min_lag) - np.abs(max_lag) + 1),
                                            scale=scale, start=np.abs(max_lag))
    max_index = np.argmax(cors_scan)
    shift = max_lag - max_index
    x_axis = np.zeros(len(cors_scan), dtype=np.int16)
    for i in range(len(cors_scan)):
        x_axis[i] = max_lag - i

    return cors_scan, x_axis, shift, cors_scan[max_index]


@numba.jit(nopython=True, parallel=True)
def calc_twosided_lags(ref_A_scan, A_scan, min_lag, max_lag, scale):
    result_forward = pycorrelate_norm.ucorrelate(ref_A_scan, A_scan, maxlag=max_lag + 1, scale=scale)
    result_forward = np.flip(result_forward)
    result_backward = pycorrelate_norm.ucorrelate(A_scan, ref_A_scan, maxlag=np.abs(min_lag), scale=scale, start=1)
    # combine both correlation passes (excluding one point where they overlap)
    cors_scan = np.concatenate((result_forward, result_backward))
    # calculating the lag in the signal
    middle = result_forward.shape[0] - 1
    max_index = np.argmax(cors_scan)
    shift = middle - max_index
    x_axis = np.zeros(len(cors_scan), dtype=np.int16)
    for i in range(len(cors_scan)):
        x_axis[i] = middle - i

    return cors_scan, x_axis, shift, cors_scan[max_index]

@numba.jit(nopython=True)
def find_lag_cor(ref_A_scan, A_scan, lag_window, ref_bounds=(0, 0), scan_bounds=(0, 0), scale=True, lag_centre=0, min_points=5) -> tuple[int, float, np.ndarray, np.ndarray]:
    """Outputs all correlation values. Finds the lag in signals using cross correlations.

    Positive lag means signal is delayed relative to reference, negative means signal is early.

    Args:
        ref_A_scan (np.ndarray): scan used as the baseline for the lag calculation. output will be the relative delay between this and A_scan
        A_scan (np.ndarray): scan to measure the delay in
        lag_window (int): maximum number of times to shift the 2 signals when calculating correlations, in both positive and negative directions.
                    For example a lag window of 2 will shift the signal twice in the positive direction and twice in the negative direction.
                    This is such that a lag window that is the same as the size of both signals minus one will search all possible lags (with a lag centre of 0)
        ref_bounds: where to cut the reference A scan in the format (starting point, ending point) (includes starting point but not endpoint)
        scan_bounds: where to cut the A_scan in the format (starting point, ending point) (includes starting point but not endpoint)
        scale: If True, the correlation at each lag will be divided by sqrt((sum of xi**2)*(sum of yi**2)), this makes the correlation between -1 and +1
               If False, the correlation values are identical to the ones calculated from numpy or scipy
        lag_centre: The centre lag of the lags to calculate. Lags will be calculated from: (lag_centre - lag_window) to (lag_center + lag_window)
                    If lag_centre=5 and lag_window=10 lags from -5 to 15 are calculated
        min_points: the minimum number of points to calculate cross correlations for.
                    For example with 2 signals that are 5 values each; a min_points of 2 will prevent a lag of 4 from being calculated
                    since only 1 point overlaps at a lag of 4

    Output
        - shift:
        - cors_scan[max_index]: highest cross correlation
        - x_axis: the lags for the cross correlations
        - cors_scan: cross correlations array at the indices in x_axis

    """

    # checking if the bounds were in the correct format
    if len(ref_bounds) != 2 or len(scan_bounds) != 2:
        raise ValueError('Included bounds were not in the correct format, (start index, end index).')
    # cutting signals to included bounds
    if ref_bounds != (0, 0):
        ref_A_scan = ref_A_scan[ref_bounds[0]:ref_bounds[1]]
    if scan_bounds != (0, 0):
        A_scan = A_scan[scan_bounds[0]:scan_bounds[1]]

    # Finding the minimum and maximum lag we will be calculating
    min_lag = lag_centre - lag_window
    max_lag = lag_centre + lag_window

    if min_points > 0:
        # if lag is +ve t is ref u is a scan
        if min_lag >= 0:
            max_lag_points = len(A_scan) - min_points
            if max_lag_points < max_lag:
                max_lag = max_lag_points
        elif max_lag <= 0:
            min_lag_points = (len(ref_A_scan) - min_points) * -1
            if min_lag_points > min_lag:
                min_lag = min_lag_points
        else:
            max_lag_points = len(A_scan) - min_points
            if max_lag_points < max_lag:
                print(f'new_max_lag: {max_lag_points}')
                max_lag = max_lag_points
            min_lag_points = (len(ref_A_scan) - min_points) * -1
            if min_lag_points > min_lag:
                min_lag = min_lag_points

    if min_lag >= 0:
        # if we only need to calculate positive lags
        cors_scan, x_axis, shift, max_cross_corr = calc_positive_lags(ref_A_scan, A_scan, min_lag, max_lag, scale)

    elif max_lag <= 0:
        # if we only need to calculate negative lags
        cors_scan, x_axis, shift, max_cross_corr = calc_negative_lags(ref_A_scan, A_scan, min_lag, max_lag, scale)

    else:
        # pycorrelate only calculates in one direction, so we need to run twice to get positive and negative lags
        cors_scan, x_axis, shift, max_cross_corr = calc_twosided_lags(ref_A_scan, A_scan, min_lag, max_lag, scale)


    shift = shift - (ref_bounds[0] - scan_bounds[0])

    for i in range(len(x_axis)):
        x_axis[i] = x_axis[i] - (ref_bounds[0] - scan_bounds[0])

    if len(x_axis) != len(cors_scan):
        x_axis = np.zeros(cors_scan.shape, dtype=np.int16)

    return shift, max_cross_corr, x_axis, cors_scan

@numba.jit(nopython=True, parallel=True, cache=True)
def find_lags_in_scans(scans, reference_scans, wavegroups, lag_range, lag_tracking, min_points, use_wavegroups, wiggle_room, cor_tracking, lag_start, median_value):
    """Loops through frames  and performs cross correlations for each scan comparing it to the reference scan in that frame.

    Args:
        scans (np.ndarray): All the scans to run cross correlation on
        reference_scans (np.ndarray): The location of the scans to use as reference. Should be 1 integer for each frame
        wavegroups (np.ndarray): The bounds of the signals to use for cross correlation
        lag_range (int): Range of lags to calculate cross correlations for
        lag_tracking (bool): Whether the lags calculated for cross correlation are based on the lag of the last circumferential
            location. Not recommended if each circumferential location has varying bounds
        min_points (int): The minimum number of points to use for cross correlation. This prevents calculating the correlation
            of very few points, creating empty arrays
        use_wavegroups (bool): Whether the signal is cropped using the wavegroup bounds before cross correlation
        wiggle_room (int): The amount to expand the wavegroups bounds by
        cor_tracking (bool): Whether to use tracking of the same peak of correlations. Only works if lag tracking is also True
        lag_start (int):
        median_value (int):

    Returns
        - lags: the lag / index for each scan that had the highest correlation
        - cors: the correlation at each of those lags
    """

    # Initialize variables
    same_peak_area = 3  # the number of points to search in both directions from the previous lag when looking for the same correlation peak
    same_cor_thresh = 0.1  # the maximum difference in correlations to still consider the same peak
    farthest_from_max = 0.2

    lags = np.zeros((len(scans), len(scans[0])), dtype=np.int16)
    cors = np.zeros((len(scans), len(scans[0])), dtype=np.float32)
    
    #Logic for tracking correlations
   
    def track_cors(lag, cor, last_cor, centre, x_axis, cors_scan, same_peak_area, same_cor_thresh, farthest_from_max):
        """"
           Attempts to find a lag when there are multiple lags that are similar in correlation
           Inputs
           --------
           -lag is the currently selected lag (lag of max correlation)
           -cor is the currently selected correlation (max correlation)
           -last_cor is the correlation of the previous circumferential location
           -centre is the lag of the previous circumferential location
           -x_axis are all the lags that the correlations were calculated for
           -cors scan are all the correlations of the lags found in x_axis
           -the remaining inputs are inherited from find_lags_in_scans
           Outputs
           --------
           -lag is the chosen lag after evaluating the possible options
           -cor is the correlation of the chosen lag
           """
        same_peak = (np.abs(x_axis - centre) <= same_peak_area).nonzero()[0]
        # removing the correlations that are within same_cor_thresh of the last correlation
        same_peak_cor = same_peak[
            (np.abs(cors_scan[same_peak] - last_cor) <= same_cor_thresh).nonzero()[0]]
        # of the remaining points, only use the ones that are local maxima (lower on both sides)
        # maxima means peaks, determined by the previous and following lags being lower in correlation
        same_peak_cor_maxima = []
        for p in same_peak_cor:  # check every point and see if it is a maxima
            if 0 < p < len(cors_scan) - 1:
                if cors_scan[p] >= cors_scan[p - 1] and cors_scan[p] >= cors_scan[p + 1]:
                    same_peak_cor_maxima.append(p)
        same_peak_cor_maxima = np.array(same_peak_cor_maxima)
        # if there are points remaining, use the one closest to the previous lag
        if len(same_peak_cor_maxima) >= 1:
            chosen_peak = same_peak_cor_maxima[np.argmin(np.abs(x_axis[same_peak_cor_maxima] - centre))]
            # only accept new value if it is within farthest from max to the max correlation
            if np.abs(cors_scan[chosen_peak] - cor) <= farthest_from_max:
                lag = x_axis[chosen_peak]
                cor = cors_scan[chosen_peak]
        # if no points were maxima, use the one that is closest to the last lag
        elif len(same_peak_cor) >= 1:
            chosen_peak = same_peak_cor[np.argmin(np.abs(x_axis[same_peak_cor] - centre))]
            # only accept new value if it is within farthest from max to the max correlation
            if np.abs(cors_scan[chosen_peak] - cor) <= farthest_from_max:
                lag = x_axis[chosen_peak]
                cor = cors_scan[chosen_peak]
        else:
            # if there were no points close to the previous correlation,
            # compare to the max correlation instead
            same_peak_cor = same_peak[
                (np.abs(cors_scan[same_peak] - cor) <= farthest_from_max).nonzero()[0]]
            # of the remaining points, only use the ones that are local maxima (lower on both sides)
            same_peak_cor_maxima = []
            for p in same_peak_cor:
                if 0 < p < len(cors_scan) - 1:
                    if cors_scan[p] >= cors_scan[p - 1] and cors_scan[p] >= cors_scan[p + 1]:
                        same_peak_cor_maxima.append(p)
            same_peak_cor_maxima = np.array(same_peak_cor_maxima)
            # if there are points remaining, use the one closest to the previous lag
            if len(same_peak_cor_maxima) >= 1:
                chosen_peak = same_peak_cor_maxima[np.argmin(np.abs(x_axis[same_peak_cor_maxima] - centre))]
                # only accept new value if it is within farthest from max to the max correlation
                if np.abs(cors_scan[chosen_peak] - cor) <= farthest_from_max:
                    lag = x_axis[chosen_peak]
                    cor = cors_scan[chosen_peak]
            # if no points were maxima, use the one that is closest to the last lag
            elif len(same_peak_cor) >= 1:
                chosen_peak = same_peak_cor[np.argmin(np.abs(x_axis[same_peak_cor] - centre))]
                # only accept new value if it is within farthest from max to the max correlation
                if np.abs(cors_scan[chosen_peak] - cor) <= farthest_from_max:
                    lag = x_axis[chosen_peak]
                    cor = cors_scan[chosen_peak]
        return int(lag), float(cor)


    if use_wavegroups:
        for i in numba.prange(len(scans)):
            last_cor = 1
            # calculating the bounds for reference signal
            reference_bounds = (
                max(0, wavegroups[i][reference_scans[i]][0] - wiggle_room),
                min(wavegroups[i][reference_scans[i]][1] + wiggle_room, len(scans[i][reference_scans[i]])))
            # if no valid bounds exist, last bounds are used
            last_bounds = reference_bounds
            lag = lag_start
            for j in range(len(scans[i])):
                # if there is a valid peak location
                if wavegroups[i][j][1] != 0:
                    bounds = (max(0, wavegroups[i][j][0] - wiggle_room), min(wavegroups[i][j][1] + wiggle_room, len(scans[i][j])))
                    if lag_tracking:
                        centre = lag + (
                                    reference_bounds[0] - bounds[0])  # using previous lag as centre when lag tracking
                        # lag centre should adjust for differences in bounds
                        if centre >= (bounds[1] - bounds[0]):
                            centre = bounds[1] - bounds[0]
                        if centre <= (reference_bounds[1] - reference_bounds[0]) * (-1):
                            centre = (reference_bounds[1] - reference_bounds[0]) * (-1)
                        lag, cor, x_axis, cors_scan = find_lag_cor(scans[i][reference_scans[i]] - median_value,
                                                                   scans[i][j] - median_value,
                                                                   lag_window=lag_range, ref_bounds=reference_bounds,
                                                                   scan_bounds=bounds, scale=True,
                                                                   lag_centre=centre, min_points=min_points)
                        if cor_tracking:
                            lag, cor = track_cors(lag, cor, last_cor, centre, x_axis, cors_scan, same_peak_area,
                                                    same_cor_thresh, farthest_from_max)
                            last_cor = cor
                    else:
                        centre = 0
                        lag, cor, x_axis, cors_scan = find_lag_cor(scans[i][reference_scans[i]] - median_value,
                                                                   scans[i][j] - median_value,
                                                                   lag_window=lag_range, ref_bounds=reference_bounds,
                                                                   scan_bounds=bounds, scale=True,
                                                                   lag_centre=centre, min_points=min_points)

                    last_bounds = bounds
                    lags[i][j] = lag
                    cors[i][j] = cor
                    lag = lag + (reference_bounds[0] - bounds[0])
                # if there is not a valid peak location
                else:
                    # using last bounds
                    bounds = last_bounds
                    if lag_tracking:
                        centre = lag
                        lag, cor, x_axis, cors_scan = find_lag_cor(scans[i][reference_scans[i]] - median_value,
                                                                   scans[i][j] - median_value,
                                                                   lag_window=lag_range, ref_bounds=reference_bounds,
                                                                   scan_bounds=bounds, scale=True,
                                                                   lag_centre=0, min_points=min_points)

                        if cor_tracking:
                           lag, cor = track_cors(lag, cor, last_cor, centre, x_axis, cors_scan, same_peak_area,
                                                  same_cor_thresh, farthest_from_max)
                           last_cor = cor
                    else:
                        centre = 0
                        lag, cor, x_axis, cors_scan = find_lag_cor(scans[i][reference_scans[i]] - median_value,
                                                                   scans[i][j] - median_value,
                                                                   lag_window=lag_range, ref_bounds=reference_bounds,
                                                                   scan_bounds=bounds, scale=True,
                                                                   lag_centre=0, min_points=min_points)
                    lags[i][j] = lag
                    cors[i][j] = cor
                    lag = lag + (reference_bounds[0] - bounds[0])
                    # shifting last bounds by the lag, important if multiple scans in a row with no wave groups
                    # find the width of the last valid bounds
                    width = bounds[1] - bounds[0]
                    # adjusting last valid upper bound by lag found
                    lag_difference = lag - lags[i][j - 1]
                    upper_bound = bounds[1] + lag_difference
                    if upper_bound > len(scans[i][j]):
                        upper_bound = len(scans[i][j])
                        lower_bound = upper_bound - width
                    else:
                        lower_bound = bounds[0] + lag_difference
                    if lower_bound < 0:
                        lower_bound = 0
                        upper_bound = lower_bound + width
                    if upper_bound < lower_bound:
                        last_bounds = (lower_bound, upper_bound)
                    else:
                        last_bounds = bounds

    else:
        # this uses lag tracking, for when the whole scan is being used
        for i in numba.prange(len(scans)):
            last_cor = 1
            lag = lag_start
            for j in range(len(scans[i])):
                if lag_tracking:
                    centre = lag
                else:
                    centre = 0
                lag, cor, x_axis, cors_scan = find_lag_cor(scans[i][reference_scans[i]] - median_value,
                                                           scans[i][j] - median_value,
                                                           lag_window=lag_range, scale=True,
                                                           lag_centre=centre, min_points=min_points)
                if cor_tracking:
                    lag, cor = track_cors(lag, cor, last_cor, centre, x_axis, cors_scan, same_peak_area,
                                          same_cor_thresh, farthest_from_max)
                    last_cor = cor
                lags[i][j] = lag
                cors[i][j] = cor

    return lags, cors


def detect_indications(lags, cors, tof_config, mode='cors'):
    """ Selects a method to use to detect indications

    Args:
        lags (np.ndarray): 1D array of 1 frame of lags in shape (integer array)
        cors (np.ndarray): 1D array of 1 frame of correlations in shape (number of A scans)
        tof_config: Dictionary made from ToF_config.yaml
        mode: mode of indication detection, 'spike' or 'cors', default='cors'.

    Returns:
        np.ndarray: 1D array of indication locations
    """
    ind_config = tof_config.indication

    filtered_jumps = filter_jumps(lags, ind_config.jump_thresh)

    if mode == 'cors':
        cors_config = ind_config.cors
        indications = filter_cors(cors, **cors_config.__dict__, max_percent_removed=ind_config.max_percent_removed)
        indications = indications | filtered_jumps
        if cors_config.verbose:
            print('Cor_filtering')
            std = np.std(cors)
            mean = np.mean(cors)
            print('std. dev. before filter: ' + str(std))
            print('mean correlation before filter: ' + str(mean))
            print('mean correlation after filter: ' + str(np.mean(cors[np.invert(indications)])))
            print('std. dev. after filter: ' + str(np.std(cors[np.invert(indications)])))
        return indications
    else:
        indications = np.zeros(len(lags), bool)
        print('Warning: no indication detection mode selected')
        return indications | filtered_jumps


def filter_cors(cors, thresh, std_indications, std_flattening, max_iter_flattening, savgol_order, savgol_window_len, verbose=False, max_percent_removed=0.4):
    """This is a method of detecting indications by trying to flatten the correlations using a filter,
    and then applying a threshold based on the mean and standard deviation

    Args:
        cors: 1D array of 1 frame of correlations in shape (number of A scans)
        thresh: float to use as a threshold , correlation below this will be counted as indications
        std_indications (int): number of standard deviations from the mean to consider an indication when removing from flattening
        std_flattening (int): number of standard deviations from the mean to consider an indication when finding the profile of the correlations
        max_iter_flattening (int): number of flattening iterations
        savgol_order (int): Order for the Savgol filter used when smoothing the correlations
        savgol_window_len (int): Window for the Savgol filter used when smoothing the correlations
        verbose (bool): If true, prints diagnostics messages to the console
        max_percent_removed (float): percentage as (0-1), maximum percentage of points that can be considered indications.
            If the indications are above this amount, no indications are returned

    Returns:
        np.ndarray: 1D array of indication locations
    """

    smooth = smooth_cor(cors, std_flattening, max_iter_flattening, savgol_order, savgol_window_len)
    flat_correlations = cors - smooth
    sigma = np.sqrt(np.square(flat_correlations).sum() / flat_correlations.shape[0])

    if thresh == 0:
        thresh = np.mean(flat_correlations) - (std_indications * sigma)

    indications = [j <= thresh for j in flat_correlations]
    indications = outlier_extent(np.array(indications), flat_correlations)
    percent_removed = float(sum(indications) / len(cors))
    if verbose:  # pragma: no cover
        print('Percent of points that are indications: ' + str(percent_removed * 100) + '%')
    if percent_removed > max_percent_removed:
        if verbose:  # pragma: no cover
            print('Percentage of points removed over threshold of ' + str(max_percent_removed * 100) + '%')
        return [False for _ in cors]
    return indications


def filter_jumps(lags: np.ndarray, jump_thresh: int):
    """Marks jumps as indications 

    Args:
        lags: array to search for jumps, 1 frame of lags
        jump_thresh: the magnitude to consider a jump

    Returns:
        Indications array of booleans where True is the area of an indication
    """
    indications = np.zeros(len(lags), bool)
    derivative = np.diff(lags)
    for i in range(len(derivative)):
        if np.abs(derivative[i]) >= jump_thresh:
            indications[i] = True
            indications[i + 1] = True
    return indications


def smooth_cor(cors, std_flattening, max_iter_flattening, savgol_order, savgol_window_len) -> np.ndarray:
    """
    Function to find the general trend of the correlations in argument "cors".
    This is done by repeatedly passing the correlations through a filter to smooth the correlations and removing outliers before smoothing again.
    After a certain number of iterations, the general trend of the correlations (i.e. the smoothed correlations) will emerge.

    Args:
        cors (np.ndarray): Correlation Array
        std_flattening (int): number of standard deviations from the mean to consider an indication when finding the profile of the correlations
        max_iter_flattening (int): number of flattening iterations
        savgol_order (int): Order for the Savgol filter used when smoothing the correlations
        savgol_window_len (int): Window for the Savgol filter used when smoothing the correlations

    Returns:
        np.ndarray: smoothed: smoothed correlations
    """
    # Get the maximum number of iterations and the limit
    not_include = np.full(cors.shape[0], False, dtype=bool)

    # Ensure the savgol window length isn't larger than the length of cors
    if len(cors) < savgol_window_len:
        savgol_window_len = len(cors)

    # Loop through the number of times
    for i in range(max_iter_flattening):
        # Obtain the signal without any identified outliers.
        # Outliers are removed and linear interpolation is used to fill in the gaps for filtering
        signal_no_outliers = interpolate_indications(cors, not_include)

        # Smooth the signal with the selected filter and calculate the residuals
        smoothed = savgol_filter(signal_no_outliers, savgol_window_len, savgol_order)

        residuals = signal_no_outliers - smoothed

        # Calculate the standard error that will be used to set the limits for the next iteration of smoothing
        sigma = np.sqrt(np.square(residuals).sum() / residuals.shape[0])

        # Find the outliers
        outliers = residuals < (- std_flattening * sigma)
        # Now remove the points around the outliers that make up the trough
        outliers = outlier_extent(outliers, residuals)

        # Remove the points from the smoothing process
        not_include = not_include | outliers

    # Return the residuals
    return smoothed


def flatten_lags(lags, savgol_order=5, savgol_window_len=0):
    """Estimates pressure tube location and subtracts that from lags to flatten them

    Args:
        lags (np.ndarray): integer array of lags
        savgol_order (int): Order for the Savgol filter used when smoothing the correlations
        savgol_window_len (int): Window for the Savgol filter used when smoothing the correlations

    Returns:
        np.ndarray: flat_lags
    """
    if savgol_window_len == 0:
        savgol_window_len = int(3600 / 4) + 1
        if len(lags) < savgol_window_len:
            savgol_window_len = len(lags) - 1
        if savgol_window_len % 2 == 0:
            savgol_window_len = savgol_window_len - 1
    filtered_lags = savgol_filter(lags, savgol_window_len, savgol_order)
    flat_lags = lags - filtered_lags
    return flat_lags


def flatten_lags_indications(lags, indications, savgol_order=5, savgol_window_len=0):
    """Estimates pressure tube location. Removes indications and subtracts that from lags to flatten them

    Args:
        lags (np.ndarray): integer array of lags
        indications (np.ndarray): boolean array showing which indices are indications
        savgol_order (int): Order for the Savgol filter used when smoothing the correlations
        savgol_window_len (int): Window for the Savgol filter used when smoothing the correlations

    Returns:
        np.ndarray: flat_lags
    """
    if savgol_window_len == 0:
        savgol_window_len = int(3600 / 4) + 1
        if len(lags) < savgol_window_len:
            savgol_window_len = len(lags) - 1
        if savgol_window_len % 2 == 0:
            savgol_window_len = savgol_window_len - 1
    clean_lags = interpolate_indications(lags, indications)
    filtered_lags = savgol_filter(clean_lags, savgol_window_len, savgol_order)
    flat_lags = lags - filtered_lags
    return flat_lags


def interpolate_indications(lag, indications):
    """This removes indications from a frame of lags, the indications should be a list of booleans (True where there is an indication)
    The function removes all areas that are True in indications,
    and does linear interpolation to replace those values

    Args:
        lag (np.ndarray): 1D integer array of lags in shape (number of A scans)
        indications (np.ndarray): 1D boolean array of indication locations in shape (number of A scans)

    Returns:
        list: new_lags - list with indications removed in shape (number of A scans)
    """

    new_lags = [i for i in lag]
    in_indication = False
    start_of_indication = -1
    for i in range(len(lag)):
        if in_indication and not indications[i]:
            new_values = np.interp(range(start_of_indication, i), [start_of_indication, i],
                                   [lag[start_of_indication], lag[i]])
            new_lags[start_of_indication:i] = new_values
            in_indication = False
        if indications[i] and not in_indication:
            start_of_indication = i
            in_indication = True

    return new_lags


@numba.njit(cache=True)
def outlier_extent(passed: np.ndarray, residuals: np.ndarray) -> np.ndarray:
    """In the function smooth_cor(), the difference between the original correlations and the correlations smoothed by
    function Filter() is calculated (i.e. the residuals).

    Any location where the original correlations is significantly less than the smoothed correlation (i.e. an outlier)
    is indicative of an indication, and this is marked by False in the Numpy array Passed.

    This function marks the entire trough that contains the outlier as False as well.

    The extent of the trough is when the residuals become positive (i.e. the original correlation is greater than the smoothed correlation).

    Args:
        passed: array of booleans where False indicates the location is initial considered an outlier, where it indicates an indication.
                The function modifies array in place to switch elements to False that are considered part of the trough.
        residuals: array with the values of the residuals between the actual and filtered correlation.

    Returns:
        passed - same as the passed input variable, since it's edited in place
    """

    # Find the indices where outliers are identified
    outliers = np.where(passed)[0]

    # Loop through all the identified outliers, and remove all points until the residuals are greater than zero.
    # This marks the extent of the trough that should be removed from the flattening.
    for i in outliers:
        # Loop behind to find the extent of the trough
        index = i - 1
        while index >= 0 > residuals[index] and (not passed[index]):
            passed[index] = True
            index = index - 1

        # Loop ahead to find the extend of the trough
        index = i + 1
        while index < residuals.shape[0] and (not passed[index]) and residuals[index] < 0:
            passed[index] = True
            index = index + 1
    return passed
