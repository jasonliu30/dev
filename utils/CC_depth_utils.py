from tslearn.metrics import dtw_path
from collections import defaultdict
import numpy as np
from scipy import signal

peak_data = {'index':0,'prominence':0,'width':0,'height':0,'left_base': 0,'right_base':0}

def find_deepest_location(flaw_bonus_peaks_info,same_location_thresh = 1,sequence_thresh = 3):
    """
    Searches for the deepest location in a sequence of frames by identifying trends in peak amplitude.

    The function identifies the location of the peaks in the amplitude and checks for sequences where the peaks are found in the same location.
    If such a sequence is found and meets the defined threshold, the function returns details about the deepest location.

    Parameters:
    ----------
        flaw_bonus_peaks_info (list of list of dict): 
            A nested list containing the peak information for each frame.
        same_location_thresh (int, optional): 
            The threshold for determining if two peaks are in the same location. Defaults to 1.
        sequence_thresh (int, optional): 
            The minimum length of a sequence required for it to be considered. Defaults to 3.

    Returns:
    -------
        found (bool): A boolean indicating whether a valid sequence was found.
        - frame_deepest (int): The index of the frame containing the deepest location.
        - circ_deepest (int): The index within the frame of the deepest location.
        - longest_sequence (int): The length of the longest sequence found.
        If no valid sequence is found, the tuple (False, 0, 0, 0) is returned.
    """
    frame_amps = [[i['height'] for i in ax] for ax in flaw_bonus_peaks_info]
    longest_sequence = 0
    sequence_length = 1
    sequence_start = 0
    peak_width_indices, peak_width_amps = [], []
    previous_index = 0

    for index, frame in enumerate(frame_amps):
        peak_indices, peak_information = signal.find_peaks(frame, width=0)
        widths = peak_information['widths']
        peak_width_index = peak_indices[np.nanargmax(widths)]
        peak_width_indices.append(peak_width_index)
        peak_width_amps.append(frame[peak_width_index])

        if np.abs(previous_index - peak_width_index) <= same_location_thresh:
            if sequence_length == 1:
                sequence_start = max(0, index - 1)
            sequence_length += 1
        else:
            if sequence_length >= sequence_thresh and sequence_length >= longest_sequence:
                longest_sequence = sequence_length
                longest_sequence_start = sequence_start
            sequence_length = 1
            sequence_start = 0
        previous_index = peak_width_index

    if sequence_length >= longest_sequence:
        longest_sequence = sequence_length
        longest_sequence_start = sequence_start

    if longest_sequence >= sequence_thresh:
        largest_amp_sequence = np.argmax(peak_width_amps[longest_sequence_start:longest_sequence_start + longest_sequence])
        return True, largest_amp_sequence + longest_sequence_start, peak_width_indices[longest_sequence_start:longest_sequence_start + longest_sequence][largest_amp_sequence], longest_sequence

    return False, 0, 0, 0

def nan_percentage(arr):
    """
    Calculate the percentage of np.nan in a numpy array.

    Parameters:
    ----------
    arr (numpy.ndarray): The input numpy array.

    Returns:
    -------
    float: The percentage of np.nan in the input numpy array.
    """
    return np.count_nonzero(np.isnan(arr)) / arr.size * 100


def find_largest_consecutive_changes(arr):
    """
    Finds the largest consecutive increases and decreases in an array.
    This function scans through the input array and identifies the largest sequences of consecutive increases
    and decreases in value. It returns the lengths of these sequences and a flag indicating whether the largest
    increase occurs before the largest decrease.

    Parameters:
    ----------
        arr (list of float): 
            The input array of numerical values.

    Returns:
    -------
        - largest_increase (int): 
            The length of the largest sequence of consecutive increases.
        - largest_decrease (int): 
            The length of the largest sequence of consecutive decreases.
        - increase_before_decrease (bool): 
            True if the largest increase occurs before the largest decrease, False otherwise.
    """
    largest_increase = largest_decrease = current_increase = current_decrease = 0
    start_increase = start_decrease = start_largest_increase = start_largest_decrease = 0

    for i in range(1, len(arr)):
        diff = arr[i] - arr[i - 1]
        if diff > 0:
            if current_increase == 0:
                start_increase = i
            current_increase += 1
            current_decrease = 0
        elif diff < 0:
            if current_decrease == 0:
                start_decrease = i
            current_decrease += 1
            current_increase = 0

        if current_increase > largest_increase:
            largest_increase = current_increase
            start_largest_increase = start_increase
        if current_decrease > largest_decrease:
            largest_decrease = current_decrease
            start_largest_decrease = start_decrease

    return largest_increase, largest_decrease, start_largest_increase < start_largest_decrease


def rmv_fwg(a_scan, wave_locations):
    """
    Removes focus wave group region in the given array (a_scan) and replaces it with the median of the array.

    Parameters:
    ----------
        a_scan (numpy.array): 
            The input array from which the wave region is to be removed.
        wave_locations (list of tuple): 
            A list containing a single tuple with two elements, defining the start
                                        and end indices of the wave region to be removed.

    Returns:
    -------
        a_scan_fwg_rmv (numpy.array): 
            A copy of the original array with the specified wave region replaced by its median value.
    """
    a_scan_fwg_rmv = a_scan.copy()
    start, end = wave_locations[0]
    a_scan_fwg_rmv[start:end] = np.median(a_scan)
    
    return a_scan_fwg_rmv


def find_surface_first_peak(config,a_scan_fwg, start):
    """
    Function to find the first peak of a surface from a given A-scan.

    Parameters:
    ----------
        config (dict): 
            Configuration containing amplitude threshold.
        a_scan_fwg (np.ndarray): 
            Input A-scan array.
        start (int): 
            Starting index for consideration.

    Returns:
    -------
        - first_peak_posn (int): Position of the first peak.
        - a_scan_peak (np.ndarray): A-scan with detected peaks.
        - a_scan_cand_peak (np.ndarray): A-scan with candidate peaks based on amplitude threshold.
        - a_scan_select_peak (np.ndarray): A-scan with the selected peak.
        - surface_not_found_reason (str): Reason if surface not found.
        - surface_found (bool): Flag indicating if the surface was found.
        - peak (dict): Dictionary containing peak details.
    """
    surface_found = True
    surface_not_found_reason = None
    a_scan_fwg_trim = a_scan_fwg[:np.argmin(a_scan_fwg)]

    peak_indices, peak_stats = signal.find_peaks(a_scan_fwg_trim, width=2)

    if len(peak_indices) == 0:
        surface_found = False
        surface_not_found_reason = "No peak found in surface"
        return (-1, -1, -1, -1), surface_not_found_reason, surface_found, peak_data.copy()

    a_scan_peak = np.zeros_like(a_scan_fwg) * np.nan
    a_scan_peak[peak_indices] = a_scan_fwg[peak_indices]

    candidate_amp_peak_posn = np.where(a_scan_peak >= config['AMP_THRSH'])[0]

    if len(candidate_amp_peak_posn) == 0:
        surface_found = False
        surface_not_found_reason = f"Amplitude threshold: {config['AMP_THRSH']}, Max amp found: {np.nanmax(a_scan_peak)} condition failed"
        return (-1, -1, -1, -1), surface_not_found_reason, surface_found, peak_data.copy()

    first_peak_posn = candidate_amp_peak_posn[0]

    a_scan_cand_peak = np.zeros_like(a_scan_fwg) * np.nan
    a_scan_cand_peak[candidate_amp_peak_posn] = a_scan_fwg[candidate_amp_peak_posn]

    a_scan_select_peak = np.zeros_like(a_scan_fwg) * np.nan
    a_scan_select_peak[first_peak_posn] = a_scan_fwg[first_peak_posn]

    peak = peak_data.copy()
    arg_peak = np.where(peak_indices == first_peak_posn)[0][0]
    peak['index'] = peak_indices[arg_peak] + start
    peak['prominence'] = peak_stats['prominences'][arg_peak]
    peak['width'] = peak_stats['widths'][arg_peak]
    peak['height'] = a_scan_fwg[peak_indices[arg_peak]]
    peak['left_base'] = peak_stats['left_bases'][arg_peak] + start
    peak['right_base'] = peak_stats['right_bases'][arg_peak] + start
    peak['start'] = start

    first_peak_posn += start
    return (first_peak_posn, a_scan_peak, a_scan_cand_peak, a_scan_select_peak), surface_not_found_reason, surface_found, peak

# pragma: no cover
def find_extra_peak_flaw_ascan(A_scan, peak_posn, amplitude_threshold=140, prominence_threshold=20, width_max=10):
    """
    Function to find peaks that may be missed by DTW due to a more similar peak being found.

    Parameters:
    ----------
        A_scan (np.ndarray): Input A-scan array.
        peak_posn (int): Position of the main peak in the A-scan.
        amplitude_threshold (int, optional): Amplitude threshold for peak detection. Defaults to 140.
        prominence_threshold (int, optional): Prominence threshold for peak detection. Defaults to 20.
        width_max (int, optional): Maximum width for peak detection. Defaults to 10.

    Returns:
    -------
        int: Position of the chosen peak or the original peak position if no extra peak is found.
    """
    peak_indices, peak_stats = signal.find_peaks(A_scan, width=2, prominence=prominence_threshold, height=amplitude_threshold)
    candidate_peaks = peak_indices[peak_indices < peak_posn]
    candidate_widths = peak_stats['widths'][candidate_peaks]
    
    if candidate_peaks.size > 0 and np.any(candidate_widths < width_max):
        candidate_peak_amplitudes = A_scan[candidate_peaks[candidate_widths < width_max]]
        chosen_peak = candidate_peaks[np.argmax(candidate_peak_amplitudes)]
        return chosen_peak
    
    return peak_posn

    
def remove_median(signal):
    """
    Removes the median value from a given signal.

    Parameters:
    ----------
        signal (np.ndarray): Input signal array.

    Returns:
    -------
        signal (np.ndarray): Modified signal with the median value subtracted.
    """
    signal = signal - np.median(signal)
    return signal

def find_distance(a, b, normalize):
    """
    Finds the Euclidean distance between two vectors a and b. Optionally normalizes the vectors before calculating.

    Parameters:
    ----------
        a (np.ndarray): The first vector.
        b (np.ndarray): The second vector.
        normalize (bool): If True, normalizes the vectors before calculating the distance.

    Returns:
    -------
        float: The Euclidean distance between the vectors.
    """
    if normalize:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)

    return np.linalg.norm(a - b)

def find_cost_dtw(config, a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_a_scan_idxs, a_scan_idxs, cost, normalize, focus):
    """
    Calculates the cost of dynamic time warping (DTW) between two signals, a_scan_fwg and surface_a_scan_fwg.
    
    Parameters:
    ----------
        config (dict): Configuration parameters including 'DTW_RANGE_CONSTANT'.
        a_scan_fwg (np.ndarray): First signal for comparison.
        surface_a_scan_fwg (np.ndarray): Second signal for comparison.
        surface_peak_posn (int): Position of the surface peak.
        surface_a_scan_idxs (np.ndarray): Indices of the surface A-scan.
        a_scan_idxs (np.ndarray): Indices of the A-scan.
        cost (float): Current cost value.
        normalize (bool): If True, normalizes the vectors before calculating the distance.
        focus (bool): If True, focuses on a specific region around the peak position.

    Returns:
    -------
        float or tuple: The calculated cost. Returns additional values when focus and normalize are True.
    """
    
    if not focus and normalize:
        cost = find_distance(surface_a_scan_fwg[surface_a_scan_idxs], a_scan_fwg[a_scan_idxs], normalize)
        return cost
        
    if focus and normalize:
        surface_a_scan_peak_idxs = surface_a_scan_idxs[(surface_a_scan_idxs >= surface_peak_posn - config['DTW_RANGE_CONSTANT']) & (surface_a_scan_idxs <= surface_peak_posn + config['DTW_RANGE_CONSTANT'])]
        a_scan_peak_idxs = a_scan_idxs[(surface_a_scan_idxs >= surface_peak_posn - config['DTW_RANGE_CONSTANT']) & (surface_a_scan_idxs <= surface_peak_posn + config['DTW_RANGE_CONSTANT'])]
        cost = find_distance(surface_a_scan_fwg[surface_a_scan_peak_idxs], a_scan_fwg[a_scan_peak_idxs], normalize)
        
        surface_range_fix = 2 * config['DTW_RANGE_CONSTANT'] + 1
        flaw_range = a_scan_fwg[a_scan_peak_idxs].shape[0]
        
        return cost, surface_range_fix, flaw_range
    

def find_lag(config,a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, start, stop, trim = 0,title=''):
    """
    Finds the lag between two signals using dynamic time warping (DTW).

    Parameters:
    ----------
        config (dict): Configuration parameters including various thresholds and constants.
        a_scan_fwg (np.ndarray): First signal for comparison.
        surface_a_scan_fwg (np.ndarray): Second signal for comparison.
        surface_peak_posn (int): Position of the surface peak.
        start (int): Start index for the range of consideration.
        stop (int): Stop index for the range of consideration.
        trim (int, optional): Number of elements to trim from both ends of the signals. Defaults to 0.
        title (str, optional): Title for any plots or visualizations. Defaults to ''.

    Returns:
    -------
        - surface_peak_posn (int): Position of the surface peak.
        - surface_peak_value (float): Value of the surface peak.
        - flaw_peak_posn (int): Position of the flaw peak.
        - flaw_peak_value (float): Value of the flaw peak.
        - lag (int): Difference between surface and flaw peak positions.
        - path (np.ndarray): Path of the DTW alignment.
        - cost_normalized (float): Normalized cost of DTW without focus.
        - cost_focus_normalized (float): Normalized cost of DTW with focus.
        - surface_range_fix (int): Fixed range around the surface peak.
        - flaw_range (int): Range of the flaw peak.
    """
    x = np.arange(start + trim, stop  - trim)
    surface_peak_posn = surface_peak_posn - start
    check_left_peak = config['EXTRA_PEAK']
    if trim != 0:
        y1 = surface_a_scan_fwg[trim:-trim ]
        y2 = a_scan_fwg[trim:-trim]
    else:
        y1 = surface_a_scan_fwg
        y2 = a_scan_fwg
    
    path, cost = dtw_path(y1, y2)
    path = np.array(path)
    
    surface_a_scan_idxs, a_scan_idxs = path[:, 0], path[:, 1]
    
    cost_normalized = find_cost_dtw(config,a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_a_scan_idxs, a_scan_idxs, cost, True, False)
    # cost calculated useng the nearby (3) points to surface peak positions
    cost_focus_normalized, surface_range_fix, flaw_range = find_cost_dtw(config,a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_a_scan_idxs, a_scan_idxs, cost, True, True)
    
    # find surface peak in "surface_a_scan_idxs"
    surface_peak_match_idx = np.where(surface_a_scan_idxs == surface_peak_posn)[0]
    
    # convert the indexes w.r.t start
    surface_peak_posns = x[surface_a_scan_idxs[surface_peak_match_idx]]
    # find surface peak in "a_scan_idxs"
    flaw_peak_posns = x[a_scan_idxs[surface_peak_match_idx]]
    flaw_peak_values = y2[a_scan_idxs[surface_peak_match_idx]]
    
    # find the corresponding difference between positions
    diff = surface_peak_posns - flaw_peak_posns
    lag_idx = np.argmax(flaw_peak_values)
    
    # find all w.r.t max diff
    lag = diff[lag_idx]
    flaw_peak_posn = flaw_peak_posns[lag_idx]
    surface_peak_posn = surface_peak_posns[lag_idx]
    flaw_peak_value = flaw_peak_values[lag_idx]
    surface_peak_value = y1[surface_a_scan_idxs[surface_peak_match_idx]][lag_idx]

    if check_left_peak:
        # Finding missed peaks using thresholds
        if config['EXTRA_PEAK_SURFACE_CRITERIA']:
            surface_peak_new = find_extra_peak_flaw_ascan(surface_a_scan_fwg,surface_peak_posn-start,config['EXTRA_PEAK_AMP_THRESH_SURFACE'],config['EXTRA_PEAK_PROMINENCE_THRESH_SURFACE'],title=title+'_surface',width_max=config['EXTRA_PEAK_WIDTH_MAX']) + start
        else:
            surface_peak_new=surface_peak_posn

        if surface_peak_new==surface_peak_posn:
            flaw_peak_posn_new = find_extra_peak_flaw_ascan(a_scan_fwg,flaw_peak_posn-start,config['EXTRA_PEAK_AMP_THRESH'],config['EXTRA_PEAK_PROMINENCE_THRESH'],title=title,width_max=config['EXTRA_PEAK_WIDTH_MAX']) + start
            if flaw_peak_posn_new!= flaw_peak_posn:
                if a_scan_fwg[flaw_peak_posn_new-start]/a_scan_fwg[flaw_peak_posn-start] >=config['EXTRA_PEAK_AMP_RATIO']:    
                    lag = surface_peak_posns[lag_idx] - flaw_peak_posn_new
                    if config['EXTRA_PEAK_OVERRIDE_CONSTRAINT'] and (flaw_peak_posn-flaw_peak_posn_new>config['EXTRA_PEAK_SAME_PEAK_AREA']):
                        flaw_peak_value = 255

    return surface_peak_posn, surface_peak_value, flaw_peak_posn, flaw_peak_value, lag, path, cost_normalized, cost_focus_normalized, surface_range_fix, flaw_range

def find_surface_posn_circ(config,apc_data_arr, ax, pred_circ_start, pred_circ_end, fwg_shear, row, flattened_lags,flaw_dic_ax, range_circ = 5,lag_difference_thresh =2,flip_ascans=False ):
    """
    Finds the position of the surface in a given circumferential range.

    Parameters:
    ----------
        config (dict): config file
        apc_data_arr (array): Array containing APC data.
        ax (int): The axial location under consideration.
        pred_circ_start (int): Predicted starting circumferential position.
        pred_circ_end (int): Predicted ending circumferential position.
        fwg_shear (array): focus wave-group Shear information.
        row (dict): row being processed.
        flattened_lags (list): List containing lag information.
        flaw_dic_ax (dict): Dictionary containing axial flaw location data.
        range_circ (int, optional): Range of the circumferential search. Defaults to 5.
        lag_difference_thresh (int, optional): Threshold for the lag difference. Defaults to 2.
        flip_ascans (bool, optional): Flag to flip A-scans. Defaults to False.

    Returns:
    -------
        surface_peak_posn (int): Position of the surface peak.
        surface_peak_posn_up (int): Upper position of the surface peak.
        surface_peak_posn_bottom (int): Bottom position of the surface peak.
        surface_a_scan (array): A-scan of the surface.
        surface_wave_loc (array): Surface wave location.
        surface_circ (int): Circ location of the surface.
        surface_ax (int): Axial location of the surface.
        peaks (tuple): Information about the peaks found.
        surface_start (int): Start position of the surface.
        surface_stop (int): Stop position of the surface.
        surface_not_found_reason (str): Reason if the surface is not found.
        surface_found (bool): Flag indicating if the surface was found.
        surface_stats (dict): Statistics related to the surface.

    Raises:
    ------
        Exception: If the surface is not found.
    """
    surface_peak_pos_list = [] # contains surface_peak_pos for every rotary loc in specific frame
    surface_a_scan_list = []
    surface_wave_loc_list = []
    peak_list = []
    surface_start_list = []
    surface_stop_list = []
    surface_found_list = []
    surface_circ_list = []
    surface_ax_list = []
    surface_not_found_reason_list = []
    surface_stat_list = []
    
    try:
        flaw_loc_scan = flaw_dic_ax[row['Filename']]
        flaw_loc_ax = [item for sublist in flaw_loc_scan[ax] for item in sublist]
    except:
        flaw_loc_ax = []
            
    n_up = n_bottom = 0
    circ_up = pred_circ_start - n_up - 1
    circ_bottom = pred_circ_end + n_bottom + 1

    while len(surface_circ_list) < 2 * range_circ and (circ_up > 0 and circ_bottom < apc_data_arr.shape[1]):
        for circ in [circ_up, circ_bottom]:
            if circ not in flaw_loc_ax and flattened_lags[circ] <= lag_difference_thresh:
                surface_a_scan = ((apc_data_arr[ax, circ, :] - config['MEDIAN_VALUE']) * (-1 if flip_ascans else 1)) + config['MEDIAN_VALUE']
                surface_start, surface_stop, *_ = fwg_shear[ax][circ]
                surface_a_scan_fwg = surface_a_scan[surface_start: surface_stop]
                peak_info, surface_not_found_reason, surface_found, peak_stats = find_surface_first_peak(config, surface_a_scan_fwg, surface_start)
                surface_peak_posn, *_ = peak_info
                surface_peak_pos_list.append(surface_peak_posn)
                surface_a_scan_list.append(surface_a_scan)
                surface_wave_loc_list.append(fwg_shear[ax][circ])
                peak_list.append(peak_info)
                surface_start_list.append(surface_start)
                surface_stop_list.append(surface_stop)
                surface_found_list.append(surface_found)
                surface_circ_list.append(circ)
                surface_ax_list.append(ax)
                surface_stat_list.append(peak_stats)

                if not surface_found:
                    surface_not_found_reason_list.append(f"{surface_not_found_reason}, frame: {ax + 1}, circ: {circ}")

        n_up += 1
        n_bottom += 1
        circ_up = pred_circ_start - n_up - 1
        circ_bottom = pred_circ_end + n_bottom + 1

    surface_peak_pos_list = np.array(surface_peak_pos_list)
    
    surface_peak_pos_valid_list = np.zeros_like(surface_peak_pos_list) * np.nan
    surface_peak_pos_valid_list[surface_found_list] = surface_peak_pos_list[surface_found_list]
    
    surface_peak_pos_up_list = surface_peak_pos_valid_list[np.array(surface_circ_list) < pred_circ_start]
    surface_peak_pos_bottom_list = surface_peak_pos_valid_list[np.array(surface_circ_list) > pred_circ_end]
    
    try:
        if config['SURFACE_CLOSEST']:
            surface_circ = np.where(surface_found_list)[0][0]
        else:
            surface_circ = np.nanargmax(surface_peak_pos_valid_list)
        
    except:
        surface_found = False
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, '\n'.join(surface_not_found_reason_list), surface_found,peak_data.copy()
    surface_peak_posn = surface_peak_pos_valid_list[surface_circ]
    
    try:
        surface_circ_up = np.nanargmax(surface_peak_pos_up_list)
        surface_peak_posn_up = surface_peak_pos_up_list[surface_circ_up]
    except:
        surface_peak_posn_up = -1
        
    try:
        surface_circ_bottom = np.nanargmax(surface_peak_pos_bottom_list)
        surface_peak_posn_bottom = surface_peak_pos_bottom_list[surface_circ_bottom]
    except:
        surface_peak_posn_bottom = -1

    surface_stats = surface_stat_list[surface_circ]
    surface_a_scan = surface_a_scan_list[surface_circ]
    surface_wave_loc =  surface_wave_loc_list[surface_circ]
    peaks = peak_list[surface_circ]

    surface_start = surface_start_list[surface_circ]
    surface_stop = surface_stop_list[surface_circ]
    surface_found = surface_found_list[surface_circ]
    surface_ax = surface_ax_list[surface_circ]
    surface_circ = surface_circ_list[surface_circ]

    return surface_peak_posn, surface_peak_posn_up, surface_peak_posn_bottom, surface_a_scan, surface_wave_loc, surface_circ, surface_ax, peaks, surface_start, surface_stop, surface_not_found_reason, surface_found,surface_stats

def find_all_surface_posns_circ(config,apc_data_arr, ax, pred_circ_start, pred_circ_end, fwg_shear, row, flattened_lags,flaw_dic_ax, range_circ = 5,lag_difference_thresh =2,flip_ascans=False ):
    """
    Finds all the positions of the surface in a given circumferential range.

    Parameters:
    ----------
        config (dict): config file
        apc_data_arr (array): Array containing APC data.
        ax (int): The axial location under consideration.
        pred_circ_start (int): Predicted starting circumferential position.
        pred_circ_end (int): Predicted ending circumferential position.
        fwg_shear (array): focus wave-group information.
        row (dict): row being processed.
        flattened_lags (list): List containing lag information.
        flaw_dic_ax (dict): Dictionary containing flaw location data.
        range_circ (int, optional): Range of the circumferential search. Defaults to 5.
        lag_difference_thresh (int, optional): Threshold for the lag difference. Defaults to 2.
        flip_ascans (bool, optional): Flag to flip A-scans. Defaults to False.

    Returns:
    -------
        surface_peak_pos_list (list): List of positions of the surface peak.
        surface_a_scan_list (list): List of A-scans of the surface.
        surface_wave_loc_list (list): List of surface wave locations.
        surface_circ_list (list): List of circumferential locations of the surface.
        surface_ax_list (list): List of axial locations of the surface.
        peak_list (list): List of information about the peaks found.
        surface_start_list (list): List of start positions of the surface.
        surface_stop_list (list): List of stop positions of the surface.
        surface_not_found_reason (str): Reason if the surface is not found.
        surface_found_list (list): List of flags indicating if the surface was found.

    Raises:
        Exception: If the surface is not found.

    """
    surface_peak_pos_list = [] # contains surface_peak_pos for every rotary loc in specific frame
    surface_a_scan_list = []
    surface_wave_loc_list = []
    peak_list = []
    surface_start_list = []
    surface_stop_list = []
    surface_found_list = []
    surface_circ_list = []
    surface_ax_list = []
    surface_not_found_reason_list = []
        
    try:
        flaw_loc_scan = flaw_dic_ax[row['Filename']]
        flaw_loc_ax = [item for sublist in flaw_loc_scan[ax] for item in sublist]
    except:
        flaw_loc_ax = []

    n_up = n_bottom = 0
    circ_up = pred_circ_start - n_up - 1
    circ_bottom = pred_circ_end + n_bottom + 1

    while len(surface_circ_list) < 2 * range_circ and (circ_up > 0 and circ_bottom < apc_data_arr.shape[1]):
        for circ in [circ_up, circ_bottom]:
            if circ not in flaw_loc_ax and flattened_lags[circ] <= lag_difference_thresh:
                surface_a_scan = ((apc_data_arr[ax, circ, :] - config['MEDIAN_VALUE']) * (-1)) + config['MEDIAN_VALUE'] if flip_ascans else apc_data_arr[ax, circ, :]
                surface_start, surface_stop, *_ = fwg_shear[ax][circ]
                surface_a_scan_fwg = surface_a_scan[surface_start: surface_stop]
                peak_info, surface_not_found_reason, surface_found, peak_stats = find_surface_first_peak(config, surface_a_scan_fwg, surface_start)

                surface_peak_posn, *_ = peak_info
                surface_peak_pos_list.append(surface_peak_posn)
                surface_a_scan_list.append(surface_a_scan)
                surface_wave_loc_list.append(fwg_shear[ax][circ])
                peak_list.append(peak_info)
                surface_start_list.append(surface_start)
                surface_stop_list.append(surface_stop)
                surface_found_list.append(surface_found)
                surface_circ_list.append(circ)
                surface_ax_list.append(ax)
                if not surface_found:
                    surface_not_found_reason_list.append(f"{surface_not_found_reason}, frame: {ax + 1}, circ: {circ}")

        n_up += 1
        n_bottom += 1
        circ_up = pred_circ_start - n_up - 1
        circ_bottom = pred_circ_end + n_bottom + 1
    
    surface_peak_pos_valid_list = np.zeros_like(surface_peak_pos_list) * np.nan
    surface_peak_pos_valid_list[surface_found_list] = surface_peak_pos_list[surface_found_list]
    try:
        surface_circ = np.nanargmax(surface_peak_pos_valid_list)
    except:
        return -1, -1, -1, -1, -1, -1, -1, -1, '\n'.join(surface_not_found_reason_list), False

    surface_peak_pos_list = np.array(surface_peak_pos_list)
    surface_a_scan_list= np.array(surface_a_scan_list)
    surface_wave_loc_list= np.array(surface_wave_loc_list)
    peak_list= np.array(peak_list)
    surface_start_list= np.array( surface_start_list)
    surface_stop_list= np.array(surface_stop_list)
    surface_found_list= np.array(surface_found_list)
    surface_circ_list= np.array(surface_circ_list)
    surface_ax_list = np.array(surface_ax_list)
                
    if sum(surface_found_list)>config['DEPTH_AVERAGING_MAX_LOCATIONS']:
        order=np.argsort(surface_peak_pos_list[surface_found_list])[::-1]
        surface_found_list[order[config['DEPTH_AVERAGING_MAX_LOCATIONS']:]]=False

                
    return surface_peak_pos_list[surface_found_list], surface_a_scan_list[surface_found_list], surface_wave_loc_list[surface_found_list],\
           surface_circ_list[surface_found_list], surface_ax_list[surface_found_list], peak_list[surface_found_list], surface_start_list[surface_found_list],\
           surface_stop_list[surface_found_list], surface_not_found_reason, surface_found_list

def find_surface_posn_ax(config,apc_data_arr, circ, pred_ax_start, pred_ax_end, fwg_shear, row, flaw_dic_circ, range_ax = 5,flip_ascans=False):
    """
    Finds the surface positions in the axial direction for a given circumferential location.

    Parameters:
    ----------
        config (dict): Configuration parameters.
        apc_data_arr (array): APC data array containing the scan data.
        circ (int): The circumferential location being considered.
        pred_ax_start (int): Predicted start of the axial region.
        pred_ax_end (int): Predicted end of the axial region.
        fwg_shear (dict): Information about shear waves.
        row (dict): Row information.
        flaw_dic_circ (dict): Dictionary containing flaw locations in the circumferential direction.
        range_ax (int, optional): Range for the axial search. Default is 5.
        flip_ascans (bool, optional): Flag to determine whether to flip the A-scans. Default is False.

    Returns:
    -------
        Contains information about the surface peak positions, A-scans, wave locations, circumferential locations,
        axial locations, peaks, starts, stops, not-found reason, and found flag. Returns a tuple of -1 values and
        False flag if no surface positions are found.
    """
    surface_peak_pos_list = [] # contains surface_peak_pos for every rotary loc in specific frame
    surface_a_scan_list = []
    surface_wave_loc_list = []
    peak_list = []
    surface_start_list = []
    surface_stop_list = []
    surface_found_list = []
    surface_ax_list = []
    surface_circ_list = []
    surface_not_found_reason_list = []
    range_ax = config['SURFACE_AX_RANGE']
    n_left = 0
    n_right = 0
    
    try:
        flaw_loc_scan = flaw_dic_circ[row['Filename']]
        flaw_loc_circ = flaw_loc_scan[circ]
        
        flaw_loc_circ = [item for sublist in flaw_loc_circ for item in sublist]
        
    except:
        flaw_loc_circ = []
   
    ax_left = pred_ax_start - n_left - 1
    ax_right = pred_ax_end + n_right + 1
    while len(surface_ax_list) < 2*range_ax and (ax_left>0 and ax_right<apc_data_arr.shape[0]):
        for ax in [ax_left, ax_right]:
            
            if ax not in flaw_loc_circ:
                if flip_ascans:
                    surface_a_scan=((apc_data_arr[ax, circ, :]- config['MEDIAN_VALUE'])*(-1))+config['MEDIAN_VALUE']
                else:
                    surface_a_scan = apc_data_arr[ax, circ, :]
                (surface_start, surface_stop, _, _, _, _) = fwg_shear[ax][circ]
                surface_a_scan_fwg = surface_a_scan[surface_start : surface_stop]
                peak_info, surface_not_found_reason, surface_found,peak = find_surface_first_peak(config,surface_a_scan_fwg, surface_start)
                
                (surface_peak_posn, _, _, _) = peak_info
                surface_peak_pos_list.append(surface_peak_posn)
                surface_a_scan_list.append(surface_a_scan)
                surface_wave_loc_list.append(fwg_shear[ax][circ])
                peak_list.append(peak_info)
                surface_start_list.append(surface_start)
                surface_stop_list.append(surface_stop)
                surface_found_list.append(surface_found)
                surface_ax_list.append(ax)
                surface_circ_list.append(circ)
                
                if not surface_found:
                    surface_not_found_reason_list.append(surface_not_found_reason + f", frame: {ax+1}, circ: {circ}")
                
        n_left+=1
        n_right+=1

        ax_left = pred_ax_start - n_left - 1
        ax_right = pred_ax_end + n_right + 1
    
    surface_peak_pos_list = np.array(surface_peak_pos_list)
    
    surface_peak_pos_valid_list = np.zeros_like(surface_peak_pos_list) * np.nan
    surface_peak_pos_valid_list[surface_found_list] = surface_peak_pos_list[surface_found_list]
    
    surface_peak_pos_left_list = surface_peak_pos_valid_list[np.array(surface_ax_list) < pred_ax_start]
    surface_peak_pos_right_list = surface_peak_pos_valid_list[np.array(surface_ax_list) > pred_ax_end]
    
    try:
        surface_ax = np.nanargmax(surface_peak_pos_valid_list)
        
    except:
        surface_found = False
        return -1, -1, -1, -1, -1, -1, -1, -1, -1, -1, '\n'.join(surface_not_found_reason_list), surface_found
    surface_peak_posn = surface_peak_pos_valid_list[surface_ax]
    
    try:
        surface_ax_left = np.nanargmax(surface_peak_pos_left_list)
        surface_peak_posn_left = surface_peak_pos_left_list[surface_ax_left]
    except:
        surface_peak_posn_left = -1
        
    try:
        surface_ax_right = np.nanargmax(surface_peak_pos_right_list)
        surface_peak_posn_right = surface_peak_pos_right_list[surface_ax_right]
    except:
        surface_peak_posn_right = -1
    surface_a_scan = surface_a_scan_list[surface_ax]
    surface_wave_loc =  surface_wave_loc_list[surface_ax]
    peaks = peak_list[surface_ax]
    surface_start = surface_start_list[surface_ax]
    surface_stop = surface_stop_list[surface_ax]
    surface_found = surface_found_list[surface_ax]
    surface_circ = surface_circ_list[surface_ax]
    surface_ax = surface_ax_list[surface_ax]
    
    return surface_peak_posn, surface_peak_posn_left, surface_peak_posn_right, surface_a_scan, surface_wave_loc, surface_circ, surface_ax, peaks, surface_start, surface_stop, surface_not_found_reason, surface_found

def flaw_loc_file(df):
    """
    Generates dictionaries containing flaw locations in axial and circumferential directions based on the provided DataFrame.

    Parameters:
    ----------
        df (Pandas DataFrame): DataFrame containing information about flaw locations, including Filename, Ax Start, Ax End,
                        Ro Start, and Ro End.

    Returns:
    -------
        Two dictionaries containing flaw locations in axial (flaw_dic_ax) and circumferential (flaw_dic_circ) directions.
    """
    flaw_dic_ax = defaultdict(lambda: defaultdict(list))
    flaw_dic_circ = defaultdict(lambda: defaultdict(list))
    
    for file in df['Filename'].unique():
        selected_df = df.loc[df['Filename'] == file]
        
        for _, row in selected_df.iterrows():
            pred_ax_start, pred_ax_end = row['Ax Start'], row['Ax End']
            pred_circ_start, pred_circ_end = row['Ro Start'], row['Ro End']

            for ax in range(pred_ax_start, pred_ax_end + 1):
                flaw_dic_ax[file][ax].append(list(range(pred_circ_start, pred_circ_end + 1)))

            for circ in range(pred_circ_start, pred_circ_end + 1):
                flaw_dic_circ[file][circ].append(list(range(pred_ax_start, pred_ax_end + 1)))

    return flaw_dic_ax, flaw_dic_circ

def swap_axes(input_list):
    """
    Swap the axes of a 2D list.

    Parameters:
    ----------
        input_list : list of lists
            The 2D list whose axes need to be swapped.

    Returns:
    -------
        swapped_list : list of lists
            The 2D list after swapping the axes.
    """
    # Get the dimensions of the input list
    rows = len(input_list)
    cols = len(input_list[0])
    
    # Create a new list to store the swapped axes
    swapped_list = [[0] * rows for _ in range(cols)]
    
    # Swap the axes
    for i in range(rows):
        for j in range(cols):
            swapped_list[j][i] = input_list[i][j]
    
    return swapped_list