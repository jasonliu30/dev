import numpy as np
import pandas as pd
import os
from tslearn.metrics import dtw_path
from scipy import signal
from depth_sizing.utils.flattening import get_pressure_tube_surface, Smoothlags, Smoothlags_remove_flaws


def find_signal_invert(signal):
    
    """This function invert the signal
    Args:
        signal(np.array): any signal
            
    Returns:
        signal(np.array): inverted signal
           
    """
    
    
    med = np.median(signal, axis = 2)
    med = np.expand_dims(med, axis = 2)
    signal = -signal + 2*med
    
    return signal 

def flaw_loc_file(df, nb_data_arr):
    
    """This function stores flaw location axially and circumferentially: used for smart surface selection, no surface belongs to the other flaw extent

    Args:
        df(dataframe): dataframe of every flaw to size the depth of, must all be within the same B-scan file. required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)

    Returns:
        flaw_dic_ax(dictionary): contains circ locations of all flaws inside a frame
        flaw_dic_circ(dictionary): contains ax locations of all flaws for a circ loc
        
    Example:
        File 'A' contains two flaws inside the frame '1' with circ start and end respectively [10, 12] and [30, 33]
        flaw_dic_ax['A']['1'] = [[10, 11, 12], [30, 31, 32, 33]]
       
    """
    flaw_dic_ax = {}
    flaw_dic_circ = {}
    
    for file in df.iloc[:]['Filename'].unique():
        flaw_dic_ax[file] = {}
        flaw_dic_circ[file] = {}
        
        selected_df = df.loc[df['Filename'] == file]
        for index, row in selected_df.iterrows():
            pred_ax_start = row['Ax Start']
            pred_ax_end = row['Ax End'] if row['Ax End'] < nb_data_arr.shape[0] else nb_data_arr.shape[0] - 1 # clipping the pred ax end to max ax end
            pred_circ_start = row['Ro Start']
            pred_circ_end = row['Ro End'] if row['Ro End'] < nb_data_arr.shape[1] else nb_data_arr.shape[1] - 1 # clipping the pred ro end to max ro end

            for ax in range(pred_ax_start, pred_ax_end + 1):
                try:
                    flaw_dic_ax[file][ax]
                except:
                    flaw_dic_ax[file][ax] = []
                flaw_dic_ax[file][ax].append(list(np.arange(pred_circ_start, pred_circ_end + 1)))
                
            for circ in range(pred_circ_start, pred_circ_end + 1):
                try:
                    flaw_dic_circ[file][circ]
                except:
                    flaw_dic_circ[file][circ] = []

                flaw_dic_circ[file][circ].append(list(np.arange(pred_ax_start, pred_ax_end + 1)))
                    
    return flaw_dic_ax, flaw_dic_circ

def find_surface_feature(surface_a_scan_fwg, start, peak_type, config):
    
        
    """This function find the surface feature/peak based on the peak_type
    Args:
        surface_a_scan_fwg(np.array): FWG of Surface a-scan
        start(int): Start of the Surface a-scan FWG
        peak_type(str):
            - 'left_most': to select left-most peak
            - 'max': to select peak with max amp
            
    Returns:
        peak_posn(int): Selected surface peak position
        a_scan_all_peak(np.array): All peaks present in the surface a-scan
        a_scan_cand_peak(np.array): Candidate peaks present in the surface a-scan
        a_scan_select_peak(np.array): Selected peak present in the surface a-scan
        surface_not_found_reason(str): Reason for not founding the surface
        surface_found(bool): True, if surface found
           
    """
    
    # initialize variables
    surface_found = True
    surface_not_found_reason = None
    
    # find all peaks in surface
    peak_indices, _ = signal.find_peaks(surface_a_scan_fwg, width = 2)

    # if no peak found
    if len(peak_indices) == 0:
        surface_found = False
        
        surface_not_found_reason = "No peak found in surface"

        return (-1, -1, -1, -1), surface_not_found_reason, surface_found
    
    # array of size same as surface a-scan with value only at peak indices
    a_scan_all_peak = np.zeros_like(surface_a_scan_fwg) * np.nan
    a_scan_all_peak[peak_indices] = surface_a_scan_fwg[peak_indices]
    
    # find peaks greater than the threshold
    candidate_amp_peak_posn = np.where(a_scan_all_peak >= config['SURFACE_AMP_THRSH'])[0]
    
    # if no peak amp is greater than the thresh
    if len(candidate_amp_peak_posn) == 0:
        surface_found = False
        
        surface_not_found_reason = f"Amplitude threshold: {config['SURFACE_AMP_THRSH']}, Max amp found: {np.nanmax(a_scan_all_peak)} condition failed"

        return (-1, -1, -1, -1), surface_not_found_reason, surface_found
    
    # array of size same as surface a-scan with value only at candidate peak indices
    a_scan_cand_peak = np.zeros_like(surface_a_scan_fwg) * np.nan
    a_scan_cand_peak[candidate_amp_peak_posn] = surface_a_scan_fwg[candidate_amp_peak_posn]
    
    # select the surface feature from the candidate peaks
    if peak_type == 'left_most':
        peak_posn = candidate_amp_peak_posn[0]
        a_scan_select_peak = np.zeros_like(surface_a_scan_fwg) * np.nan
        a_scan_select_peak[peak_posn] = surface_a_scan_fwg[peak_posn]
        
    if peak_type == 'max':
        surface_found = True
        surface_not_found_reason = None
        peak_posn = np.where(surface_a_scan_fwg == np.max(surface_a_scan_fwg))[0]
        
        # if multiple peaks have max amp use the left most
        try:
            peak_posn = peak_posn[0]
        except:
            pass
        
        # array of size same as surface a-scan with value only at selected peak/surface feature
        a_scan_select_peak = np.zeros_like(surface_a_scan_fwg) * np.nan
        a_scan_select_peak[peak_posn] = surface_a_scan_fwg[peak_posn]
    
    # change peak position w.r.t start
    peak_posn += start

    # surface feature info: selected peak, candidate peak, all peaks
    surface_feature_info = (peak_posn, a_scan_all_peak, a_scan_cand_peak, a_scan_select_peak)

    return surface_feature_info, surface_not_found_reason, surface_found

def remove_abrupt_change(depth_list, max_diff):
    
    """This function remove any sudden change in the list

    Args:
        depth_list(list): contains depth for all circ locs inside a single frame
        max_diff(float): max differnce allowed between neighbor values
        
    Returns:
        depth_list(list): contains depth for all circ locs inside a single frame, after removing the sudden changes
       
    """
    
    # remove any single sudden change in depth
    try:
        if len(np.where(np.diff(depth_list) >= max_diff)[0]) > 0:
            depth_list[np.where(np.diff(depth_list) >= max_diff)[0][0] + 1] = np.nan
            
        if len(np.where(np.diff(depth_list) <= -max_diff)[0]) > 0:
            depth_list[np.where(np.diff(depth_list) <= -max_diff)[0][0]] = np.nan
            
    except:
        pass
    
    return depth_list

def units_depth(surface_peak_posn, flaw_peak_posn, curvature_correction, config):
    
    """This function finds the depth using the lag between surface_peak_posn and flaw_peak_posn

    Args:
        surface_peak_posn(int): surface feature/peak position
        flaw_peak_posn(int): flaw feature/peak position
        curvature_correction(float): curvature correction required sue to ovality of PT
            
    Returns:
        depth(float)
           
    """
    
    units = (flaw_peak_posn - surface_peak_posn) - curvature_correction
    
    units = min(config['MAX_PEAK_DIFF'], units)
    
    tof = units * config['UNIT_MICRO_SEC']
    depth = max(np.round(tof * config['MICRO_SEC_depth_NB'], 4), 0)
    
    return depth

def find_wavegroups_in_frame(a_scan, REFLECTION_AMP_THRSH = None, 
                             number_of_wavegroups=2,
                             amplitude_threshold=0, wavegroup_width=300,
                             max_peak_distances=100, median_value=128, n_multi_reflections = 0,
                             multi_reflection = False):
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
    
    scan_bounds = [0, 0]
    wg_dtype = np.dtype([('start_index', 'i2'), ('stop_index', 'i2'), ('peak_index', 'i2'), ('peak_value', 'i2'), ('average_value', 'f4'), ('area_under_curve', 'f4')])
    wavegroup_locations = np.zeros(0, dtype=wg_dtype)
    
    if np.max(a_scan) < amplitude_threshold:
        pass
    
    else:

        median_scan = a_scan - median_value
        if scan_bounds[1] != 0:
            a_scan_subset = a_scan[scan_bounds[0]:scan_bounds[1]].copy()
            median_scan_subset = median_scan[scan_bounds[0]:scan_bounds[1]].copy()
        else:
            a_scan_subset = a_scan.copy()
            median_scan_subset = median_scan.copy()
    
        # Convert width to half-width, to it can easily be added/subtracted from positions
        wavegroup_width = int(wavegroup_width / 2)
    
        ########
        # Wavegroup detection
        ########
        for i in range(number_of_wavegroups):
            
            if multi_reflection and n_multi_reflections >= 4:
                # print('Multiple refelections: choosing 1st peak')
                peak_idxs, _ = signal.find_peaks(a_scan, width = 2, height = REFLECTION_AMP_THRSH)
                peak_posn = np.min(peak_idxs)
            else:
                peak_posn = np.argmax(median_scan_subset)
    
            # Take subset of the scan based on the wavegroup widths. Check to make sure the wavegroup width doesn't exceed the scan indices.
            scan = a_scan_subset[max(0, peak_posn - wavegroup_width): min(len(a_scan_subset) - 1, peak_posn + wavegroup_width)]

            # Take median value of the scan, then find the peak_posn value
            scan = scan - median_value
            peak_value = np.nanmax(scan)
    
            # Find the index of the last peak_posn location
            peak_indices = np.argwhere(scan == peak_value)
            peak_index = np.max(peak_indices)
    
            # Shift the peak_posn index if necessary. pos_peak refers to the index in `scan`, which might be a subset of `a_scan`
            peak_posn = peak_index + max(0, peak_posn - wavegroup_width)
    
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
            
    return wavegroup_locations

def find_surface(nb_data_arr, flaw_dic_ax, flattened_lags, ax, circ, row, ITERATION, config):
    
    """This function finds the surface/reference signal for a flaw signal at "ax" & "circ"
        The surface with features closer to the flaw signal is selected.
        
    Args:
        nb_data_arr(np array): B-scan array
        flaw_dic_ax(dic): FLaw location in each frame
        flattened_lags(array): Flattened lags
        ax(int): axial position of flaw a-scan
        circ(int): circ position of flaw a-scan
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        ITERATION(int): Iteration number
        config(dic): Config dictionary

    Returns:
        surface_a_scan(np.array): Selected surface a-scan
        surface_wave_locations(np.array): info about the surface a-scan fwg
        surface_circ: Circ loc of the selected surface a-scan
       
    """
    scan_circ_end = nb_data_arr.shape[1] - 1
    
    surface_a_scan_list = []
    surface_wave_loc_list = []
    surface_a_scan_fwg_list = []
    cost_list = []
    surface_circ_list = []
    
    # no of circ location outside the flaw extent to be considered while selecting surface a-scan
    range_circ = config['SURFACE_CIRC_RANGE']
    
    # circ start and end of the flaw extent
    pred_circ_start = row['Ro Start']
    pred_circ_end = row['Ro End']
    pred_circ_start-=config['SURFACE_CIRC_BUFFER']
    pred_circ_end+=config['SURFACE_CIRC_BUFFER']
    if config['FLAW_TYPE'] == 'FBBPF':
        lag_difference_thresh = config['SURFACE_LAG_DIFFERENCE_CRITERIA']
        wg_width = config['WAVEGROUP_WIDTH']
    if config['FLAW_TYPE'] == 'DEBRIS':
        lag_difference_thresh = None
        wg_width = config[f'WAVEGROUP_WIDTH_{ITERATION}']
    
    # find flaw a-scan FWG
    flaw_a_scan = nb_data_arr[ax, circ, :]
    flaw_wave_locations = find_wavegroups_in_frame(flaw_a_scan,
                                              number_of_wavegroups = 1,
                                              wavegroup_width = wg_width)
                                              
    flaw_a_scan_fwg = flaw_a_scan[flaw_wave_locations[0][0] : flaw_wave_locations[0][1]]
    
    # find all flaw circ locations present on ax loc
    try:
        flaw_loc_scan = flaw_dic_ax[row['Filename']]
        flaw_loc_ax = flaw_loc_scan[ax]
        
        flaw_loc_ax = [item for sublist in flaw_loc_ax for item in sublist]
        
    except:
        flaw_loc_ax = []

    # Go through n_range_circ a-scans outside of the flaw extent (both sides)
    # select the surface a-scan with features similar to flaw a-scan
    n_up = 0
    n_bottom = 0
    while len(surface_a_scan_list) < 2*range_circ:
        circ_up = pred_circ_start - n_up - 1
        circ_bottom = pred_circ_end + n_bottom + 1
    
        # make sure circ_bottom is not greater than scan_circ_end
        if circ_bottom > scan_circ_end:
            circ_bottom = scan_circ_end
        
        for circ in [circ_up, circ_bottom]:
            
            if config['FLAW_TYPE'] == 'FBBPF':
                circ_selection_condn = circ not in flaw_loc_ax and flattened_lags[circ]<=lag_difference_thresh
            if config['FLAW_TYPE'] == 'DEBRIS':
                circ_selection_condn = circ not in flaw_loc_ax
            if circ_selection_condn:
                surface_a_scan = nb_data_arr[ax, circ, :]
                
                wave_locations = find_wavegroups_in_frame(surface_a_scan,
                                                          number_of_wavegroups = 1,
                                                          wavegroup_width = wg_width)
                
                surface_a_scan_fwg = surface_a_scan[wave_locations[0][0] : wave_locations[0][1]]
                
                
                
                surface_a_scan_list.append(surface_a_scan)
                surface_a_scan_fwg_list.append(surface_a_scan_fwg)
                surface_wave_loc_list.append(wave_locations)
                _, cost = dtw_path(surface_a_scan_fwg, flaw_a_scan_fwg)
                cost_list.append(cost)
                surface_circ_list.append(circ)
        n_up+= 1
        n_bottom+= 1
        # if no of locations searched > than the scan dimensions
        if n_up >= scan_circ_end or n_bottom >= scan_circ_end:
            print('Surface not found')
            raise ValueError('Surface not found')

    # select the surface a-scan with features similar to flaw a-scan: min cost
    surface_peak_posn_idx = np.argmin(cost_list) # find surface with shape/feature similar to flaw
    surface_a_scan = surface_a_scan_list[surface_peak_posn_idx]
    surface_wave_locations = surface_wave_loc_list[surface_peak_posn_idx]
    surface_circ = surface_circ_list[surface_peak_posn_idx]

    return surface_a_scan, surface_wave_locations, surface_circ

def find_distance(a, b, normalize):
    
    """This function finds the distance between two vectors

    Args:
        a(np.array): First vector
        b(np.array): Second vector
        normalize(bool): if True, calculates the normalized distance

    Returns:
        cost(float): DTW cost
       
    """
    
    if normalize:
        a = a / np.linalg.norm(a)
        b = b / np.linalg.norm(b)
        
    return np.linalg.norm(a - b)



def find_cost_dtw(flaw_a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_a_scan_idxs, flaw_a_scan_idxs, normalize, focus):
    
    """This function finds the DTW cost

    Args:
        flaw_a_scan_fwg(np.array): Fwg of flaw a-scan
        surface_a_scan_fwg(np.array): Fwg of surface a-scan
        surface_peak_posn(int): Surface feature position
        surface_a_scan_idxs(np.array): surface a-scan indices matched with flaw a-scan
        flaw_a_scan_idxs(np.array): flaw a-scan indices matched with surface a-scan
        cost(float): DTW cost
        normalize(bool): if True, Normalize cost
        focus(bool): if True, focus on the area near to selected flaw and surface feature

    Returns:
        cost(float): DTW cost
       
    """
    
    if not focus:
        cost = find_distance(surface_a_scan_fwg[surface_a_scan_idxs], flaw_a_scan_fwg[flaw_a_scan_idxs], normalize)
        
    if focus:
        
        surface_a_scan_peak_idxs = surface_a_scan_idxs[(surface_a_scan_idxs >= surface_peak_posn) & (surface_a_scan_idxs <= surface_peak_posn)]
        
        flaw_a_scan_peak_idxs = flaw_a_scan_idxs[(surface_a_scan_idxs >= surface_peak_posn) & (surface_a_scan_idxs <= surface_peak_posn)]
        
        cost = find_distance(surface_a_scan_fwg[surface_a_scan_peak_idxs], flaw_a_scan_fwg[flaw_a_scan_peak_idxs], normalize)
        
    return cost


def find_lag(flaw_a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_fwg_start, surface_fwg_stop, flaw_fwg_start, flaw_fwg_stop):
    
    """This function selects the flaw feature and calculates the lag 

    Args:
        flaw_a_scan_fwg(np.array): Fwg of flaw a-scan
        surface_a_scan_fwg(np.array): Fwg of surface a-scan
        surface_peak_posn(int): Surface feature position
        surface_fwg_start(int): Fwg start position of surface a-scan
        surface_fwg_stop(int): Fwg stop position of surface a-scan
        flaw_fwg_start(int): Fwg start position of flaw a-scan
        flaw_fwg_stop(int): Fwg stop position of flaw a-scan

    Returns:
        surface_peak_posn(int): Selected surface feature position
        surface_peak_value(int): Selected surface feature amp value
        flaw_peak_posn(int): Selected flaw feature position
        flaw_peak_value(int): Selected flaw feature amp value
        lag(int): Lag between surface and flaw feature
        path(array): DTW indices matching path: used for plotting
        cost_normalized(float): Dtw cost between overall flaw and surface signal
        cost_focus_normalized(float): Dtw cost between area around selected flaw and surface feature
       
    """
    
    # x-axis for flaw and surface
    x_surface = np.arange(surface_fwg_start, surface_fwg_stop)
    x_flaw = np.arange(flaw_fwg_start, flaw_fwg_stop)
    
    # distance between start of flaw fwg and stop of surface fwg
    dist_fwgs =  flaw_fwg_start - surface_fwg_stop
    
    # surface_peak_posn w.r.t fwg
    surface_peak_posn = surface_peak_posn - surface_fwg_start
    
    # use DTW to match the features across surface and flaw and find the flaw feature
    y1 = surface_a_scan_fwg
    y2 = flaw_a_scan_fwg
    
    path, cost = dtw_path(y1, y2)
    path = np.array(path)
    
    surface_a_scan_idxs = path[:, 0]
    flaw_a_scan_idxs = path[:, 1]
    
    # calculate DTW cost
    cost_normalized = find_cost_dtw(flaw_a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_a_scan_idxs, flaw_a_scan_idxs, False, False)
    # cost calculated useng the nearby (3) points to surface peak positions
    cost_focus_normalized = find_cost_dtw(flaw_a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, surface_a_scan_idxs, flaw_a_scan_idxs, True, True)
    
    # find surface peak in "surface_a_scan_idxs"
    surface_peak_match_idx = np.where(surface_a_scan_idxs == surface_peak_posn)[0]
    
    # convert the indexes w.r.t start
    surface_peak_posns = x_surface[surface_a_scan_idxs[surface_peak_match_idx]]
    # find surface peak in "a_scan_idxs"
    flaw_peak_posns = x_flaw[flaw_a_scan_idxs[surface_peak_match_idx]]
    flaw_peak_values = y2[flaw_a_scan_idxs[surface_peak_match_idx]]
    
    # find the corresponding difference between positions
    diff = (flaw_peak_posns - flaw_fwg_start) + dist_fwgs + (surface_fwg_stop - surface_peak_posns)
    lag_idx = np.argmax(flaw_peak_values)
    
    # find all w.r.t max diff
    lag = diff[lag_idx]
    flaw_peak_posn = flaw_peak_posns[lag_idx]
    surface_peak_posn = surface_peak_posns[lag_idx]
    flaw_peak_value = flaw_peak_values[lag_idx]
    surface_peak_value = y1[surface_a_scan_idxs[surface_peak_match_idx]][lag_idx]

    return surface_peak_posn, surface_peak_value, flaw_peak_posn, flaw_peak_value, lag, path, cost_normalized, cost_focus_normalized

def get_flattened_lags(ax, lags, backup_lags, flaw_dic_ax, row, config):
    
    """This function calculates the flattened lag

    Args:
        ax(int): Axial location starts from 0
        lags(array): The lags from performing cross-correlations on the entire A-scan
        backup_lags(array): The lags from performing cross-correlations on the focus wave group
        flaw_dic_ax(dictionary): contains circ locations of all flaws inside a frame
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
    Returns:
        pressure_tube_location: 
        flattened_lags(array): Lags after flattening
    """
    
    if np.sum(np.gradient(lags[ax])) <= np.sum(np.gradient(backup_lags[ax])):
        used_lags = lags[ax]
    else:
        used_lags = backup_lags[ax]
   
    if config['FLATTENNING_METHOD']=='FBBPF':
        indications = np.zeros(len(used_lags),dtype=bool)
        locations = flaw_dic_ax[row['Filename']][ax]
        for i in locations:
            indications[i] = True
        pressure_tube_location = Smoothlags_remove_flaws(used_lags, config['SD_LIMIT'], config['MAXITER'],indications, config['SG_ORD'], config['SG_FLEN'], config['OUTLIER_GRADIENT'])[0]
    else:
        pressure_tube_location = get_pressure_tube_surface(used_lags, row['Ax Start'], row['Ax End'], config['SG_FLEN'], config['SG_ORD'])
    flattened_lags = np.abs(used_lags - pressure_tube_location)
    
    return pressure_tube_location, flattened_lags

def select_pred_depth(pred_depth_whole_bscan_all_iter_probe, row, index, save_location):
    
    OUTAGE_NUMBER = row['Outage Number']
    CHANNEL = row['Channel']
    FILENAME = row['Filename'].split('.')[0]
    
    # selected probe and iteration in post processor
    iteration, probe = row['ITERATION'], row['PROBE']
    
    is_depth_fwg_rmv = False # is depth caluclated using reflections or removing fwg
    is_depth_inverted = False # is depth caluclated using inverted signal
    [pred_depth_whole_bscan_all, pred_depth_whole_bscan_fwg_rmv_all, pred_depth_whole_bscan_invert_all, pred_depth_fwg_rmv_whole_bscan_invert_all] = pred_depth_whole_bscan_all_iter_probe[probe + str(iteration)]
    
    # check depth calculated using inverted or fwg was removed
    if row['pred_depth_nb1_nb2'] == row['pred_depth_fwg_rmv'] or row['pred_depth_nb1_nb2'] == row['pred_depth_fwg_rmv_invert'] or row['pred_depth_nb1_nb2'] == row['pred_depth_first_peak']:
        is_depth_fwg_rmv = True
    if row['pred_depth_nb1_nb2'] == row['pred_depth_invert'] or row['pred_depth_nb1_nb2'] == row['pred_depth_fwg_rmv_invert']:
        is_depth_inverted = True
    # print(is_depth_fwg_rmv, is_depth_inverted)
    # select correct pred depth array
    if not is_depth_fwg_rmv and not is_depth_inverted:
        pred_depth_arr = pred_depth_whole_bscan_all[index]
        flaw_ax, flaw_circ = row['flaw_ax'], row['flaw_circ']
        flaw_peak = row['flaw_feature_amp']
    if is_depth_fwg_rmv and not is_depth_inverted:
        flaw_ax, flaw_circ = row['flaw_fwg_rmv_ax'], row['flaw_fwg_rmv_circ']
        flaw_peak = row['flaw_feature_amp_fwg_rmv']
        pred_depth_arr = pred_depth_whole_bscan_fwg_rmv_all[index]
    if not is_depth_fwg_rmv and is_depth_inverted:
        flaw_ax, flaw_circ = row['flaw_ax_invert'], row['flaw_circ_invert']
        flaw_peak = row['flaw_feature_amp_invert']
        pred_depth_arr = pred_depth_whole_bscan_invert_all[index]
    if is_depth_fwg_rmv and is_depth_inverted:
        flaw_ax, flaw_circ = row['flaw_fwg_rmv_ax_invert'], row['flaw_fwg_rmv_circ_invert']
        flaw_peak = row['flaw_feature_amp_fwg_rmv_invert']
        pred_depth_arr = pred_depth_fwg_rmv_whole_bscan_invert_all[index]
        
    assert pred_depth_arr.shape == (row['Ax End'] - row['Ax Start'], row['Ro End'] - row['Ro Start'])
    # save predicted depth for whole b-scan
    save_path = os.path.join(save_location, 'Depth Sizing', 'Pred depth whole b-scan', str(row['Indication']))
    os.makedirs(os.path.join(save_path), exist_ok=True)
    pd.DataFrame(pred_depth_whole_bscan_all[index]).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_whole_bscan.xlsx'))
    pd.DataFrame(pred_depth_whole_bscan_fwg_rmv_all[index]).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_whole_bscan_fwg_rmv.xlsx'))
    pd.DataFrame(pred_depth_whole_bscan_invert_all[index]).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_whole_bscan_invert.xlsx'))
    pd.DataFrame(pred_depth_fwg_rmv_whole_bscan_invert_all[index]).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_fwg_rmv_whole_bscan_invert.xlsx'))

        
    return pred_depth_arr, flaw_ax, flaw_circ, flaw_peak, is_depth_fwg_rmv, is_depth_inverted