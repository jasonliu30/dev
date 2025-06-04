"""This script finds the depth and other flaw-info, such as flaw bottom
     location, max amp, etc., for Debris flaws"""

import itertools
import os
import numpy as np
import pandas as pd
from numpy import unravel_index
from depth_sizing.utils.flattening import get_pressure_tube_surface
from depth_sizing.utils.debris_post_processor import debris_post_processor
from depth_sizing.utils.depth_sizing_flag import flag_cases_debris
from depth_sizing.utils.chatter_magnitude import measure_chatter
from depth_sizing.utils.reflections_utils import rmv_fwg,\
    check_multi_reflections, find_consecutive_true, check_reflections_main_fwg,\
    ignore_reflections_open_surface, open_surface_middle_depth
from depth_sizing.utils.plot_utils import plot_depth_profile_single_flaw
from depth_sizing.utils.utils import flaw_loc_file, remove_abrupt_change, find_lag,\
    find_wavegroups_in_frame, find_surface, units_depth,\
    find_signal_invert, find_surface_feature, select_pred_depth
from utils.logger_init import create_dual_loggers

# create loggers
_, file_logger  = create_dual_loggers()

def update_flaw_feature_amp_inverted(stats_df, results_df, b_scans):
    
    """This function update the flaw feature amp in case of inverted signal is used
    Args:
        stats_df(dataframe): statistics of depth calculation
        results_df(dataframe): results of depth calculation with columns:
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'pred_depth_nb1_nb2' (predicted depth in mm)
            -'flaw_ax' (frame location of deepest location,starts with 1)
            -'flaw_circ(circ location of deepest location,starts with 0)
            -'surface_ax' (surface frame location,starts with 1)
            -'surface_circ(surface circ location,starts with 0)
            -'Flaw Maximum Amplitude' (the maximum amplitude of the peaks used to measure the flaw)
        b_scans(list): List of NB1 and NB2 b-scans
        
    Returns:
        results_df(dataframe): results with changed flaw feature amp value
    """
    # loop through all instances
    for i, row in stats_df.iterrows():
        try:
            # update the flaw feature amp if the depth is calculated using inverted signal
            if row['is_depth_inverted']:
                flaw_ax = int(results_df.loc[i, 'flaw_ax'])
                flaw_circ = int(results_df.loc[i,'flaw_circ'])
                nb_data_arr = b_scans[0] if row['PROBE'] == 'NB1' else b_scans[1]
                flaw_feature_amp_inverted = results_df.loc[i, 'flaw_feature_amp']
                flaw_feature_amp = 2*np.median(nb_data_arr[flaw_ax, flaw_circ]) - flaw_feature_amp_inverted
                results_df.loc[i, 'flaw_feature_amp'] = max(0, min(flaw_feature_amp, 255))
        except:
            pass

    return results_df

def measure_chatter_all(df, lags_all, backup_lags_all):
    """This function measure the chatter for all the instances in df
    Args:
        df(dataframe): dataframe of every flaw to size the depth of, must all be
        within the same B-scan file. required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        lags_all(list) : List of the NB1 and NB2 lags from performing cross-correlations on the entire A-scan
        backup_lags_all(list) : List of the NB1 and NB2 lags from performing cross-correlations on the g4 of A-scan
        
    Returns:
        chatter_all(list): measure chatter for nb1 and nb2 probes
           
    """
    
    chatter_nb1 = []
    chatter_nb2 = []
    chatter_all = [chatter_nb1, chatter_nb2]

    # loop through every flaw
    for i, (lags, backup_lags) in enumerate(zip(lags_all, backup_lags_all)):
        for index, row in df.iterrows():
            try:
                chatter = measure_chatter(FILENAME_FULL,lags,backup_lags, flaw_dic_ax,
                                            row['Ax Start'], row['Ax End'],
                                            row['Ro Start'], row['Ro End'],config, FLAW_TYPE)
                chatter_all[i].append(chatter)
            except:
                file_logger.error("Chatter amp cannot be calculated")
                chatter_all[i].append('Chatter amp cannot be calculated')
    return chatter_all

def plot_depth_profile_all_flaws(stats_df, pred_depth_whole_bscan_all_iter_probe,save_location):

    """This function plots depth profile for all flaws
    Args:
        stats_df(dataframe): Dataframe with statistics and meta data of the flaw
        pred_depth_whole_bscan_all_iter_probe(dic): pred depth for all possible combos of iteration and probe
        save_location(str): Location to save the plots
    Returns:
        is_depth_fwg_rmv_all(list): True, if the fwg is removed to calc the depth
        is_depth_inverted_all(list): True, if the signal is inverted to calc the depth
        flaw_ax_all(list): Flaw axial location
        flaw_circ_all(list): Flaw circ location
        flaw_peak_all(list): Flaw peak amp    
    """

    ####### plot depth profile based on selected iteration and probe ##############
    is_depth_fwg_rmv_all = []
    is_depth_inverted_all = []
    flaw_ax_all = []
    flaw_circ_all = []
    flaw_peak_all = []

    for index, (i, row) in enumerate(stats_df.iterrows()):
        pred_depth_arr, flaw_ax, flaw_circ, flaw_peak, is_depth_fwg_rmv,\
            is_depth_inverted = select_pred_depth(pred_depth_whole_bscan_all_iter_probe,
                                                  row, index, save_location)
        
        plot_depth_profile_single_flaw(row, pred_depth_arr, 'normal_beam', SAVE_ROOT)
        
        is_depth_fwg_rmv_all.append(is_depth_fwg_rmv)
        is_depth_inverted_all.append(is_depth_inverted)
        flaw_ax_all.append(flaw_ax)
        flaw_circ_all.append(flaw_circ)
        flaw_peak_all.append(flaw_peak)
        
    return is_depth_fwg_rmv_all, is_depth_inverted_all, flaw_ax_all, flaw_circ_all, flaw_peak_all


def find_depth_single_circ(
        nb_data_arr, row, ax, circ, reflection_std_amp_arr, scan_stats,
        pressure_tube_location, peak_posn_diff_thrsh, PLOTTING,save_location):

    """This function calculates depth for a single circ loc (circ) inside an ax loc (ax)

    Args:
        nb_data_arr: B-scan array
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        ax(int): Axial location starts from 0
        circ(int): Circ location starts from 0
        reflection_std_amp_arr(list): list of arrays, contains reflections stats: std, amp, amp ratio, amp diff & strong reflections
        scan_stats: Scan stats such as std and amp
        pressure_tube_location: 
        peak_posn_diff_thrsh(int): Min position difference between surface and flaw feature/peak, report np.nan when not satisfied
        PLOTTING(bool): To plot intermediate plots
        save_location(str): Location to save the plots
            
    Returns:
        surface_circ_loc(int): Circumferential location of surface
        flaw_peak_value(int): Flaw feature/peak value
        depth_circ(float): predicted depth from one circ loc
        cost_normalized(float): DTW normalized cost
        cost_focus_normalized(float): DTW normalized cost focusing near to flaw and surface feature
        scan_stats(dic): Reflection stats: amp and std
        reflections_fwg(bool): True, if reflections are present based on the fwg
        open_surface(bool): True, if surface is open to the surface
        n_multi_reflections(int): no of multiple feature inside the reflections
       
    """

    frame = ax+1
    curvature_correction = 0
    n_multi_reflections = 0
    
    [reflection_std_arr, # change in std dev across the reflection window, reflection window is the area around the max peak after removing main FWG
     reflection_amp_arr, # max amp in the reflection window
     reflection_amp_ratio_arr, # amp ratio (flaw amp / surface amp) in the reflection window
     reflection_amp_diff_arr, # amp ratio (flaw amp / surface amp) in the reflection window
     reflection_strong_arr] = reflection_std_amp_arr

    # file_logger.debug(f"Calculating depth for a single circ loc ({circ}) inside a frame ({frame})")

    # find surface signal
    file_logger.debug("Selecting surface signal...")
    surface_a_scan, surface_wave_locations, surface_circ_loc = find_surface(nb_data_arr,
                                                                            flaw_dic_ax, None,
                                                                            ax, circ, row,
                                                                            ITERATION,
                                                                            config)
    
    # flaw a-scan at ax, circ
    flaw_a_scan = nb_data_arr[ax, circ, :]
    
    # find flaw a-scan main fwg
    flaw_wave_locations = find_wavegroups_in_frame(flaw_a_scan,
                                                   REFLECTION_AMP_THRSH,
                                                  number_of_wavegroups = 1,
                                                  wavegroup_width = config[f'WAVEGROUP_WIDTH_{ITERATION}'],
                                                  amplitude_threshold = config['FLAW_AMP_THRSH'],
                                                  multi_reflection = multi_reflection)
    
    # if no main flaw fwg found
    if len(flaw_wave_locations) == 0:
        # print('No primary FWG found due to flaw feature amp less than the threshold')
        
        flaw_peak_value = np.nan
        depth_circ = np.nan
        cost_normalized = np.nan
        flaw_fwg_rmv_peak_value = np.nan
        depth_circ_rmv_fwg = np.nan
        cost_focus_normalized = np.nan
        reflections_fwg = np.nan
        open_surface = np.nan
        surface_circ_loc = np.nan
        
        return surface_circ_loc, [flaw_peak_value, flaw_fwg_rmv_peak_value], [depth_circ, depth_circ_rmv_fwg], cost_normalized,\
         cost_focus_normalized, scan_stats, reflections_fwg, open_surface, n_multi_reflections
    
    # start and stop position of fwg
    surface_fwg_start, surface_fwg_stop = surface_wave_locations[0][0], surface_wave_locations[0][1]
    flaw_fwg_start, flaw_fwg_stop = flaw_wave_locations[0][0], flaw_wave_locations[0][1]
    
    # remove the main fwg to find the reflections
    surface_a_scan_fwg_rmv, surface_a_scan_fwg = rmv_fwg(surface_a_scan, surface_fwg_start, surface_fwg_stop)
    flaw_a_scan_fwg_rmv, flaw_a_scan_fwg = rmv_fwg(flaw_a_scan, flaw_fwg_start, flaw_fwg_stop)
    
    # surface feature selection
    # finds left most peak with amp > amp thresh inside the fwg
    file_logger.debug("Selecting surface feature...")
    (surface_peak_posn, surface_a_scan_all_peak, surface_a_scan_cand_peak, surface_a_scan_select_peak),\
        surface_not_found_reason, surface_found = find_surface_feature(surface_a_scan_fwg,
                                                                       surface_fwg_start,
                                                                       peak_type = 'max',
                                                                       config = config)
    
    if surface_found:
        # select flaw peak and find the lag between surface and flaw peak/feature
        file_logger.debug("Selecting flaw feature...")
        surface_peak_posn, surface_peak_value, flaw_peak_posn, flaw_peak_value,\
            lag, path, cost_normalized, cost_focus_normalized = find_lag(flaw_a_scan_fwg,
                                                                         surface_a_scan_fwg,
                                                                         surface_peak_posn,
                                                                         surface_fwg_start,
                                                                         surface_fwg_stop,
                                                                         flaw_fwg_start,
                                                                         flaw_fwg_stop)
    else:
        #if no surface found return nans
        file_logger.debug("No surface found")
        flaw_peak_value = np.nan
        depth_circ = np.nan
        cost_normalized = np.nan
        flaw_fwg_rmv_peak_value = np.nan
        depth_circ_rmv_fwg = np.nan
        cost_focus_normalized = np.nan
        reflections_fwg = np.nan
        open_surface = np.nan
        surface_circ_loc = np.nan
        
        return surface_circ_loc, [flaw_peak_value, flaw_fwg_rmv_peak_value], [depth_circ, depth_circ_rmv_fwg],\
              cost_normalized, cost_focus_normalized, scan_stats, reflections_fwg, open_surface, n_multi_reflections
    
    # curvature correction
    if config['FLATTENNING']:
        curvature_correction = pressure_tube_location[surface_circ_loc] - pressure_tube_location[circ]
        
    # positional difference between flaw and surface feature/peak
    posn_diff = flaw_peak_posn - surface_peak_posn
    
    # if the positonal difference between the primary FWG < peak_posn_diff_thrsh, pred depth as nan
    if  posn_diff < peak_posn_diff_thrsh:
        depth_circ = np.nan
        flaw_peak_value = np.nan

    else:
        depth_circ = units_depth(surface_peak_posn, flaw_peak_posn, curvature_correction, config)
     
    # check for the multiple features in the reflection window
    multi_reflections_idx = check_multi_reflections(flaw_a_scan_fwg_rmv, amp_thresh = 140)
    
    # report multi reflections if there are at least two features        
    if len(multi_reflections_idx) >= 2:
        # print(f'Multi reflections: {len(multi_reflections_idx)}')
        n_multi_reflections = len(multi_reflections_idx)
        
    
    # find flaw FWG for reflections
    flaw_fwg_rmv_wave_locations = find_wavegroups_in_frame(flaw_a_scan_fwg_rmv,
                                                           REFLECTION_AMP_THRSH,
                                                          number_of_wavegroups = 1,
                                                          wavegroup_width = config[f'WAVEGROUP_WIDTH_{ITERATION}'],
                                                          n_multi_reflections=n_multi_reflections,
                                                          multi_reflection = multi_reflection)
    
    (flaw_fwg_rmv_start, flaw_fwg_rmv_stop, flaw_fwg_rmv_peak_posn, flaw_fwg_rmv_peak_value, _, _) = flaw_fwg_rmv_wave_locations[0]
    
    # slice the reflection window from the flaw_a_scan_fwg_rmv
    flaw_a_scan_fwg_2 = flaw_a_scan_fwg_rmv[flaw_fwg_rmv_start : flaw_fwg_rmv_stop]
    # slice the surface a scan from the same location
    surface_a_scan_roi = surface_a_scan_fwg_rmv[flaw_fwg_rmv_start : flaw_fwg_rmv_stop]
    
    # compare reflection window with surface a scan roi to see any increase in std/variance, this is used to confirm the reflections
    ref_std = (np.std(flaw_a_scan_fwg_2) - np.std(surface_a_scan_roi)) / np.std(surface_a_scan_roi)
    
    # check reflections based on the fwg
    reflections_fwg, open_surface, fwg_amp_diff = check_reflections_main_fwg(flaw_a_scan,
                                                                             surface_a_scan,
                                                                             surface_fwg_start,
                                                                             surface_fwg_stop,
                                                                             ref_std, config,
                                                                             ITERATION,
                                                                             OPEN_SURFACE_AMP_DIFF_THRSH)
        
        
    # if circ loc is open to surface, position diff bet flaw and surface feature > thresh,
    #  later the reflections will be ignored for the open surface = True
    if  posn_diff >= peak_posn_diff_thrsh:
        open_surface = True
    
    # ignore the too far away reflections
    if flaw_fwg_rmv_peak_posn - surface_peak_posn < config['MAX_PEAK_DIFF']:
        
        # reflections stats, calculated from the reflection window
        ref_std = (np.std(flaw_a_scan_fwg_2) - np.std(surface_a_scan_roi)) / np.std(surface_a_scan_roi)
        ref_max_amp = np.nanmax(flaw_a_scan_fwg_2)
        ref_amp_diff = (np.nanmax(flaw_a_scan_fwg_2) - np.nanmax(surface_a_scan_roi))
        ref_amp_ratio = np.nanmax(flaw_a_scan_fwg_2 / surface_a_scan_roi)
        scan_stats['reflections_std'].append(ref_std)
        scan_stats['reflections_max_amp'].append(ref_max_amp)
        
        # print("Reflection stats std, amp, amp diff, reflection_fwg: ", ref_std, ref_max_amp, ref_amp_diff, reflections_fwg)

        # check for strong reflections
        if ref_std >= config['REFLECTION_STD_PERCENT_THRSH_STRONG'] and ref_max_amp >= config['REFLECTION_AMP_THRSH_STRONG']:
            reflection_strong_arr[ax - row['Ax Start'], circ - row['Ro Start']] = True
        
        # reflections stats
        reflection_std_arr[ax - row['Ax Start'], circ - row['Ro Start']] = ref_std
        reflection_amp_arr[ax - row['Ax Start'], circ - row['Ro Start']] = flaw_fwg_rmv_peak_value
        reflection_amp_ratio_arr[ax - row['Ax Start'], circ - row['Ro Start']] = flaw_fwg_rmv_peak_value / surface_peak_value
        reflection_amp_diff_arr[ax - row['Ax Start'], circ - row['Ro Start']] = flaw_fwg_rmv_peak_value - surface_a_scan_roi[flaw_fwg_rmv_peak_posn - flaw_fwg_rmv_start]

        # Check for reflections based on two conditions: Amp and reflection flag from fwg
        reflections_condn_amp = ref_max_amp >= REFLECTION_AMP_THRSH
        # print(reflections_condn_amp, REFLECTION_AMP_THRSH)
        if reflections_fwg and reflections_condn_amp:
            
            # select flaw feature and find lag bet surface and flaw feature using reflection window
            file_logger.debug("Selecting surface & flaw feature after removing the FWG...")
            surface_peak_posn, surface_peak_value,\
                flaw_fwg_rmv_peak_posn, flaw_fwg_rmv_peak_value,\
                flaw_fwg_rmv_lag, flaw_fwg_rmv_path,\
                flaw_fwg_rmv_cost_normalized,\
                flaw_fwg_rmv_cost_focus_normalized = find_lag(flaw_a_scan_fwg_2,
                                                              surface_a_scan_fwg,
                                                              surface_peak_posn,
                                                              surface_fwg_start,
                                                              surface_fwg_stop,
                                                              flaw_fwg_rmv_start,
                                                              flaw_fwg_rmv_stop)
                
            depth_circ_rmv_fwg = units_depth(surface_peak_posn,
                                             flaw_fwg_rmv_peak_posn,
                                             curvature_correction,
                                             config)
            
            # print("Depth from reflection: ", depth_circ_rmv_fwg, surface_peak_posn, flaw_fwg_rmv_peak_posn, curvature_correction)
            # min reportable depth from reflections
            if depth_circ_rmv_fwg >= (peak_posn_diff_thrsh - 2) * config['UNIT_MICRO_SEC'] * config['MICRO_SEC_depth_NB']:
                # print("Depth from reflection: ", depth_circ_rmv_fwg)
                save_path = os.path.join(save_location, 'Depth Sizing', row['Indication'])
            else:
                depth_circ_rmv_fwg = np.nan
                flaw_fwg_rmv_peak_value = np.nan
        else:
            depth_circ_rmv_fwg = np.nan
            flaw_fwg_rmv_peak_value = np.nan
      
    else:
        depth_circ_rmv_fwg = np.nan
        flaw_fwg_rmv_peak_value = np.nan
        ref_std = np.nan
        ref_max_amp = np.nan
        ref_amp_diff = np.nan
        ref_amp_ratio = np.nan
        
        scan_stats['reflections_std'].append(ref_std)
        scan_stats['reflections_max_amp'].append(ref_max_amp)
        
    # update reflections stats
    reflection_std_arr[ax - row['Ax Start'], circ - row['Ro Start']] = ref_std
    reflection_amp_arr[ax - row['Ax Start'], circ - row['Ro Start']] = ref_max_amp
    reflection_amp_ratio_arr[ax - row['Ax Start'], circ - row['Ro Start']] = ref_amp_ratio
    reflection_amp_diff_arr[ax - row['Ax Start'], circ - row['Ro Start']] = ref_amp_diff
    
    flaw_peak_value = [flaw_peak_value, flaw_fwg_rmv_peak_value]
    depth_circ = [depth_circ, depth_circ_rmv_fwg]
    
    return surface_circ_loc, flaw_peak_value, depth_circ, cost_normalized, cost_focus_normalized,\
          scan_stats, reflections_fwg, open_surface, n_multi_reflections
    

def find_depth_single_frame(
        nb_data_arr, row, ax, reflection_std_amp_arr,
        NB_lags, peak_posn_diff_thrsh, PLOTTING,save_location):

    """This function calculates depth for a single frame (ax) or for all circ locs inside a single frame

    Args:
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
            
        ax(int): Axial location starts from 0
        reflection_std_amp_arr(list): list of arrays, contains reflections stats: std, amp, amp ratio, amp diff & strong reflections
        NB_lags(np.ndarray) : The lags from performing cross-correlations on the entire A-scan
        peak_posn_diff_thrsh(int): Min position difference between surface and flaw feature/peak, report np.nan when not satisfied
        PLOTTING(bool): To plot intermediate plots
        save_location(str): Location to save the plots
            
    Returns:
        depth_ax(list): list of arrays, contains the predicted depth from main fwg and after removing main fwg for a single frame/ax and all circ locs 
        cost_normalized_list(list): contains the dtw cost for a single frame/ax and all circ locs 
        scan_stats(dic): Reflection stats: amp and std
        n_circ_reflections(int): no of circ locs with reflections inside a frame
        reflections_fwg_frame_list (list): True, if circ loc has reflections, shape: (1 * no of circ locs) 
        open_surface_frame_list(list): True, if circ loc is open to surface, shape: (1 * no of circ locs)
        n_multi_reflections_list(list): no of multiple features/reflections present at a circ loc, shape: (1 * no of circ locs)
        flaw_peak_value_list: List of flaw feature values in a frame
        surface_circ_loc_ax_list (list): Surface circ location for all frames
       
    """
    
    n_circ_reflections = 0 # no of circ locs with reflections inside a frame
    reflections_fwg_frame_list = [] # True, if circ loc has reflections, shape: (1 * no of circ locs)
    open_surface_frame_list = [] # True, if circ loc is open to surface, shape: (1 * no of circ locs)
    n_multi_reflections_list = []# no of multiple features/reflections present at a circ loc, shape: (1 * no of circ locs)
    
    # smooth the frame of lags to get the general trend
    if config['FLATTENNING']:
        lags=NB_lags[ax]
        pressure_tube_location = get_pressure_tube_surface(lags,row['Ro Start'],
                                                           row['Ro End'],config['SG_FLEN'],
                                                           config['SG_ORD'])
    
    
    depth_ax_list = [] # stores depth for every circ loc in specific frame
    depth_ax_fwg_rmv_list = [] # stores depth after removing fwg for every circ loc in specific frame
    flaw_peak_value_list = [] # stores flaw peak values for every circ loc in specific frame
    flaw_fwg_rmv_peak_value_list = [] # stores flaw peak values after removing fwg for every circ loc in specific frame
    cost_normalized_list = [] # stores dtw cost for every circ loc in specific frame
    cost_focus_normalized_list = []
    surface_circ_loc_ax_list = [] # stores surface circ location for every circ loc in specific frame
    
    scan_stats = {
                'reflections_std' : [],
                'reflections_max_amp' : []
        }

    # for every circ location
    for circ in range(row['Ro Start'], row['Ro End']):
        # calculate depth for a single circ loc        
        surface_circ_loc, [flaw_peak_value, flaw_fwg_rmv_peak_value],\
        [depth_circ, depth_circ_rmv_fwg], cost_normalized,\
        cost_focus_normalized, scan_stats, reflections_fwg_circ,\
        open_surface_circ, n_multi_reflections  = find_depth_single_circ(nb_data_arr,
                                                                         row, ax, circ,
                                                                         reflection_std_amp_arr,
                                                                         scan_stats,
                                                                        pressure_tube_location,
                                                                        peak_posn_diff_thrsh,
                                                                        PLOTTING,
                                                                        save_location)
        n_multi_reflections_list.append(n_multi_reflections)
        reflections_fwg_frame_list.append(reflections_fwg_circ)
        open_surface_frame_list.append(open_surface_circ)
        
        if not np.isnan(depth_circ_rmv_fwg):
            n_circ_reflections+=1
            
        flaw_peak_value_list.append(flaw_peak_value)
        flaw_fwg_rmv_peak_value_list.append(flaw_fwg_rmv_peak_value)

        depth_ax_list.append(depth_circ)
        depth_ax_fwg_rmv_list.append(depth_circ_rmv_fwg)
        
        cost_normalized_list.append(cost_normalized)
        cost_focus_normalized_list.append(cost_focus_normalized)

        surface_circ_loc_ax_list.append(surface_circ_loc)
        
        depth_ax = [depth_ax_list, depth_ax_fwg_rmv_list]
        flaw_peak = [flaw_peak_value_list, flaw_fwg_rmv_peak_value_list]

    return depth_ax, cost_normalized_list, scan_stats, n_circ_reflections,\
          reflections_fwg_frame_list, open_surface_frame_list, n_multi_reflections_list, flaw_peak, surface_circ_loc_ax_list

def find_depth_single_flaw(
        row, NB_lags, nb_data_arr, peak_posn_diff_thrsh,
        high_std_null_depth = False, signal_invert = False,save_location=None):

    """This function calculates depth for a single flaw instances in row (single row of df)

    Args:
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
            
        NB_lags(np.ndarray) : The lags from performing cross-correlations on the entire A-scan
        peak_posn_diff_thrsh(int): Min position difference between surface and flaw feature/peak, report np.nan when not satisfied
        high_std_null_depth(bool): True, if frames with high std and null depth are present: used to lower amp thresh only for these frames
        signal_invert(bool): Also calculate the depth using inverted a-scans
        save_location(str): Location to save the plots
            
    Returns:
            pred_depth(list): Deepest/max depth from main fwg and after removing fwg
            pred_depth_loc(list): Location of the deepest depth from main fwg and after removing fwg
            reflection_std_amp_arr(list): list of arrays, contains reflections stats: std, amp, amp ratio, amp diff & strong reflections
            cost(float): DTW cost
            reflections_stat(list): list of reflection stats
            high_std_null_depth_flaw_list(list): True for frames with std and null depth
            flaw_peak_amp(int): Selected flaw feature amp value
            depth_flaw_arr(list): list of arrays, contains predicted depth from main fwg and fwg removed for every ax and circ loc
            surface_circ_loc(list): Surface Circumential location corresponds to flaw a-scan with deepest feature
    """

    global REFLECTION_AMP_THRSH # Min amp threshold for reflections
    # file_logger.info(f"Calculating depth for Indication: {row['Indication']}, Filename: {row['Filename']}")

    
    n_frame_reflections = 0 # no of frames with reflections
    reflections_fwg = False # True, for any frame inside the flaw extent, if reflections condition satisfies based on the fwg
    reflections_fwg_flaw_list = [] # store "reflections_fwg" for complete flaw, shape: no of frames * 1
    n_multi_reflections_flaw_list = [] # no of multi reflections at every location, shape: no of frames * no circ locs
    high_std_null_depth_flaw_list = [] # True for frames with std and null depth, shape: no of frames * 1
    flaw_peak_value_flaw_list = [] # Flaw peak amp values: no of frames * no circ locs
    flaw_peak_fwg_rmv_value_flaw_list = []

    # define reflections array with shape: no of frames * no circ locs
    ones_arr = np.ones((row['Ax End'] - row['Ax Start'], row['Ro End'] - row['Ro Start'])) * np.nan
    reflection_std_arr = ones_arr.copy()
    reflection_amp_arr = ones_arr.copy()
    reflection_amp_ratio_arr = ones_arr.copy()
    reflection_amp_diff_arr = ones_arr.copy()
    reflection_strong_arr = ones_arr.copy()
    
    # reflection stats
    reflection_std_amp_arr = [reflection_std_arr, reflection_amp_arr,
                              reflection_amp_ratio_arr, reflection_amp_diff_arr,
                              reflection_strong_arr]

    depth_flaw = [] # pred depth for every location, shape: no of frames * no circ locs
    depth_flaw_fwg_rmv = [] # pred depth after removing the fwg for every location, shape: no of frames * no circ locs
    cost_flaw = [] # DTW cost for every location, shape: no of frames * no circ locs
    surface_circ_flaw = [] # store surface circ for every location, shape: no of frames * no circ locs
        
    # invert the signal
    if signal_invert:
        nb_data_arr = find_signal_invert(nb_data_arr)
        
    # predicted flaw extent coordinates
    pred_ax_start = row['Ax Start']
    pred_ax_end = row['Ax End']
    pred_circ_start = row['Ro Start']
    pred_circ_end = row['Ro End']
    
    # loop through every frame
    for ax in range(pred_ax_start, pred_ax_end):
        
        REFLECTION_AMP_THRSH = config[f'REFLECTION_AMP_THRSH_{ITERATION}']

        # calculate depth for a frame
        [depth_ax_list, depth_ax_fwg_rmv_list], cost_normalized_list,\
            scan_stats, n_circ_reflections, reflections_fwg_frame_list,\
                open_surface_frame_list, n_multi_reflections_list, flaw_peak,\
                      surface_circ_loc_ax_list = find_depth_single_frame(nb_data_arr,
                                                                         row, ax,
                                                                         reflection_std_amp_arr,
                                                                         NB_lags,
                                                                         peak_posn_diff_thrsh,
                                                                         PLOTTING_GLOBAL,
                                                                         save_location)
        
        #check if any of the frames has null depth and high std: this could be due to the presence of low amp reflections
        frame_high_std_low_amp = np.any(find_consecutive_true(reflection_std_arr[ax - row['Ax Start']] >= 1, 3)) and np.all(np.isnan(depth_ax_fwg_rmv_list))
        high_std_null_depth_flaw_list.append(frame_high_std_low_amp)
        pred_depth_high_std_low_amp = high_std_null_depth
        
        # Iteratively decrease reflection amp thrsh for frames with high std and null depth
        # update/change the depth_ax_fwg_rmv_list with the new one that searches for low amp reflections
        while pred_depth_high_std_low_amp and frame_high_std_low_amp:
            REFLECTION_AMP_THRSH -= 2
            [depth_ax_list, depth_ax_fwg_rmv_list], cost_normalized_list,\
                scan_stats, n_circ_reflections, reflections_fwg_frame_list,\
                    open_surface_frame_list, n_multi_reflections_list, flaw_peak,\
                          surface_circ_loc_ax_list = find_depth_single_frame(nb_data_arr,
                                                                             row, ax,
                                                                             reflection_std_amp_arr,
                                                                             NB_lags,
                                                                             peak_posn_diff_thrsh,
                                                                             PLOTTING_GLOBAL,
                                                                             save_location)
# stop if low amp reflections found or threshold drops below 134
            if not np.all(np.isnan(depth_ax_fwg_rmv_list)) or REFLECTION_AMP_THRSH < 134:
                pred_depth_high_std_low_amp = False
        
        [flaw_peak_value_list, flaw_fwg_rmv_peak_value_list] = flaw_peak

        # no of frames with reflections
        if n_circ_reflections != 0:
            
            n_frame_reflections+=1
        
        # for those circ locs where the surface is open do not consider the reflections
        open_surface_frame_list = np.array(open_surface_frame_list)
        reflections_fwg_frame_list = np.array(reflections_fwg_frame_list)
        
        if not multi_reflection:
            reflection_ignore_condn = open_surface_frame_list.any()
        else:
            reflection_ignore_condn = open_surface_frame_list.any() and max(depth_ax_list) >= 0.1
            
        if reflection_ignore_condn:
            # print('Reflections ignored due to the open surface')
            reflections_fwg_frame_list = ignore_reflections_open_surface(reflections_fwg_frame_list,
                                                                        open_surface_frame_list,
                                                                        open_surface_near_circ_loc = 1)
                    
        # only consider reflections present at consecutive circ locs  
        reflections_fwg_frame_list = find_consecutive_true(reflections_fwg_frame_list,
                                                           MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH)
        reflections_fwg = np.any(reflections_fwg_frame_list)
            
        # removing depth points which can't be seen in consecutive circ loc
        depth_ax_fwg_rmv_list = np.array(depth_ax_fwg_rmv_list) * reflections_fwg_frame_list
        depth_ax_fwg_rmv_list[depth_ax_fwg_rmv_list == 0] = np.nan
        
        # ignore sudden changes in the depth
        depth_ax_list = remove_abrupt_change(depth_ax_list, max_diff = 0.5)
        depth_ax_fwg_rmv_list = remove_abrupt_change(depth_ax_fwg_rmv_list, max_diff = 0.5)
        
        
        # if open surface found get the depth from the middle of the open surface
        open_surface_idxs = np.where(reflections_fwg_frame_list)[0]
        if multi_reflection and len(open_surface_idxs)>=2:
            depth_ax_list, depth_ax_fwg_rmv_list = open_surface_middle_depth(open_surface_idxs,
                                                                             depth_ax_list,
                                                                             depth_ax_fwg_rmv_list)

        # predicted depth list
        depth_flaw.append(depth_ax_list)
        depth_flaw_fwg_rmv.append(depth_ax_fwg_rmv_list)

        # surface circ location list
        surface_circ_flaw.append(surface_circ_loc_ax_list)
        
        # DTW cost list
        cost_flaw.append(cost_normalized_list)
        
        # flaw peak amp list
        flaw_peak_value_flaw_list.append(flaw_peak_value_list)
        flaw_peak_fwg_rmv_value_flaw_list.append(flaw_fwg_rmv_peak_value_list)

        # reflections
        reflections_fwg_flaw_list.append(reflections_fwg)
        n_multi_reflections_flaw_list.append(n_multi_reflections_list)
        
    # predicted depth is max of all
    depth_flaw_fwg = np.array(depth_flaw)
    depth_flaw_fwg_rmv_arr = np.array(depth_flaw_fwg_rmv)
    assert depth_flaw_fwg.shape == (pred_ax_end - pred_ax_start, pred_circ_end - pred_circ_start)
    pred_depth_flaw = np.nanmax(depth_flaw_fwg)
    pred_depth_flaw_fwg_rmv = np.nanmax(depth_flaw_fwg_rmv_arr)
    surface_circ_flaw = np.array(surface_circ_flaw)
    
    # Flaw deepest depth location
    try:
        (flaw_ax, flaw_circ) = unravel_index(np.nanargmax(depth_flaw_fwg), depth_flaw_fwg.shape)
    except:
        # if no depth found
        (flaw_ax, flaw_circ) = (np.nan, np.nan)
        
    try:
        (flaw_fwg_rmv_ax, flaw_fwg_rmv_circ) = unravel_index(np.nanargmax(depth_flaw_fwg_rmv_arr), depth_flaw_fwg_rmv_arr.shape)
    except:
        # if no depth found
        (flaw_fwg_rmv_ax, flaw_fwg_rmv_circ) = (np.nan, np.nan)
    
    # no of multi reflections: max of all
    n_multi_reflections_flaw_list = np.array(n_multi_reflections_flaw_list)
    try:
        n_multi_reflections = np.nanmax(n_multi_reflections_flaw_list)
    except:
        n_multi_reflections = np.nan
        
    # flaw peak amp
    try:
        flaw_peak_value_flaw_list = np.array(flaw_peak_value_flaw_list)
        flaw_peak_amp = flaw_peak_value_flaw_list[flaw_ax, flaw_circ]
    except:
        flaw_peak_amp = np.nan

    try:
        flaw_peak_fwg_rmv_value_flaw_list = np.array(flaw_peak_fwg_rmv_value_flaw_list)
        flaw_peak_amp_fwg_rmv = flaw_peak_fwg_rmv_value_flaw_list[flaw_fwg_rmv_ax, flaw_fwg_rmv_circ]
    except:
        flaw_peak_amp_fwg_rmv = np.nan

    try:
        surface_circ = surface_circ_flaw[flaw_ax, flaw_circ]
    except:
        surface_circ = np.nan
    try:
        surface_circ_fwg_rmv = surface_circ_flaw[flaw_fwg_rmv_ax, flaw_fwg_rmv_circ]
    except:
        surface_circ_fwg_rmv = np.nan
        
    surface_circ_loc = [surface_circ, surface_circ_fwg_rmv]

    # ax & circ loc w.r.t PT start
    flaw_ax += pred_ax_start + 1
    flaw_circ += pred_circ_start
    flaw_fwg_rmv_ax += pred_ax_start + 1
    flaw_fwg_rmv_circ += pred_circ_start
    
    # DTW cost
    cost_flaw = np.array(cost_flaw)
    try:
        cost = cost_flaw[flaw_ax - (pred_ax_start + 1), flaw_circ - (pred_circ_start)]
    except:
        cost = np.nan
      
    pred_depth = [pred_depth_flaw, pred_depth_flaw_fwg_rmv]
    pred_depth_loc = [[flaw_ax, flaw_circ], [flaw_fwg_rmv_ax, flaw_fwg_rmv_circ]]
    reflections_stat = [n_frame_reflections, reflections_fwg_flaw_list, n_multi_reflections]
    depth_flaw_arr = [depth_flaw_fwg, depth_flaw_fwg_rmv_arr]
    flaw_peak = [flaw_peak_amp, flaw_peak_amp_fwg_rmv]

    return pred_depth, pred_depth_loc, reflection_std_amp_arr, cost,\
          reflections_stat, high_std_null_depth_flaw_list, flaw_peak, depth_flaw_arr, surface_circ_loc
    
def pred_debris_depth_single_iter_probe(df, NB_lags, nb_data_arr, probe, iteration,save_location):
    
    """This function calculates depth for all Debris flaw instances in df using B-scans

    Args:
        df(dataframe): dataframe of every flaw to size the depth of, must all be within the same B-scan file. required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
            
        NB_lags(np.ndarray) : The lags from performing cross-correlations on the entire A-scan
        nb_data_arr(np.ndarray): B-scan
        probe(str): Probe to use to calculate depth, for example: 'NB1'
        iteration(int): no of iterations, some constant values changes in each iteration
        save_location(str): Location to save the plots

    Returns:
        results_df(dataframe): results of depth calculation with columns:
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'pred_depth' (predicted depth in mm)
            -'flaw_ax' (frame location of deepest location,starts with 1)
            -'flaw_circ(circ location of deepest location,starts with 0)
            -'surface_ax' (surface frame location,starts with 1)
            -'surface_circ(surface circ location,starts with 0)
            -'Flaw Maximum Amplitude' (the maximum amplitude of the peaks used to measure the flaw)
            
        missing_df: dataframe of flaws for which a depth could not be calculated, has columns:
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Error' (error that caused the indication to not be measured)
        
        pred_depth_whole_bscan(list): list of arrays, contains predicted depth from main fwg, inverted, fwg removed and inverted and fwg removed for every ax and circ loc

    """
    global PROBE
    global ITERATION
    global multi_reflection # True, if multiple features present in the reflections
    global MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH # Min no of consecutive circ locn with reflections inside a single frame to consider the reflections
    global OPEN_SURFACE_AMP_DIFF_THRSH # Amp thresh to consider the frame as open surface: if True, reflections could be ignored
    global flaw_dic_ax

    file_logger.info(f"Calculating flaw depth using probe: {probe} & Iteration: {iteration}")
    # dataframes to store the results
    df_all = pd.DataFrame({})
    
    # Lists to store predicted depth
    pred_depth_all = []
    pred_depth_invert_all = []
    pred_depth_fwg_rmv_all = []
    pred_depth_fwg_rmv_invert_all = []
    pred_depth_first_peak = []
    
    # Lists to store predicted depth for each ax and circ locs
    pred_depth_whole_bscan_all = []
    pred_depth_whole_bscan_invert_all = []
    pred_depth_whole_bscan_fwg_rmv_all = []
    pred_depth_fwg_rmv_whole_bscan_invert_all = []
    
    # Lists to store the flaw deepest location
    flaw_ax_all = []
    flaw_circ_all = []
    flaw_ax_invert_all = []
    flaw_circ_invert_all = []
    flaw_fwg_rmv_ax_all = []
    flaw_fwg_rmv_circ_all = []
    flaw_fwg_rmv_ax_invert_all = []
    flaw_fwg_rmv_circ_invert_all = []
    
    # Flaw peak amp
    flaw_peak_amp_all = []
    flaw_peak_amp_fwg_rmv_all = []
    flaw_peak_amp_invert_all = []
    flaw_peak_amp_fwg_rmv_invert_all = []

    max_amp_overall_all = []
    
    # Flaw info
    indication = []
    
    # Lists to store reflection stats
    ref_std_all = []
    ref_amp_all = []
    ref_amp_ratio_all = []
    ref_amp_diff_all = []
    frame_ref_all = []
    n_strong_reflections_all = []
    reflections_fwg_all = []
    n_multi_reflections_all = []
    n_frame_high_std_low_amp_all= []
    
    # Lists to store DTW cost
    cost_normalized_all = []
    
    # surface location
    surface_circ_all = []
    surface_circ_fwg_rmv_all = []
    surface_circ_invert_all = []
    surface_circ_fwg_rmv_invert_all = []
    
    # Dict to store IDs & Indications for which a depth could not be calculated 
    missing_df = {'Indication': [], 'Error': []}
    
    # setting local to global variables
    PROBE = probe
    ITERATION = iteration
    
    flaw_dic_ax, _ = flaw_loc_file(df, nb_data_arr)

    
    # loop through all flaw instances
    for i, row in df.iterrows():
        multi_reflection = False
        MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH = config[f'MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH_{ITERATION}']
        OPEN_SURFACE_AMP_DIFF_THRSH = 100
        
        try:
        # try to calculate the depth, if depth could not be calculated, use missing_df dataframe to add ID and Indication
            
            # calculate depth using original a-scans i.e signal_invert = False
            file_logger.info(f"Calculating flaw depth using positive features")
            pred_depth, pred_depth_loc, reflection_std_amp_arr, cost,\
                reflections_stat, high_std_null_depth_flaw_list,\
                    flaw_peak_amp_flaw, pred_depth_whole_bscan, surface_circ_loc = find_depth_single_flaw(row,
                                                                                                        NB_lags,
                                                                                                        nb_data_arr,
                                                                                                        peak_posn_diff_thrsh = 13,
                                                                                                        high_std_null_depth = False,
                                                                                                        signal_invert = False,
                                                                                                        save_location=save_location)
    
            # no of frames with high std reflections but low amp: used in flagging model
            n_frame_high_std_low_amp = sum(high_std_null_depth_flaw_list)
    
            # if no of frame with null depth, high std and low amp > 2,
            # calculate the depth by lower the reflection amp thresh for these frames
            if n_frame_high_std_low_amp >= 2:
                # print('Reflections with high std and low amp found!!!')
                pred_depth, pred_depth_loc, reflection_std_amp_arr, cost,\
                    reflections_stat, high_std_null_depth_flaw_list,\
                        flaw_peak_amp_flaw, pred_depth_whole_bscan, surface_circ_loc = find_depth_single_flaw(row,
                                                                                                            NB_lags,
                                                                                                            nb_data_arr,
                                                                                                            peak_posn_diff_thrsh = 13,
                                                                                                            high_std_null_depth = True,
                                                                                                            signal_invert = False,
                                                                                                            save_location=save_location)
            [pred_depth_flaw, pred_depth_flaw_fwg_rmv] = pred_depth
            [[flaw_ax, flaw_circ], [flaw_fwg_rmv_ax, flaw_fwg_rmv_circ]] = pred_depth_loc
            [surface_circ, surface_circ_fwg_rmv] = surface_circ_loc
            [n_frame_reflections, reflections_fwg_flaw_list, n_multi_reflections] = reflections_stat
            [reflection_std_arr, reflection_amp_arr, reflection_amp_ratio_arr,
              reflection_amp_diff_arr, reflection_strong_arr] =reflection_std_amp_arr
            [pred_depth_whole_bscan_fwg, pred_depth_whole_bscan_fwg_rmv] = pred_depth_whole_bscan
    
    
            # reflections stats: used in flagging model
            if (not np.isnan(flaw_fwg_rmv_ax)) and (not np.isnan(flaw_fwg_rmv_circ)):
                ref_amp = reflection_amp_arr[flaw_fwg_rmv_ax - row['Ax Start'] - 1, flaw_fwg_rmv_circ - row['Ro Start']]
                ref_std = reflection_std_arr[flaw_fwg_rmv_ax - row['Ax Start'] - 1, flaw_fwg_rmv_circ - row['Ro Start']]
                ref_amp_ratio = reflection_amp_ratio_arr[flaw_fwg_rmv_ax - row['Ax Start'] - 1, flaw_fwg_rmv_circ - row['Ro Start']]
                ref_amp_diff = reflection_amp_diff_arr[flaw_fwg_rmv_ax - row['Ax Start'] - 1, flaw_fwg_rmv_circ - row['Ro Start']]
            else:
                ref_amp = np.nan
                ref_std = np.nan
                ref_amp_ratio = np.nan
                ref_amp_diff = np.nan
    
            # total no of strong reflections inside a single flaw: used in flagging model
            n_strong_reflections = np.nansum(reflection_strong_arr)
    
            # reflections_fwg = True, if there is any frame with reflections: used in flagging model
            if np.array(reflections_fwg_flaw_list).any():
                reflections_fwg = True
            else:
                reflections_fwg = False
    
            # in case of multi features in the reflections, use left-most peak/feature
            if n_multi_reflections >= 3 and reflections_fwg and np.isnan(pred_depth_flaw):
                # print('Multi Reflections found!!!')
                multi_reflection = True
                MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH = max(MIN_N_CONSECUTIVE_CIRC_REFLECTIONS_THRSH - 1, 1)
                OPEN_SURFACE_AMP_DIFF_THRSH = 90
                [_, pred_depth_flaw_fwg_rmv_first_peak], pred_depth_loc,\
                    _, _, [_, _, _], _, flaw_peak_amp_flaw, pred_depth_whole_bscan,\
                          surface_circ_loc = find_depth_single_flaw(row,
                                                                    NB_lags,
                                                                    nb_data_arr,
                                                                    peak_posn_diff_thrsh = 13,
                                                                    signal_invert = False,
                                                                    save_location=save_location)
                pred_depth_first_peak.append(pred_depth_flaw_fwg_rmv_first_peak)
                [_, pred_depth_whole_bscan_fwg_rmv] = pred_depth_whole_bscan
                [[_, _], [flaw_fwg_rmv_ax, flaw_fwg_rmv_circ]] = pred_depth_loc
                [_, surface_circ_fwg_rmv] = surface_circ_loc
    
            else:
                pred_depth_first_peak.append(np.nan)
    
            # store flaw deepest location
            flaw_ax_all.append(flaw_ax)
            flaw_circ_all.append(flaw_circ)
            flaw_fwg_rmv_ax_all.append(flaw_fwg_rmv_ax)
            flaw_fwg_rmv_circ_all.append(flaw_fwg_rmv_circ)

            # store surface circ location
            surface_circ_all.append(surface_circ)
            surface_circ_fwg_rmv_all.append(surface_circ_fwg_rmv)
    
            # flaw peak amp
            [flaw_peak_amp, flaw_peak_amp_fwg_rmv] = flaw_peak_amp_flaw
            flaw_peak_amp_all.append(flaw_peak_amp)
            flaw_peak_amp_fwg_rmv_all.append(flaw_peak_amp_fwg_rmv)
    
            # store predicted depth
            pred_depth_all.append(pred_depth_flaw)
            pred_depth_fwg_rmv_all.append(pred_depth_flaw_fwg_rmv)
    
            # store predicted depth for full b-scan
            pred_depth_whole_bscan_all.append(pred_depth_whole_bscan_fwg)
            pred_depth_whole_bscan_fwg_rmv_all.append(pred_depth_whole_bscan_fwg_rmv)
    
            indication.append(row['Indication'])
    
            # store reflection stats
            ref_amp_all.append(ref_amp)
            ref_std_all.append(ref_std)
            ref_amp_ratio_all.append(ref_amp_ratio)
            ref_amp_diff_all.append(ref_amp_diff)
            frame_ref_all.append(n_frame_reflections)
            n_strong_reflections_all.append(n_strong_reflections)
            reflections_fwg_all.append(reflections_fwg)
            n_multi_reflections_all.append(n_multi_reflections)
            n_frame_high_std_low_amp_all.append(n_frame_high_std_low_amp)
    
            # store dtw cost
            cost_normalized_all.append(cost)
    
            # calculate depth using inverted a-scans
            if config['INVERT']:
                file_logger.info(f"Calculating flaw depth using positive features")
                pred_depth_invert, pred_depth_loc_invert, _, cost_invert,\
                    reflections_stat_invert, _, flaw_peak_amp_flaw_invert,\
                        pred_depth_whole_bscan_invert, surface_circ_loc_invert = find_depth_single_flaw(row,
                                                                                                NB_lags,
                                                                                                nb_data_arr,
                                                                                                peak_posn_diff_thrsh = 13,
                                                                                                high_std_null_depth = False,
                                                                                                signal_invert = True,
                                                                                                save_location=save_location)
                        
                [pred_depth_flaw_invert, pred_depth_flaw_fwg_rmv_invert] = pred_depth_invert
                [surface_circ_invert, surface_circ_fwg_rmv_invert] = surface_circ_loc_invert
                [[flaw_ax_invert, flaw_circ_invert], [flaw_fwg_rmv_ax_invert, flaw_fwg_rmv_circ_invert]] = pred_depth_loc_invert
                [n_frame_reflections_invert, reflections_fwg_flaw_list_invert, n_multi_reflections_invert] = reflections_stat_invert
                [pred_depth_whole_bscan_invert, pred_depth_whole_bscan_fwg_rmv_invert] = pred_depth_whole_bscan_invert
    
                # store flaw deepest location
                flaw_ax_invert_all.append(flaw_ax_invert)
                flaw_circ_invert_all.append(flaw_circ_invert)
                flaw_fwg_rmv_ax_invert_all.append(flaw_fwg_rmv_ax_invert)
                flaw_fwg_rmv_circ_invert_all.append(flaw_fwg_rmv_circ_invert)

                surface_circ_invert_all.append(surface_circ_invert)
                surface_circ_fwg_rmv_invert_all.append(surface_circ_fwg_rmv_invert)
    
                # flaw peak amp
                [flaw_peak_amp_invert, flaw_peak_amp_fwg_rmv_invert] = flaw_peak_amp_flaw_invert
                flaw_peak_amp_invert_all.append(flaw_peak_amp_invert)
                flaw_peak_amp_fwg_rmv_invert_all.append(flaw_peak_amp_fwg_rmv_invert)
    
                # store predicted depth
                pred_depth_invert_all.append(pred_depth_flaw_invert)
                pred_depth_fwg_rmv_invert_all.append(pred_depth_flaw_fwg_rmv_invert)
    
                # store predicted depth for full b-scan
                pred_depth_whole_bscan_invert_all.append(pred_depth_whole_bscan_invert)
                pred_depth_fwg_rmv_whole_bscan_invert_all.append(pred_depth_whole_bscan_fwg_rmv_invert)
    
            # Overall max amp
            flaw_extent = nb_data_arr[row['Ax Start'] : row['Ax End'] + 1, row['Ro Start'] : row['Ro End'] + 1]
            max_amp_overall = np.max(flaw_extent)
            max_amp_overall_all.append(max_amp_overall)
                    
        # if there is any error while calculating depth, add info to missing_df and continue
        except Exception as e:
            file_logger.error(f'Error in calculating depth for flaw instance: {row["Indication"]}: {e}. Continuing to next instance.', exc_info=True)
            missing_df['Indication'].append(row['Indication'])
            missing_df['Error'].append(e)
            continue
        
    missing_df = pd.DataFrame(missing_df)
    
    # flaw info
    df_all['Indication'] = indication
    
    # flaw deepest location
    df_all['flaw_ax'] = flaw_ax_all
    df_all['flaw_circ'] = flaw_circ_all
    df_all['flaw_fwg_rmv_ax'] = flaw_fwg_rmv_ax_all
    df_all['flaw_fwg_rmv_circ'] = flaw_fwg_rmv_circ_all
    
    # surface feature location
    df_all['surface_ax'] = flaw_ax_all
    df_all['surface_circ'] = surface_circ_all
    df_all['surface_fwg_rmv_ax'] = flaw_fwg_rmv_ax_all
    df_all['surface_fwg_rmv_circ'] = surface_circ_fwg_rmv_all

    
    # DTW cost
    df_all['cost_norm'] = cost_normalized_all
    
    # Predicted depth
    df_all['pred_depth'] = pred_depth_all
    df_all['pred_depth_fwg_rmv'] = pred_depth_fwg_rmv_all
    df_all['pred_depth_first_peak'] = pred_depth_first_peak

    
    # reflection stats
    df_all['ref_std'] = ref_std_all
    df_all['ref_amp'] = ref_amp_all
    df_all['ref_amp_diff'] = ref_amp_diff_all
    df_all['ref_amp_ratio'] = ref_amp_ratio_all
    df_all['n_frame_reflections'] = frame_ref_all
    df_all['n_circ_strong_reflections'] = n_strong_reflections_all
    df_all['reflections_fwg'] = reflections_fwg_all
    df_all['n_multi_reflections'] = n_multi_reflections_all
    df_all['n_frame_high_std_low_amp'] = n_frame_high_std_low_amp_all
    
    # Flaw peak amp
    df_all['flaw_feature_amp'] = flaw_peak_amp_all
    df_all['flaw_feature_amp_fwg_rmv'] = flaw_peak_amp_fwg_rmv_all
    df_all['flaw_feature_amp_invert'] = flaw_peak_amp_invert_all
    df_all['flaw_feature_amp_fwg_rmv_invert'] = flaw_peak_amp_fwg_rmv_invert_all
    df_all['flaw_max_amp'] = max_amp_overall_all

    if config['INVERT']:
        
        # flaw deepest location using inverted a-scans
        df_all['flaw_ax_invert'] = flaw_ax_invert_all
        df_all['flaw_circ_invert'] = flaw_circ_invert_all
        df_all['flaw_fwg_rmv_ax_invert'] = flaw_fwg_rmv_ax_invert_all
        df_all['flaw_fwg_rmv_circ_invert'] = flaw_fwg_rmv_circ_invert_all

        
        # Predicted depth using inverted a-scans
        df_all['pred_depth_invert'] = pred_depth_invert_all
        df_all['pred_depth_fwg_rmv_invert'] = pred_depth_fwg_rmv_invert_all
        
        df_all['pred_depth_invert_reflections'] = df_all['pred_depth'].fillna(df_all['pred_depth_invert'])
        df_all['pred_depth_invert_reflections'] = df_all['pred_depth'].fillna(df_all['pred_depth_fwg_rmv'])

        # surface feature location
        df_all['surface_ax_invert'] = flaw_ax_invert_all
        df_all['surface_circ_invert'] = surface_circ_invert_all
        df_all['surface_fwg_rmv_ax_invert'] = flaw_fwg_rmv_ax_invert_all
        df_all['surface_fwg_rmv_circ_invert'] = surface_circ_fwg_rmv_invert_all
        
    pred_depth_whole_bscan = [pred_depth_whole_bscan_all, pred_depth_whole_bscan_fwg_rmv_all,\
                               pred_depth_whole_bscan_invert_all, pred_depth_fwg_rmv_whole_bscan_invert_all]

    return df_all, missing_df, pred_depth_whole_bscan


def pred_debris_depth(df, b_scans, NB_lags_whole, NB_lags_g4, conf, plotting, probes, save_files, run_name,save_location, out_root):
    global config, PLOTTING_GLOBAL, SAVE_ROOT, FILENAME, FILENAME_FULL, CHANNEL, OUTAGE_NUMBER, RUN_NAME, FLAW_TYPE

    file_logger.info(f"Initialising the depth sizing.")
    config = conf
    PLOTTING_GLOBAL = plotting
    RUN_NAME = run_name
    if save_files:
        SAVE_ROOT = os.path.join(out_root, run_name)
        OUTAGE_NUMBER = df['Outage Number'].unique()[0]
        CHANNEL = df['Channel'].unique()[0]
        FILENAME_FULL = df['Filename'].unique()[0]
        FILENAME = FILENAME_FULL.split('.')[0]
        
    df = df.sort_values(by='Indication')
    FLAW_TYPE = df['flaw_type'].unique()[0]
    df_orig = df.copy()

    ITERATIONS = [0]
    df_all_probe_iter = []
    missing_df_all = pd.DataFrame()
    pred_depth_whole_bscan_all_iter_probe = {}
    
    total_iterations = len(ITERATIONS) * len(probes)
    
    for iteration, probe in itertools.product(ITERATIONS, probes):
        nb_data_arr = b_scans[0] if probe == 'NB1' else b_scans[1]
        NB_lags = NB_lags_whole[0] if probe == 'NB1' else NB_lags_whole[1]
        
        df_all_cols, missing_df, pred_depth_whole_bscan = pred_debris_depth_single_iter_probe(df, NB_lags, nb_data_arr, probe, iteration,save_location)
        df_all_probe_iter.append(df_all_cols)
        missing_df['ITERATION'] = iteration
        #missing_df['PROBE'] = probe
        missing_df_all = missing_df_all.append([missing_df], ignore_index=True)
        pred_depth_whole_bscan_all_iter_probe[probe + str(iteration)] = pred_depth_whole_bscan
        
        if save_files:
            file_logger.info('Saving intermediate files.')
            save_path = os.path.join(save_location, 'Depth Sizing', 'Debris all iterations results')
            os.makedirs(os.path.join(save_path), exist_ok=True)
            df_all_cols.to_excel(os.path.join(save_path, f'pred_depth_{FLAW_TYPE}_iter{iteration}_{probe}.xlsx'))

    file_logger.info('Post processing the output from all iterations')
    df_all_cols = debris_post_processor(df_all_probe_iter, conf)
    
    if save_files:
        save_path = os.path.join(save_location, 'Depth Sizing', 'Debris all iterations results')
        os.makedirs(os.path.join(save_path), exist_ok=True)
        df_all_cols.to_excel(os.path.join(save_path, f'pred_depth_{FLAW_TYPE}_post_processed.xlsx'))
        
    if df_all_cols.empty:
        stats_df = df_orig.copy()
        cols = [col for col in df_all_cols.columns if col not in stats_df.columns]
        stats_df[cols] = np.nan
    else:
        stats_df = pd.merge(left=df_all_cols, right=df_orig, how='left', on=['Indication']) 

    file_logger.info('Measuring Chatter')
    [chatter_nb1, chatter_nb2] = measure_chatter_all(stats_df, NB_lags_whole, NB_lags_g4)

    file_logger.info('Plotting depth profile')
    try:
        stats_df = stats_df.sort_values(by='Indication')
        if (stats_df['pred_depth_nb1_nb2'].isnull()).all():
            raise ValueError('No measured depth!')
        is_depth_fwg_rmv_all, is_depth_inverted_all, flaw_ax_all, flaw_circ_all, flaw_peak_all = plot_depth_profile_all_flaws(stats_df, pred_depth_whole_bscan_all_iter_probe,save_location)
    except Exception as e:
        if save_files:
            save_path = os.path.join(SAVE_ROOT, 'Depth Sizing Profile', FILENAME)
            os.makedirs(save_path, exist_ok=True)
            with open(os.path.join(save_path, f"{OUTAGE_NUMBER} {CHANNEL}.txt"), 'w') as file:
                inds = stats_df['Indication'].to_list()
                error_msg = f'Not able to plot depth profile for Indications: {inds}!!!\nError: ' + str(e)
                file_logger.error(error_msg, exc_info=True)
                file.write(error_msg)
        [is_depth_fwg_rmv_all, is_depth_inverted_all, flaw_ax_all, flaw_circ_all, flaw_peak_all] = [np.nan]*5

    results_col = ['Indication', 'pred_depth_nb1_nb2', 'flaw_max_amp']
    results_df = stats_df.loc[:, results_col]
    results_df['flaw_ax'] = stats_df['flaw_ax']
    results_df['flaw_circ'] = stats_df['flaw_circ']
    results_df['flaw_feature_amp'] = flaw_peak_all
    results_df['chatter_amp_nb1'] = chatter_nb1
    results_df['chatter_amp_nb2'] = chatter_nb2
    results_df['note_2'] = np.nan
    results_df['probe_depth'] = stats_df['PROBE']
    results_df['probe_depth'].fillna('NB1', inplace=True)

    results_df.loc[results_df['probe_depth'] == 'NB1', 'chatter_amplitude'] = results_df['chatter_amp_nb1']
    results_df.loc[results_df['probe_depth'] == 'NB2', 'chatter_amplitude'] = results_df['chatter_amp_nb2']
    
    stats_df['pred_depth_fwg_rwv_inc'] = (stats_df['pred_depth_fwg_rmv'] - stats_df['pred_depth']) / stats_df['pred_depth']

    try:
        y = flag_cases_debris(stats_df, config)
        stats_df['flag_high_error'] = y 
        results_df['flag_high_error'] = y
        flag_high_error_true = results_df['flag_high_error'] == True
        results_df.loc[flag_high_error_true, 'pred_depth_nb1_nb2'] = results_df.loc[flag_high_error_true, 'pred_depth_nb1_nb2'].astype('str') + ' (2)'
        results_df.loc[flag_high_error_true, 'note_2'] = 'Measured depth is flagged, need to be reviewed'
    except:
        y = 'Not able to flag'
        file_logger.error(y, exc_info=True)
        stats_df['flag_high_error'] = y 
        results_df['flag_high_error'] = y
        
    stats_df['is_depth_inverted'] = is_depth_inverted_all
    stats_df['is_depth_fwg_rmv'] = is_depth_fwg_rmv_all
    stats_df['Indication'] = stats_df['Indication']
    stats_df['Filename'] = stats_df['Filename']
    results_df.rename(columns={'pred_depth_nb1_nb2': 'flaw_depth'}, inplace=True)
    
    results_df = update_flaw_feature_amp_inverted(stats_df.copy(), results_df.copy(), b_scans)

    if save_files:
        save_path = os.path.join(save_location, 'Depth Sizing')
        os.makedirs(save_path, exist_ok=True)
        stats_df.to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_nb_stats.xlsx'), index=False)  
        results_df.to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_nb_results.xlsx'), index=False)    
        missing_df_all.to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_nb_missing.xlsx'), index=False)

    return results_df, missing_df_all, stats_df