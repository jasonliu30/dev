import os
import numpy as np
import pandas as pd
from tqdm import tqdm
from numpy import unravel_index
from scipy import signal
import traceback
from depth_sizing.utils.flattening import Smoothlags
from depth_sizing.utils.CC_flagging import criteria_flagging, flagging_issues, deepest_flaw_bonus_peak,\
                                     bonus_peak_info, swap_axes
from depth_sizing.utils.CC_depth_utils import find_all_surface_posns_circ, find_surface_posn_circ,\
                                              find_surface_posn_ax, find_lag, nan_percentage, flaw_loc_file,\
                                              find_deepest_location, peak_data
from depth_sizing.utils.depth_sizing_flag import flag_cases
from depth_sizing.utils.chatter_magnitude import measure_chatter
from depth_sizing.utils.plot_utils import plot_depth_profile_single_flaw
from utils.logger_init import create_dual_loggers

# create loggers
_, file_logger  = create_dual_loggers()

def pred_depth_single_flaw(config,row,FLIP_ASCANS,APC_lags_whole_scan,APC_cor_g4,
                           APC_lags_g4,APC_wavegroups,apc_data_arr,flaw_dic_ax, 
                           flaw_dic_circ,PLOTTING=False,save_path='',save_files=False,save_location=''):
    """
    Calculate the predicted depth and related attributes of a single flaw.

    Parameters
    ----------
        config : dict
            Configuration file.
        row : pandas Series
            A row of data containing information about the flaw.
        FLIP_ASCANS : bool
            Flag to determine whether to flip the A-scans.
        APC_lags_whole_scan : ndarray
            Lag data for the entire scan.
        APC_cor_g4 : ndarray
            APC correlation data for group 4.
        APC_lags_g4 : ndarray
            Lag data for group 4.
        APC_wavegroups : ndarray
            Wavegroup information for APC data.
        apc_data_arr : ndarray
            Raw APC data array.
        flaw_dic_ax : dict
            Dictionary containing axial flaw information.
        flaw_dic_circ : dict
            Dictionary containing circumferential flaw information.
        PLOTTING : bool, optional
            Flag to enable/disable plotting (default is False).
        save_path : str, optional
            Path to save the files if required (default is an empty string).
        save_files : bool, optional
            Flag to enable/disable saving files (default is False).
        save_location : str, optional

    Returns
    -------
        pred_ax_start, pred_ax_end, pred_circ_start, pred_circ_end : int
            Predicted start and end coordinates for the axial and circumferential flaw extent.
        flaw_ax, flaw_circ : int
            Axial and circumferential flaw coordinates.
        total_missing_depths_percent : float
            Percentage of missing depths within the flaw.
        pred_depth_flaw, pred_depth_flaw_up, pred_depth_flaw_bottom, pred_depth_flaw_left, pred_depth_flaw_right : float
            Predicted flaw depths at various locations.
        surface_circ_selected, surface_ax_selected : int
            Selected circumferential and axial surface coordinates.
        depth_flaw_arr, depth_flaw_arr_unfiltered, dtw_range_flaw, dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw : ndarray
            Arrays containing depth and related attributes for the flaw.
        all_frame_peaks, APC_cors, all_curvature_corrections, all_frame_bonus_peaks, all_frame_surface_peaks, depth_flaw_start_of_peak_arr : list
            Lists containing various flaw peak information and curvature corrections.
    """
    # Depth-related lists
    depth_flaw, depth_flaw_unfiltered, depth_flaw_up, depth_flaw_bottom, depth_flaw_left, depth_flaw_right, depth_flaw_start_of_peak = ([] for _ in range(7))

    # Surface and DTW-related lists
    surface_circ_flaw, surface_ax_flaw, dtw_range_flaw, dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw = ([] for _ in range(6))

    # Flaw peak info lists
    all_frame_surface_peaks, all_frame_peaks, all_frame_bonus_peaks, all_curvature_corrections = ([] for _ in range(4))

    #file_logger.info(f"Calculating depth for Indication: {row['Indication']}, Filename: {row['Filename']}")

    # Predicted flaw extent coordinates and corresponding APC values
    pred_ax_start, pred_ax_end, pred_circ_start, pred_circ_end = row['Ax Start'], row['Ax End'], row['Ro Start'], row['Ro End']
    APC_cors = APC_cor_g4[pred_ax_start:pred_ax_end + 1, pred_circ_start:pred_circ_end + 1]

    for ax in range(pred_ax_start, pred_ax_end + 1):
        
        # smooth the frame of lags to get the general trend
        if config['FLATTENING']:
            # Pick the lags to use for curvature correction
            if np.std(APC_lags_whole_scan[ax]) <= config['LAG_STD_THRESHOLD']:
                lags = APC_lags_whole_scan[ax]
            elif np.std(APC_lags_g4[ax]) <= config['LAG_STD_THRESHOLD']:
                lags = APC_lags_g4[ax]
            else:
                lags = np.zeros(APC_lags_whole_scan[ax].shape)

            pressure_tube_location = Smoothlags(lags,config['SD_LIMIT'],config['MAXITER'],config['SG_ORD'],config['SG_FLEN'],config['OUTLIER_GRADIENT'])[0]

            flattened_lags = np.abs(lags - pressure_tube_location)

        else:
            flattened_lags = lags = np.zeros(APC_lags_whole_scan[ax].shape)

        frame = ax + 1

        # Depth lists
        depth_ax_list, depth_ax_up_list, depth_ax_bottom_list, depth_ax_left_list, depth_ax_right_list, depth_ax_start_of_peak_list = ([] for _ in range(6))

        # Surface lists
        surface_circ_list, surface_ax_list = [], []

        # other lists
        var_prcnt_list, amp_diff_list, flaw_peak_value_list, cost_normalized_list, cost_focus_normalized_list, surface_range_fix_list, flaw_range_list = ([] for _ in range(7))
        flaw_peak_ratio_list = []

        # Frame lists
        frame_surface_peaks, frame_peaks, frame_bonus_peaks, frame_curvature_corrections = ([] for _ in range(4))

        if PLOTTING and save_files:
            os.makedirs(os.path.join(save_path, f"Frame-{frame}", "full_frame_analysis"), exist_ok=True)
            
        
        for n_circ, circ in enumerate(range(pred_circ_start, pred_circ_end + 1)):
            
            lag_correction = 0

            if PLOTTING and save_files:
                os.makedirs(os.path.join(save_path, f"Frame-{frame}", "full_frame_analysis"), exist_ok=True)
            
    
            if FLIP_ASCANS:
                flaw_a_scan=((apc_data_arr[ax, circ, :] - config['MEDIAN_VALUE'])*(-1))+config['MEDIAN_VALUE']
            else:
                flaw_a_scan = apc_data_arr[ax, circ, :]
                
            (flaw_start, flaw_stop, _, _, _, _) = APC_wavegroups[ax][circ]
            use_averaged_depth=False
            if config['DEPTH_AVERAGING']:
                file_logger.debug("Selecting surface signal...")
                file_logger.debug("Selecting surface feature...")
                avg_surface_peak_pos_list, avg_surface_a_scan_list, avg_surface_wave_loc_list,\
                avg_surface_circ_list, avg_surface_ax_list, avg_peak_list, avg_surface_start_list,\
                avg_surface_stop_list, avg_surface_not_found_reason, avg_surface_found_list=find_all_surface_posns_circ(config,apc_data_arr, ax, pred_circ_start, pred_circ_end, APC_wavegroups, row, flattened_lags, flaw_dic_ax, range_circ = config['SURFACE_CIRC_RANGE'],lag_difference_thresh=config['SURFACE_LAG_DIFFERENCE_CRITERIA'],flip_ascans=FLIP_ASCANS)
                if sum(avg_surface_found_list)>0:
                    use_averaged_depth=True
                    avg_depths=[]
                    for surface_num in range(len(avg_surface_peak_pos_list)):
                        start = min(avg_surface_start_list[surface_num], flaw_start)
                        stop = max(avg_surface_stop_list[surface_num], flaw_stop)
                        surface_a_scan_fwg = avg_surface_a_scan_list[surface_num][start : stop]
                        a_scan_fwg = flaw_a_scan[start : stop]
                        a_scan_fwg_max = np.max(a_scan_fwg)
                        title = str(row['Filename'])+'_'+str(row['Indication'])+'_frame_'+str(frame)+'_circ_'+str(circ)
                        file_logger.debug("Selecting flaw feature...")
                        surface_peak_posn, surface_peak_value, flaw_peak_posn, flaw_peak_value, lag, path, cost_normalized, cost_focus_normalized, surface_range_fix, flaw_range = find_lag(config,a_scan_fwg, surface_a_scan_fwg, avg_surface_peak_pos_list[surface_num], start, stop, trim = 0,title=title)
                        # Correcting lag for curvature
                        lag_correction_avg=0
                        if config['FLATTENING']:
                            lag_correction_avg = (pressure_tube_location[avg_surface_circ_list[surface_num]] - pressure_tube_location[circ])
                        lag = lag - lag_correction_avg
                        avg_depths.append(max(np.round(lag * config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'], 4), 0))
                    depth_circ = np.mean(avg_depths)
            
            surface_peak_posn, surface_peak_posn_up, surface_peak_posn_bottom, surface_a_scan, surface_wave_loc, surface_circ_loc, surface_ax_loc, surface_peaks, surface_start, surface_stop, surface_not_found_reason, surface_found,surface_stats = find_surface_posn_circ(config,apc_data_arr, ax, pred_circ_start, pred_circ_end, APC_wavegroups, row, flattened_lags,flaw_dic_ax, range_circ = config['SURFACE_CIRC_RANGE'],lag_difference_thresh=config['SURFACE_LAG_DIFFERENCE_CRITERIA'],flip_ascans=FLIP_ASCANS)
            
            _, surface_peak_posn_left, surface_peak_posn_right, _, _, _, _, _, _, _, _, _ = find_surface_posn_ax(config,apc_data_arr, circ, pred_ax_start, pred_ax_end, APC_wavegroups, row, flaw_dic_circ, flip_ascans=FLIP_ASCANS)

            if not surface_found:
                warn = f'Surface not found for frame: {frame}, circ: {circ}'
                file_logger.debug(warn)
                # surface_not_found_list.append(warn)
                
                depth_circ = np.nan
                depth_circ_up = np.nan
                depth_circ_bottom = np.nan
                depth_ax_left = np.nan
                depth_ax_right = np.nan
                depth_circ_start_of_peak = np.nan
                var_prcnt_list.append(np.nan)
                amp_diff_list.append(np.nan)
                flaw_peak_value_list.append(np.nan)
                cost_normalized_list.append(np.nan)
                cost_focus_normalized_list.append(np.nan)
                surface_range_fix_list.append(np.nan)
                flaw_range_list.append(np.nan)
                flaw_peak_ratio_list.append(np.nan)
        
                depth_ax_list.append(depth_circ)
                depth_ax_up_list.append(depth_circ_up)
                depth_ax_bottom_list.append(depth_circ_bottom)
                depth_ax_left_list.append(depth_ax_left)
                depth_ax_right_list.append(depth_ax_right)
                depth_ax_start_of_peak_list.append(depth_circ_start_of_peak)
                
                surface_circ_list.append(surface_circ_loc)
                surface_ax_list.append(surface_ax_loc)

                peak = peak_data.copy()
                frame_peaks.append(peak)
                frame_bonus_peaks.append(peak)
                frame_surface_peaks.append(surface_stats)
                frame_curvature_corrections.append((-1)*lag_correction*config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'])
                continue

            frame_surface_peaks.append(surface_stats)
            start = min(surface_start, flaw_start)
            stop = max(surface_stop, flaw_stop)
            surface_a_scan_fwg = surface_a_scan[start : stop]
            a_scan_fwg = flaw_a_scan[start : stop]
            a_scan_fwg_max = np.max(a_scan_fwg)

            title = str(row['Filename'])+'_'+str(row['Indication'])+'_frame_'+str(frame)+'_circ_'+str(circ)
            surface_peak_posn, surface_peak_value, flaw_peak_posn, flaw_peak_value, lag, path, cost_normalized, cost_focus_normalized, surface_range_fix, flaw_range = find_lag(config,a_scan_fwg, surface_a_scan_fwg, surface_peak_posn, start, stop, trim = 0,title=title)
            
            # GETTING FLAW PEAK INFO
            peak_values,peak_stats = signal.find_peaks(a_scan_fwg,width =2)
            peak = peak_data.copy()
            arg_peak = np.argmin(np.abs(peak_values-(flaw_peak_posn-start)))
            peak['index'] = peak_values[arg_peak]+start
            peak['prominence'] = peak_stats['prominences'][arg_peak]
            peak['width'] = peak_stats['widths'][arg_peak]
            peak['height'] = a_scan_fwg[peak_values[arg_peak]]
            peak['left_base'] = peak_stats['left_bases'][arg_peak]+start
            peak['right_base'] = peak_stats['right_bases'][arg_peak]+start
            peak['start'] = start

            frame_peaks.append(peak)

            bonus_peak = peak_values[peak_values<peak_values[arg_peak]]
            
            if len(bonus_peak)>0:
                bonus_peak_index = bonus_peak[np.argmax(a_scan_fwg[bonus_peak])]
                arg_bonus_peak = np.where(peak_values==bonus_peak_index)[0][0]
                bonus_peak = peak_data.copy()
                bonus_peak['index'] = peak_values[arg_bonus_peak]+start
                bonus_peak['prominence'] = peak_stats['prominences'][arg_bonus_peak]
                bonus_peak['width'] = peak_stats['widths'][arg_bonus_peak]
                bonus_peak['height'] = a_scan_fwg[peak_values[arg_bonus_peak]]
                bonus_peak['left_base'] = peak_stats['left_bases'][arg_bonus_peak]+start
                bonus_peak['right_base'] = peak_stats['right_bases'][arg_bonus_peak]+start
                bonus_peak['start'] = start
                frame_bonus_peaks.append(bonus_peak)
            else:
                frame_bonus_peaks.append(peak_data.copy())

            # Correcting lag for curvature
            if config['FLATTENING']:
                lag_correction = (pressure_tube_location[surface_circ_loc] - pressure_tube_location[circ])
            
            lag = lag - lag_correction

            def calculate_depth(surface_peak_posn, flaw_peak_posn, unit_micro_sec, micro_sec_depth_shear):
                if surface_peak_posn != -1:
                    return max(np.round((surface_peak_posn - flaw_peak_posn) * unit_micro_sec * micro_sec_depth_shear, 4), 0)
                return np.nan

            if not use_averaged_depth:
                depth_circ = max(np.round(lag * config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'], 4), 0)

            depth_circ_up = calculate_depth(surface_peak_posn_up, flaw_peak_posn, config['UNIT_MICRO_SEC'], config['MICRO_SEC_DEPTH_SHEAR'])
            depth_circ_bottom = calculate_depth(surface_peak_posn_bottom, flaw_peak_posn, config['UNIT_MICRO_SEC'], config['MICRO_SEC_DEPTH_SHEAR'])
            depth_ax_left = calculate_depth(surface_peak_posn_left, flaw_peak_posn, config['UNIT_MICRO_SEC'], config['MICRO_SEC_DEPTH_SHEAR'])
            depth_ax_right = calculate_depth(surface_peak_posn_right, flaw_peak_posn, config['UNIT_MICRO_SEC'], config['MICRO_SEC_DEPTH_SHEAR'])

            if surface_stats['index'] != 0 and peak['index'] != 0:
                lag_start = (surface_stats['index'] - surface_stats['width']) - (peak['index'] - peak['width'])
                lag_start = lag_start - lag_correction
                depth_circ_start_of_peak = max(np.round(lag_start * config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'], 4), 0)
            else:
                depth_circ_start_of_peak = np.nan

            var_prcnt_list.append(np.var(a_scan_fwg) / np.var(surface_a_scan_fwg))
            amp_diff_list.append(surface_peak_value - flaw_peak_value)
            flaw_peak_value_list.append(flaw_peak_value)
            cost_normalized_list.append(cost_normalized)
            cost_focus_normalized_list.append(cost_focus_normalized)
            surface_range_fix_list.append(surface_range_fix)
            flaw_range_list.append(flaw_range)
            flaw_peak_ratio_list.append(flaw_peak_value / a_scan_fwg_max)

            # -- Calculating the depth from start of the peaks --       
            depth_ax_list.append(depth_circ)
            depth_ax_up_list.append(depth_circ_up)
            depth_ax_bottom_list.append(depth_circ_bottom)
            depth_ax_left_list.append(depth_ax_left)
            depth_ax_right_list.append(depth_ax_right)
            depth_ax_start_of_peak_list.append(depth_circ_start_of_peak)
            surface_circ_list.append(surface_circ_loc)
            surface_ax_list.append(surface_ax_loc)
            frame_curvature_corrections.append((-1)*lag_correction* config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'])

        depth_flaw_unfiltered.append(np.array(depth_ax_list))

        # runs for every frame
        if config['CONSTRAINTS'] == True:
            
            depth_ax_list = np.array(depth_ax_list, dtype = float)
            depth_ax_start_of_peak_list = np.array(depth_ax_start_of_peak_list, dtype = float)
            flaw_peak_ratio_list = np.array(flaw_peak_ratio_list, dtype = float)
            flaw_peak_value_list = np.array(flaw_peak_value_list, dtype = float)

            amp_ratio_flaw.append(flaw_peak_ratio_list)  # for flagging
            amplitudes_flaw.append(flaw_peak_value_list)  # for flagging
            dtw_range_flaw.append(np.array(flaw_range_list) >= np.array(surface_range_fix_list) + config['DTW_RANGE_DELTA'])
            dtw_range_flaw_difference.append(np.array(flaw_range_list) - np.array(surface_range_fix_list))
            
            # ignore circ loc inside a single frame where flaw range exceeds some threshold
            depth_nan_idx = np.where(np.array(flaw_range_list) >= np.array(surface_range_fix_list) + config['DTW_RANGE_DELTA'])[0]

            depth_nan_idx = depth_nan_idx[flaw_peak_value_list[depth_nan_idx] <= config['FLAW_AMP_THRSH_MAX']]
            
            if config['AMP_RATIO']:
                depth_nan_idx = depth_nan_idx[flaw_peak_ratio_list[depth_nan_idx] < config['FLAW_AMP_RATIO_THRSH']]
            flaw_peak_value_list[depth_nan_idx] = 0

            depth_nan_idx = np.where(flaw_peak_value_list <= config['FLAW_AMP_THRSH_MAX'])[0]
            
            if config['AMP_RATIO']:
                depth_nan_idx = depth_nan_idx[flaw_peak_ratio_list[depth_nan_idx] < config['FLAW_AMP_RATIO_THRSH']]
            depth_ax_list[depth_nan_idx] = np.nan
            depth_ax_start_of_peak_list[depth_nan_idx] = np.nan

        depth_flaw.append(depth_ax_list)
        depth_flaw_up.append(depth_ax_up_list)
        depth_flaw_bottom.append(depth_ax_bottom_list)
        depth_flaw_left.append(depth_ax_left_list)
        depth_flaw_right.append(depth_ax_right_list)
        depth_flaw_start_of_peak.append(depth_ax_start_of_peak_list)
        surface_circ_flaw.append(surface_circ_list)
        surface_ax_flaw.append(surface_ax_list)
    
        all_frame_peaks.append(frame_peaks)
        all_frame_bonus_peaks.append(frame_bonus_peaks)
        all_frame_surface_peaks.append(frame_surface_peaks)
        all_curvature_corrections.append(frame_curvature_corrections)

        # if nans for all circ locs/single frame
        try:
            flaw_circ = np.nanargmax(np.array(depth_ax_list))
        except:
            flaw_circ = np.nan

        flaw_circ += pred_circ_start
        
    depth_flaw_arr = np.array(depth_flaw)
    depth_flaw_arr_unfiltered = np.array(depth_flaw_unfiltered)
    depth_flaw_arr_up = np.array(depth_flaw_up)
    depth_flaw_arr_bottom = np.array(depth_flaw_bottom)
    depth_flaw_arr_left = np.array(depth_flaw_left)
    depth_flaw_arr_right = np.array(depth_flaw_right)
    depth_flaw_start_of_peak_arr = np.array(depth_flaw_start_of_peak)
    surface_circ_flaw_arr = np.array(surface_circ_flaw)
    surface_ax_flaw_arr = np.array(surface_ax_flaw)
    
    assert depth_flaw_arr.shape == (pred_ax_end - pred_ax_start + 1, pred_circ_end - pred_circ_start + 1)


    total_missing_depths_percent = nan_percentage(depth_flaw_arr)
    if not np.isnan(depth_flaw_arr).all():
        pred_depth_flaw = np.nanmax(depth_flaw_arr)
        (flaw_ax, flaw_circ) = unravel_index(np.nanargmax(depth_flaw_arr), depth_flaw_arr.shape)
    else:
        return pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,0,0,total_missing_depths_percent,0,\
            0,0, 0, 0,0,0,\
            depth_flaw_arr, depth_flaw_arr_unfiltered, dtw_range_flaw,dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw,all_frame_peaks,\
            APC_cors,all_curvature_corrections,all_frame_bonus_peaks,all_frame_surface_peaks,depth_flaw_start_of_peak_arr

    # plot depth profile
    try:
        file_logger.info('Plotting depth profile')
        plot_depth_profile_single_flaw(row, depth_flaw_arr, 'pitch_catch', SAVE_ROOT)
    except:
        save_path = os.path.join(SAVE_ROOT, 'Depth Sizing Profile', FILENAME)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"{OUTAGE_NUMBER} {CHANNEL}.txt"), 'w') as file:
            error_msg = f'Not able to plot depth profile for Indication: {row["Indication"]}!!!\nError: '
            file_logger.error(error_msg, exc_info=True)
            file.write(error_msg)
    
    pred_depth_flaw_up = depth_flaw_arr_up[flaw_ax, flaw_circ]
    pred_depth_flaw_bottom = depth_flaw_arr_bottom[flaw_ax, flaw_circ]
    
    pred_depth_flaw_left = depth_flaw_arr_left[flaw_ax, flaw_circ]
    pred_depth_flaw_right = depth_flaw_arr_right[flaw_ax, flaw_circ]
    
    surface_circ_selected = surface_circ_flaw_arr[flaw_ax, flaw_circ]
    surface_ax_selected = surface_ax_flaw_arr[flaw_ax, flaw_circ] + 1

    if save_files:
        save_path = os.path.join(save_location, 'Depth Sizing', 'Pred depth whole b-scan', str(row['Indication']))
        os.makedirs(os.path.join(save_path), exist_ok=True)
        pd.DataFrame(depth_flaw_arr).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_whole_bscan.xlsx'))

    return pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,flaw_ax,flaw_circ,total_missing_depths_percent,pred_depth_flaw,\
            pred_depth_flaw_up,pred_depth_flaw_bottom, pred_depth_flaw_left, pred_depth_flaw_right,surface_circ_selected,surface_ax_selected,\
            depth_flaw_arr, depth_flaw_arr_unfiltered, dtw_range_flaw,dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw,all_frame_peaks,\
            APC_cors,all_curvature_corrections,all_frame_bonus_peaks,all_frame_surface_peaks,depth_flaw_start_of_peak_arr

def pred_CC_depth(config,df,APC_lags_whole_scan,APC_cor_g4,APC_lags_g4,APC_wavegroups,apc_data_arr,
                  run_name, save_location='',save_name='flaw',save_files=False,plotting=False, out_root='auto-analysis-results'):
    """
    Calculates the depth of a CC flaw using B-scans

    Parameters:
    ----------
        config: a dictionairy containing constants required to run the code. can be found in sizing_config.yaml under 'CC'
        df: a dataframe of every flaw to size the depth of, must all be within the same B-scan file. required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'ID' (file id used to separate flaws by which file they come from)
            -'Filename' (name of B-scan file, used to track the flaw locations)
            Optional columns:
            -'depth' (the actual reported depth of the flaw for comparison, used on plots and to calculate error for the stats dataframe)
        APC_lags_whole_scan:  The lags from performing cross-correlations on the entire A-scan
        APC_cor_g4: The correlations from performing cross-correlations on the focus wave group
        APC_lags_g4: The lags from performing cross-correlations on the focus wave group
        APC_wavegroups: The locations of the focus wave groups fro each A-scan in the B-scans
        apc_data_arr: B-scan array to perform depth sizing with
        save_location: directory to save files in
        save_name: name to use when saving files
        save_files: whether to save any files
        plotting: whether to create any plots

    Returns:
        results_df: results of depth calculation with columns:
            -'ID' (file id used to separate flaws by which file they come from)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'pred_depth' (predicted depth in mm)
            -'flaw_ax' (frame location of deepest location,starts with 1)
            -'flaw_circ(circ location of deepest location,starts with 0)
            -'surface_ax' (surface frame location,starts with 1)
            -'surface_circ(surface circ location,starts with 0)
            -'Flaw Maximum Amplitude' (the maximum amplitude of the peaks used to measure the flaw)
            -'Flipped A scans' (depth is measured by flipping the A-scans, right now only doing this if normal measurement fails)

        missing_df: dataframe of flaws for which a depth could not be calculated, has columns:
            -'ID' (file id used to separate flaws by which file they come from)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Error' (error that caused the indication to not be measured)

        stats_df: dataframe with statistics columns for each indication, also has columns:
            -'ID' (file id used to separate flaws by which file they come from)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 

    """

    global SAVE_ROOT
    global FILENAME
    global CHANNEL
    global OUTAGE_NUMBER
    file_logger.info(f"Initialising the depth sizing.")

    # store flaw locations
    flaw_dic_ax, flaw_dic_circ = flaw_loc_file(df)
    save_path_CC = os.path.join(save_location, 'depth sizing', 'CC')
    if plotting and save_files:
        os.makedirs(save_path_CC, exist_ok=True)
        with open(os.path.join(save_path_CC, "all_constants.txt"), 'w') as f:
            for k, v in config.items():
                f.writelines(k + ' : ' + str(v) + '\n')

    FILENAME_FULL = df['Filename'].unique()[0]
    OUTAGE_NUMBER = df['Outage Number'].unique()[0]
    CHANNEL = df['Channel'].unique()[0]
    FLAW_TYPE =  df['flaw_type'].unique()[0]
    SAVE_ROOT = os.path.join(out_root, run_name)
    FILENAME = FILENAME_FULL.split('.')[0]

    (pred_depth_all, pred_depth_up_all, pred_depth_bottom_all, pred_depth_left_all, 
    pred_depth_right_all, act_depth_all, flaw_ax_all, flaw_circ_all, 
    surface_circ_all, surface_ax_all, ID, indication, surface_not_found_list, 
    flipped_ascans_list, chatter_all, flagging_issues_lst, flagging_criteria_lst, 
    deepest_flaw_bonus_peaks_lst, bonus_peak_info_out_lst) = ([] for _ in range(19))
    missing = pd.DataFrame([])

    #Missing depths lists
    for index, row in df.iterrows():
        flipped_ascans = False
        save_path_CC_ind = os.path.join(save_path_CC, str(row['Indication']))
          
        if plotting and save_files:
            os.makedirs(save_path_CC_ind, exist_ok=True)

        try:
            # First try without flipping ascans
            file_logger.info(f"Calculating flaw depth using positive features")
            pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,flaw_ax,flaw_circ,total_missing_depths_percent,pred_depth_flaw,\
            pred_depth_flaw_up,pred_depth_flaw_bottom, pred_depth_flaw_left, pred_depth_flaw_right,surface_circ_selected,surface_ax_selected,\
            depth_flaw_arr, depth_flaw_arr_unfiltered,dtw_range_flaw,dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw, all_frame_peaks,\
            APC_cors,all_curvature_corrections,all_frame_bonus_peaks, all_frame_surface_peaks,depth_flaw_start_of_peak_arr = pred_depth_single_flaw(config,row,False,APC_lags_whole_scan,APC_cor_g4,APC_lags_g4,APC_wavegroups,apc_data_arr, flaw_dic_ax, flaw_dic_circ,plotting,save_path_CC_ind, save_files=True,save_location=save_location)
            # Check the condition and raise an exception if needed
            if pred_depth_flaw == 0:
                raise ValueError
        except Exception as e:
            try:
                file_logger.info(f"Calculating flaw depth using negative features")
                save_path_flipped =os.path.join(save_path_CC_ind,'flipped')
                if plotting and save_files:
                    os.makedirs(save_path_flipped, exist_ok=True)
                # If not flipped failed, try flipping the A-scans
                pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,flaw_ax,flaw_circ,total_missing_depths_percent,pred_depth_flaw,\
                pred_depth_flaw_up,pred_depth_flaw_bottom, pred_depth_flaw_left, pred_depth_flaw_right,surface_circ_selected,surface_ax_selected,\
                depth_flaw_arr, depth_flaw_arr_unfiltered,dtw_range_flaw,dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw, all_frame_peaks,\
                APC_cors,all_curvature_corrections,all_frame_bonus_peaks,all_frame_surface_peaks,depth_flaw_start_of_peak_arr = pred_depth_single_flaw(config,row,True,APC_lags_whole_scan,APC_cor_g4,APC_lags_g4,APC_wavegroups,apc_data_arr,flaw_dic_ax,flaw_dic_circ,plotting,save_path_flipped,save_location=save_location)
                flipped_ascans = True
                if pred_depth_flaw == 0:
                    raise ValueError("PC Depth cannot be calculated for: " + str(row['Indication']) + ", Continuing run")
            except Exception as e2:
                    #If both fail skip this flaw
                    file_logger.error(f'Error in calculating depth for flaw instance: {row["Indication"]}: {e}. Continuing to next instance.', exc_info=True)
                    if str(e)==str(e2):
                        traceback.print_exc()
                        missing=missing.append(pd.DataFrame.from_dict({'ID':[row['Filename']],'Indication': [row['Indication']],'Error': ['Both_'+str(e)]}))
                    else:
                        traceback.print_exc()
                        missing=missing.append(pd.DataFrame.from_dict({'ID':[row['Filename']],'Indication':[row['Indication']],'Error': ['Error1_'+str(e)+'_Error2_'+str(e2)]}))
                    print(e)
                    print(e2)
                    continue

        try:
            if total_missing_depths_percent > config['MISSING_DEPTHS_PERCENT_FLIP_THRESH']*100 and not flipped_ascans:
            # if too many depths were not measured, try measuring again
                save_path_flipped =os.path.join(save_path_CC,'flipped')
                if plotting and save_files:
                    os.makedirs(save_path_flipped, exist_ok=True)
                flaw_results_flipped = pred_depth_single_flaw(config,row,True,APC_lags_whole_scan,APC_cor_g4,APC_lags_g4,APC_wavegroups,apc_data_arr,flaw_dic_ax,flaw_dic_circ,plotting,save_path_flipped,save_location=save_location)
                if flaw_results_flipped[7]>pred_depth_flaw:
                    #if a greater depth was measured replace results with flipped results
                    pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,flaw_ax,flaw_circ,total_missing_depths_percent,pred_depth_flaw,\
                    pred_depth_flaw_up,pred_depth_flaw_bottom, pred_depth_flaw_left, pred_depth_flaw_right,surface_circ_selected,surface_ax_selected,\
                    depth_flaw_arr, depth_flaw_arr_unfiltered,dtw_range_flaw,dtw_range_flaw_difference, amp_ratio_flaw, amplitudes_flaw, all_frame_peaks,\
                    APC_cors,all_curvature_corrections,all_frame_bonus_peaks,all_frame_surface_peaks,depth_flaw_start_of_peak_arr = flaw_results_flipped
                    flipped_ascans = True
            # Measuring Chatter
            try:
                file_logger.info('Measuring Chatter')
                chatter_all.append(measure_chatter(FILENAME_FULL,APC_lags_whole_scan,APC_lags_g4, flaw_dic_ax, row['Ax Start'], row['Ax End'], row['Ro Start'], row['Ro End'] ,config, 'CC'))
            except:
                file_logger.error('Chatter amp cannot be calculated')
                chatter_all.append('Chatter amp cannot be calculated')

            # -- Flagging issues --
            flagging_issue = flagging_issues(pred_circ_start, pred_circ_end, pred_ax_start, pred_ax_end, flaw_circ, 
                                            flaw_ax, config, depth_flaw_arr_unfiltered, pred_depth_flaw, depth_flaw_arr,
                                            depth_flaw_start_of_peak_arr)
            flagging_issues_lst.append(flagging_issue)
            # -- Criteria Flagging --
            flagging_criteria = criteria_flagging(pred_circ_start,pred_circ_end,pred_ax_start,
                                                pred_ax_end,flaw_circ, flaw_ax,APC_cors,all_frame_peaks,
                                                all_curvature_corrections, dtw_range_flaw,
                                                dtw_range_flaw_difference, amplitudes_flaw, amp_ratio_flaw, config)
            flagging_criteria_lst.append(flagging_criteria)
            # -- deepest_flaw_bonus_peaks --
            flaw_bonus_peaks_info = all_frame_bonus_peaks
            flaw_bonus_peaks_info_circ= swap_axes(flaw_bonus_peaks_info)
            deepest_flaw_bonus_peaks= deepest_flaw_bonus_peak(all_frame_bonus_peaks,find_deepest_location,
                                                                                                                  config,pred_ax_start)
            deepest_flaw_bonus_peaks_lst.append(deepest_flaw_bonus_peaks)
            # -- bonus_peak_info --
            bonus_peak_info_out = bonus_peak_info(flaw_circ,flaw_ax,pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,flaw_bonus_peaks_info_circ,config)
            bonus_peak_info_out_lst.append(bonus_peak_info_out)

            try:
                found_location,frame_deepest,circ_deepest,longest_sequence=find_deepest_location(flaw_bonus_peaks_info,config['EXTRA_PEAK_SAME_CIRC_THRESH'],config['EXTRA_PEAK_SEQUENCE_THRESH'])
            except Exception:
                found_location,frame_deepest,circ_deepest,longest_sequence = False,0,0,0
           
            try:
                if all_frame_bonus_peaks[frame_deepest][circ_deepest]['height'] >= config['EXTRA_PEAK_AMP_THRESH']  and all_frame_bonus_peaks[frame_deepest][circ_deepest]['prominence']>=config['EXTRA_PEAK_PROMINENCE_THRESH'] and found_location and all_frame_bonus_peaks[frame_deepest][circ_deepest]['width'] <= config['EXTRA_PEAK_WIDTH_MAX'] :
                    lag_bonus = all_frame_surface_peaks[frame_deepest][circ_deepest]['index'] - all_frame_bonus_peaks[frame_deepest][circ_deepest]['index']
                    bonus_depth_circ = max(np.round(lag_bonus * config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'], 4), 0) + all_curvature_corrections[frame_deepest][circ_deepest]
                else:
                    lag_bonus = all_frame_surface_peaks[frame_deepest][circ_deepest]['index'] - all_frame_bonus_peaks[frame_deepest][circ_deepest]['index']
                    bonus_depth_circ = max(np.round(lag_bonus * config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'], 4), 0) + all_curvature_corrections[frame_deepest][circ_deepest]
                    bonus_depth_circ = (-1) *bonus_depth_circ
            except Exception:
                traceback.print_exc()

            #depth info 
            flaw_ax += pred_ax_start + 1
            flaw_circ += pred_circ_start
                
            flaw_ax_all.append(flaw_ax)
            flaw_circ_all.append(flaw_circ)
            pred_depth_all.append(pred_depth_flaw)
            pred_depth_up_all.append(pred_depth_flaw_up)
            pred_depth_bottom_all.append(pred_depth_flaw_bottom)
            pred_depth_left_all.append(pred_depth_flaw_left)
            pred_depth_right_all.append(pred_depth_flaw_right)
            if 'depth' in row.keys():
                act_depth_all.append(row['depth'])
            else:
                act_depth_all.append(np.nan)
            surface_circ_all.append(surface_circ_selected)
            surface_ax_all.append(surface_ax_selected)
            ID.append(row['Filename'])
            indication.append(row['Indication'])
            flipped_ascans_list.append(flipped_ascans)

        except Exception as e:
            missing=missing.append(pd.DataFrame.from_dict({'ID':[row['Filename']],'Indication': [row['Indication']],'Error': [str(e)]}))
            print(e)
            continue

    flagging_issue = {key: [d[key] for d in flagging_issues_lst] for key in flagging_issues_lst[0]}        
    flagging_criteria = {key: [d[key][0] for d in flagging_criteria_lst] for key in flagging_criteria_lst[0]}
    flagging_deepest_flaw_bonus_peak = {key: [d[key][0] for d in deepest_flaw_bonus_peaks_lst] for key in deepest_flaw_bonus_peaks_lst[0]}
    flagging_bonus_peak_info = {key: [d[key][0] for d in bonus_peak_info_out_lst] for key in bonus_peak_info_out_lst[0]}
    
    flagging_information = {**flagging_issue, **flagging_criteria, **flagging_deepest_flaw_bonus_peak, **flagging_bonus_peak_info}        
    error = np.array(pred_depth_all) - np.array(act_depth_all)

    # -- Preparing result dataframes --
    df_surface_not_found = pd.DataFrame({'error': surface_not_found_list})

    results_df = pd.DataFrame({'ID': ID, 'Indication' : indication,
                               'flaw_depth' : pred_depth_all,'flaw_ax' : flaw_ax_all,
                               'flaw_circ' : flaw_circ_all, 'surface_ax' : surface_ax_all,
                               'flaw_feature_amp': flagging_information['flaw_loc_amplitude'],
                               'surface_circ' : surface_circ_all, 'Flipped A scans': flipped_ascans_list,
                               'flaw_max_amp':flagging_information['max_amplitude'],
                               'chatter_amplitude':chatter_all, 'probe_depth': 'APC', 'note_2' : np.nan})
    

    stats_df_dict = {'ID': ID, 'Indication' : indication, 'act_depth' : act_depth_all,
                               'pred_depth' : pred_depth_all, 'pred_depth_up' : pred_depth_up_all,
                               'pred_depth_bottom' : pred_depth_bottom_all,
                               'pred_depth_left' : pred_depth_left_all,
                               'pred_depth_right' : pred_depth_right_all,
                               'error' : error, 'flaw_ax' : flaw_ax_all,
                               'flaw_circ' : flaw_circ_all, 'surface_ax' : surface_ax_all,
                               'surface_circ' : surface_circ_all,
                               'Flipped A scans': flipped_ascans_list
                            }
    stats_df_dict.update(flagging_information)
    stats_df = pd.DataFrame(stats_df_dict)
    stats_df['pred_depth_up'][stats_df['pred_depth_up'].isna()] = stats_df['pred_depth_bottom'][stats_df['pred_depth_up'].isna()]
    stats_df['pred_depth_bottom'][stats_df['pred_depth_bottom'].isna()] = stats_df['pred_depth_up'][stats_df['pred_depth_bottom'].isna()]
    assert not stats_df['pred_depth_up'].isna().any() and not stats_df['pred_depth_bottom'].isna().any()
    stats_df['depth_avg'] = (stats_df['pred_depth_up'] + stats_df['pred_depth_bottom']) / 2
    missing_df=missing.reset_index(drop=True)

    if config['FLAGGING']:
        try:
            flagged_cases = flag_cases(stats_df,config)
            results_df['flag_high_error'] = flagged_cases
            flag_high_error_true = results_df['flag_high_error'] == True
            results_df.loc[flag_high_error_true, 'flaw_depth'] = results_df.loc[flag_high_error_true, 'flaw_depth'].astype('str') + ' (2)'
            results_df.loc[flag_high_error_true, 'note_2'] = 'Measured depth is flagged, need to be reviewed'
        except:
            y = 'Not able to flag'
            file_logger.error(y, exc_info=True)
            results_df['flag_high_error'] = y

    if save_files:
        file_logger.info('Saving intermediate files.')
        save_path = os.path.join(save_location, 'Depth Sizing')
        os.makedirs(os.path.join(save_path), exist_ok=True)
        stats_df.to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_pc_stats.xlsx'), index = False)  
        results_df.to_excel(os.path.join(save_path,  f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_pc_results.xlsx'), index = False)    
        missing_df.to_excel(os.path.join(save_path,  f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_pc_missing.xlsx'), index = False)
        
    return results_df,missing_df,stats_df
