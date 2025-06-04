"""Find the depth and flaw-info, such as flaw bottom location, max amp, etc., for FBBPF flaws"""

import os
import numpy as np
import pandas as pd
from numpy import unravel_index
from depth_sizing.utils.chatter_magnitude import measure_chatter
from depth_sizing.utils.depth_sizing_flag import flag_cases
from depth_sizing.utils.plot_utils import plot_depth_profile_single_flaw
from depth_sizing.utils.utils import (flaw_loc_file, remove_abrupt_change,
                                    find_lag, find_wavegroups_in_frame,
                                    find_surface, units_depth, find_signal_invert,
                                    find_surface_feature, get_flattened_lags)
from utils.logger_init import create_dual_loggers

# create loggers
_, file_logger  = create_dual_loggers()

def find_depth_single_circ(
        nb_data_arr, row, ax, circ, pressure_tube_location,
        flattened_lags, flaw_dic_ax, flaw_dic_circ):
    
    """This function calculates depth for a single circ loc (circ) inside a frame or ax loc (ax)

    Args:
        nb_data_arr: b-scan array
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
            
        ax(int): Axial location starts from 0
        circ(int): Circ location starts from 0
        pressure_tube_location: 
        flattened_lags(np.ndarray) : The flattened lags from performing cross-correlations on the entire A-scan
        flaw_dic_ax(dictionary): contains circ locations of all flaws inside a frame
        flaw_dic_circ(dictionary): contains ax locations of all flaws for a circ loc
            
    Returns:
        flaw_peak_value(int): Flaw feature/peak value
        depth_circ(float): predicted depth from one circ loc
        cost(float): DTW cost
    """
    
    curvature_correction = 0
    file_logger.debug(f"Calculating depth for a single circ loc ({circ}) inside a frame ({ax+1})")

    # find surface signal
    file_logger.debug("Selecting surface signal...")
    surface_a_scan, surface_wave_locations, surface_circ_loc = find_surface(nb_data_arr, flaw_dic_ax,
                                                                            flattened_lags, ax,
                                                                            circ, row, None, config)
 
    # flaw a-scan at ax, circ
    flaw_a_scan = nb_data_arr[ax, circ, :]
    # find flaw a-scan main fwg

    flaw_wave_locations = find_wavegroups_in_frame(flaw_a_scan,
                                                   None,
                                                  number_of_wavegroups = 1,
                                                  wavegroup_width = config['WAVEGROUP_WIDTH'],
                                                  amplitude_threshold = 140,
                                                  multi_reflection = False)
    
    # if no main flaw fwg found
    if len(flaw_wave_locations) == 0:
        
        flaw_peak_value = np.nan
        depth_circ = np.nan
        cost_normalized = np.nan
        cost_focus_normalized = np.nan
        cost = [cost_normalized, cost_focus_normalized]
        
        return flaw_peak_value, depth_circ , cost
    
    # start and stop position of fwg
    surface_fwg_start, surface_fwg_stop = surface_wave_locations[0][0], surface_wave_locations[0][1]
    flaw_fwg_start, flaw_fwg_stop = flaw_wave_locations[0][0], flaw_wave_locations[0][1]
    
    # surface and flaw fwg
    surface_a_scan_fwg = surface_a_scan[ surface_fwg_start : surface_fwg_stop ]
    flaw_a_scan_fwg = flaw_a_scan[flaw_fwg_start: flaw_fwg_stop]


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
        cost_focus_normalized = np.nan
        cost = [cost_normalized, cost_focus_normalized]

        return flaw_peak_value, depth_circ , cost
    
    # curvature correction
    if config['FLATTENNING']:
        curvature_correction = pressure_tube_location[circ] - pressure_tube_location[surface_circ_loc] 
    
    # find the position difference between flaw and surface feature and convert to mm
    depth_circ = units_depth(surface_peak_posn, flaw_peak_posn, curvature_correction,config)
  
    # DTW cost
    cost = [cost_normalized, cost_focus_normalized]
    
    return flaw_peak_value, depth_circ , cost
    

def find_depth_single_frame(
        nb_data_arr, row, ax, pressure_tube_location,
        flattened_lags, flaw_dic_ax, flaw_dic_circ):
    
    """This function calculates depth for a single frame (ax) or for all circ locs inside a single frame

    Args:
        nb_data_arr(array): b-scan array
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
            
        ax(int): Axial location starts from 0
        pressure_tube_location(list): 
        flattened_lags(np.ndarray) : The flattened lags from performing cross-correlations on the entire A-scan
        flaw_dic_ax(dictionary): contains circ locations of all flaws inside a frame
        flaw_dic_circ(dictionary): contains ax locations of all flaws for a circ loc
            
    Returns:
        depth_ax(list): list of arrays, contains the predicted depth for a single frame/ax and all circ locs 
        cost(list): contains the dtw cost for a single frame/ax and all circ locs 
        flaw_peak_value_list: List of flaw feature values in a frame
       
    """
    
    depth_ax_list = [] # stores depth for every circ loc in specific frame
    flaw_peak_value_list = [] # stores flaw peak values for every circ loc in specific frame
    
    # stores dtw cost for every circ loc in specific frame
    cost_normalized_list = []
    cost_focus_normalized_list = []

    # calculates depth for every circ location
    for circ in range(pred_circ_start, pred_circ_end):
        
        flaw_peak_value,  depth_circ,\
        [cost_normalized, cost_focus_normalized] = find_depth_single_circ(nb_data_arr,
                                                                           row, ax, circ,
                                                                           pressure_tube_location,
                                                                           flattened_lags,
                                                                           flaw_dic_ax,
                                                                           flaw_dic_circ)

        flaw_peak_value_list.append(flaw_peak_value)
        
        depth_ax_list.append(depth_circ)      

        cost_normalized_list.append(cost_normalized)
        cost_focus_normalized_list.append(cost_focus_normalized)
        
    cost = [cost_normalized_list,cost_focus_normalized_list]

    return depth_ax_list, flaw_peak_value_list, cost

def find_depth_single_flaw(
        row, flagging, lags, backup_lags, nb_data_arr,
        flaw_dic_ax, flaw_dic_circ, signal_invert = False):
    
    """This function calculates depth for a single flaw instances in row (single row of df)

    Args:
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        
        flagging(dic): contains features required for flagging model
        lags(array): The lags from performing cross-correlations on the entire A-scan
        backup_lags(array): The lags from performing cross-correlations on the focus wave group
        nb_data_arr(array): b-scan array
        flaw_dic_ax(dic): contains circ locations of all flaws inside a frame
        flaw_dic_circ(dic): contains ax locations of all flaws for a circ loc
        signal_invert(bool): if True, also calculates depth using inverted a-scans
            
    Returns:
            pred_depth_flaw(float): Max flaw depth from flaw bottom
            flaw_loc(list): Ax and cir loc of flaw bottom
            flaw_peak_amp(int): Selected flaw feature amp value
            cost(float): DTW cost
            flagging(dic): contains features required for flagging model
    """
    
    global depth_flaw
    global cost_flaw
    global pred_ax_start
    global pred_ax_end
    global pred_circ_start
    global pred_circ_end
    global save_path
    
    # file_logger.info(f"Calculating depth for Indication: {row['Indication']}, Filename: {row['Filename']}")
    depth_flaw = [] # pred depth for every location, shape: no of frames * no circ locs
    cost_flaw = [] # DTW cost for every location, shape: no of frames * no circ locs
    flaw_peak_value_list_all = [] # # Flaw peak amp values: no of frames * no circ locs
   
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
        
        if PLOTTING_GLOBAL:
            frame = ax + 1
            os.makedirs(os.path.join(save_path, f"Frame-{frame}"), exist_ok=True)

        # smooth the frame of lags to get the general trend
        if config['FLATTENNING']:
            pressure_tube_location, flattened_lags = get_flattened_lags(ax, lags,
                                                                        backup_lags,
                                                                        flaw_dic_ax,
                                                                        row, config)
        else:
            pressure_tube_location = np.zeros(len(lags[ax]))
            flattened_lags = np.zeros(len(lags[ax]))
        
        # calculate depth for a frame
        depth_ax_list, flaw_peak_value_list, cost = find_depth_single_frame(nb_data_arr,
                                                                            row, ax,
                                                                            pressure_tube_location,
                                                                            flattened_lags,
                                                                            flaw_dic_ax,
                                                                            flaw_dic_circ)
        [cost_normalized_list,cost_focus_normalized_list] = cost
        
        # ignore sudden changes in the depth
        depth_ax_list = remove_abrupt_change(depth_ax_list, max_diff = 0.5)
        
        # predicted depth list
        depth_flaw.append(depth_ax_list)
        cost_flaw.append(cost_normalized_list)
        flaw_peak_value_list_all.append(flaw_peak_value_list)
       
    # convert to arrays
    depth_flaw_arr = np.array(depth_flaw)
    flaw_peak_value_arr = np.array(flaw_peak_value_list_all)
    assert depth_flaw_arr.shape == (pred_ax_end - pred_ax_start, pred_circ_end - pred_circ_start)
    
    # predicted depth is max of all
    pred_depth_flaw = np.nanmax(depth_flaw_arr)
    
    # Flaw deepest depth location
    try:
        (flaw_ax, flaw_circ) = unravel_index(np.nanargmax(depth_flaw_arr), depth_flaw_arr.shape)
    except:
        (flaw_ax, flaw_circ) = (np.nan, np.nan)
        
    # flaw peak amp
    try:
        flaw_peak_amp = flaw_peak_value_arr[flaw_ax, flaw_circ]
    except:
        flaw_peak_amp = np.nan
        
    # ax & circ loc w.r.t PT start
    flaw_ax += pred_ax_start + 1
    flaw_circ += pred_circ_start
    flaw_loc = [flaw_ax, flaw_circ]
    
    # max depth per frame
    
    # features for flagging
    flagging['Min Amplitude'].append(np.nanmin(flaw_peak_value_arr))
    flagging['Max Amplitude'].append(np.nanmax(flaw_peak_value_arr))
    flagging['Percent_within_one_hundredth'].append(sum(depth_flaw_arr.flatten()>=pred_depth_flaw-0.01)/len(depth_flaw_arr.flatten()))
    flagging['Number_within_one_hundredth'].append(sum(depth_flaw_arr.flatten()>=pred_depth_flaw-0.01))

    flagging['Percent_within_one_tenth'].append(sum(depth_flaw_arr.flatten()>=pred_depth_flaw-0.1)/len(depth_flaw_arr.flatten()))
    flagging['Number_within_one_tenth'].append(sum(depth_flaw_arr.flatten()>=pred_depth_flaw-0.1))
    
    # DTW cost
    cost_flaw = np.array(cost_flaw)
    try:
        cost = cost_flaw[flaw_ax - (pred_ax_start + 1), flaw_circ - (pred_circ_start)]
    except:
        cost = np.nan
    
    # plot depth profile
    # try:
    #     file_logger.info('Plotting depth profile')
    #     plot_depth_profile_single_flaw(row, depth_flaw_arr, 'normal_beam', SAVE_ROOT)
    # except:
    #     save_path = os.path.join(SAVE_ROOT, 'Depth Sizing Profile', FILENAME)
    #     os.makedirs(save_path, exist_ok=True)
    #     with open(os.path.join(save_path, f"{OUTAGE_NUMBER} {CHANNEL}.txt"), 'w') as file:
    #         error_msg = f'Not able to plot depth profile for Indication: {row["Indication"]}!!!\nError: '
    #         file_logger.error(error_msg, exc_info=True)
    #         file.write(error_msg)

    return pred_depth_flaw, flaw_peak_amp, flaw_loc, cost, flagging, depth_flaw_arr

    
def pred_FBBPF_depth(
        conf, df, lags_whole, lags_G4, nb_data_arr,
        plotting, save_files, run_name,save_location, out_root):
    """
    This is the main function to calculate the depth of a FBBPF flaw using B-scans

    Parameters:

    config: a dictionairy containing constants required to run the code. can be found in sizing_config.yaml under 'CC'

    df: a dataframe of every flaw to size the depth of, must all be within the same B-scan file. required columns are:
        -'Ax Start' (frame number of start of flaw starting with 0)
        -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
        -'Ro Start' (rotary location of start of flaw starting with 0)
        - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
        -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
        -'ID' (file id used to separate flaws by which file they come from)
        -'Filename' (name of B-scan file, used to track the flaw locations)

    lags_whole:  The lags from performing cross-correlations on the entire A-scan

    lags_G4:  The lags from performing cross-correlations on the focus wave group

    nb_data_arr: B-scan array to perform depth sizing with

    plotting: whether to create any plots

    save_files: whether to save any files
    
    run_name(str): Name of the run

    save_location(str): Location to save the results

    Returns:
    results_df: results of depth calculation with columns:
        -'ID' (file id used to separate flaws by which file they come from)
        -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
        -'pred_depth' (predicted depth in mm)
        -'flaw_ax' (frame location of deepest location,starts with 1)
        -'flaw_circ(circ location of deepest location,starts with 0)
        -'surface_ax' (surface frame location,starts with 1)
        -'surface_circ (surface circ location,starts with 0)
        -'Flaw Maximum Amplitude' (the maximum amplitude of the peaks used to measure the flaw)
        -'Chatter Magnitude (mm)' (the estimated chatter magnitude around the indication)

    missing_df_df: dataframe of flaws for which a depth could not be calculated, has columns:
        -'ID' (file id used to separate flaws by which file they come from)
        -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
        -'Error' (error that caused the indication to not be measured)

    stats_df: dataframe with statistics columns for each indication, also has columns:
        -'ID' (file id used to separate flaws by which file they come from)
        -'Indication' (Number of indication, as a string ) for example 'Ind 3' 

    """
    
    global PLOTTING_GLOBAL
    # global SAVE_ROOT
    global FILENAME
    global CHANNEL
    global OUTAGE_NUMBER
    global config
    global FLAW_TYPE
    
    file_logger.info(f"Initialising the depth sizing.")
    # SAVE_ROOT = os.path.join(out_root, run_name)
    OUTAGE_NUMBER = df['Outage Number'].unique()[0]
    CHANNEL = df['Channel'].unique()[0]
    FILENAME_FULL = df['Filename'].unique()[0]
    FILENAME = FILENAME_FULL.split('.')[0]
    config = conf
    PLOTTING_GLOBAL = plotting
    FLAW_TYPE =  df['flaw_type'].unique()[0]
    
    # Find the loc of the flaw w.r.t ax & circ loc
    flaw_dic_ax, flaw_dic_circ = flaw_loc_file(df,nb_data_arr)
    
    # lags
    lags = lags_whole
    backup_lags = lags_G4
    
    # flagging dic is used for in flagging high error cases
    flagging = {'Indication':[],'Max Amplitude': [], 'Min Amplitude': [],
                'Percent_within_one_hundredth':[],
                'Number_within_one_hundredth':[],'Percent_within_one_tenth':[],
                'Number_within_one_tenth':[],'BM':[],'cost_norm':[],
                'deepest_location_percent_ax':[], 'deepest_location_percent_ro':[]
                }
    if config['INVERT']:
        flagging_invert = flagging.copy()
        
    # dataframe to store the output
    results_df = pd.DataFrame({})
    
    # predicted depth
    pred_depth_all = []
    pred_depth_invert_all = []
    
    # Flaw location
    flaw_ax_all = []
    flaw_circ_all = []
    flaw_ax_invert_all = []
    flaw_circ_invert_all = []
    
    # flaw feature amp
    flaw_peak_amp_all = []
    flaw_peak_amp_invert_all = []
    
    # DTW cost
    cost_normalized_all = []
    
    # store any indication for which is not calculated
    missing_df = {'Indication': [],'Error':[]}
    
    # indications
    indication = []
    
    # chatter measurements
    chatter_all = []
    
    # loop through all flaw instances
    for index, row in df.iterrows():    
    
        try:
            # try to calculate the depth, if depth could not be calculated, use missing_df_df dataframe to add ID and Indication
            
            # measure chatter
            try:
                file_logger.info('Measuring Chatter')
                chatter = measure_chatter(FILENAME_FULL,lags,backup_lags, flaw_dic_ax,
                                            row['Ax Start'], row['Ax End'],
                                            row['Ro Start'], row['Ro End'] ,config, FLAW_TYPE)
                chatter_all.append(chatter)
            except:
                file_logger.error('Chatter amp cannot be calculated')
                chatter_all.append('Chatter amp cannot be calculated')
            
            # calculate depth using original a-scans for a single flaw
            file_logger.info(f"Calculating flaw depth using positive features")
            pred_depth_flaw, flaw_peak_amp, [flaw_ax, flaw_circ],\
            cost, flagging, depth_flaw_arr = find_depth_single_flaw(row, flagging, lags,
                                                        backup_lags, nb_data_arr,
                                                        flaw_dic_ax, flaw_dic_circ,
                                                        signal_invert = False)
            
            # flaw location
            flaw_ax_all.append(flaw_ax)
            flaw_circ_all.append(flaw_circ)
            
            # flaw feature amp
            flaw_peak_amp_all.append(flaw_peak_amp)
            
            # predicted depth
            pred_depth_all.append(pred_depth_flaw)
            
            indication.append(row['Indication'])
    
            # required features in flagging
            flagging['Indication'].append(row['Indication'])
            length = row['Ax End']- row['Ax Start']
            width = row['Ro End']- row['Ro Start']
            flagging['deepest_location_percent_ax'].append(100*((flaw_ax-row['Ax Start'])/length))
            flagging['deepest_location_percent_ro'].append(100*((flaw_circ-row['Ro Start'])/width))
            if config['INVERT']:
                flagging_invert['Indication'].append(row['Indication']) 
            
            # DTW cost
            cost_normalized_all.append(cost)
                
            # calculate depth using original a-scans for a single flaw, using inverted a-scan
            if config['INVERT']:
                file_logger.info(f"Calculating flaw depth using negative features")
                pred_depth_flaw_invert, flaw_peak_amp_invert,\
                [flaw_ax_invert, flaw_circ_invert],cost_invert,\
                flagging_invert, depth_flaw_arr_invert = find_depth_single_flaw(row,
                                                            flagging_invert,
                                                            lags,
                                                            backup_lags,
                                                            nb_data_arr,
                                                            flaw_dic_ax,
                                                            flaw_dic_circ,
                                                            signal_invert = True)
                flaw_ax_invert_all.append(flaw_ax_invert)
                flaw_circ_invert_all.append(flaw_circ_invert)
                pred_depth_invert_all.append(pred_depth_flaw_invert)
                
                # flaw feature amp
                flaw_peak_amp_invert_all.append(flaw_peak_amp_invert)
                
                assert len(flaw_ax_all) == len(flaw_ax_invert_all)
                
                # save depth for whole b-scan
            if save_files:
                file_logger.info('Saving intermediate files.')
                save_path = os.path.join(save_location, 'Depth Sizing', 'Pred depth whole b-scan', str(row['Indication']))
                os.makedirs(os.path.join(save_path), exist_ok=True)
                pd.DataFrame(pd.DataFrame(depth_flaw_arr)).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_whole_bscan.xlsx'))
         
        # if there is any error while calculating depth, add info to missing_df_df and continue
        except Exception as e:
            file_logger.error(f'Error in calculating depth for flaw instance: {row["Indication"]}: {e}. Continuing to next instance.', exc_info=True)
            missing_df['Indication'].append(row['Indication'])
            missing_df['Error'].append(str(e))
            continue
        
    missing_df = pd.DataFrame(missing_df)
    
    # save the results in dataframe
    results_df['Indication'] = indication
    results_df['flaw_ax'] = flaw_ax_all
    results_df['flaw_circ'] = flaw_circ_all
    results_df['flaw_feature_amp'] = flaw_peak_amp_all
    results_df['pred_depth'] = pred_depth_all
    results_df['flaw_max_amp'] = flagging['Max Amplitude']
    results_df['chatter_amplitude'] = chatter_all
    results_df['note_2'] = np.nan
    results_df['probe_depth'] = 'NB1'

    if config['INVERT']:
        results_df['flaw_ax_invert'] = flaw_ax_invert_all
        results_df['flaw_circ_invert'] = flaw_circ_invert_all
        results_df['pred_depth_invert'] = pred_depth_invert_all
        results_df['flaw_feature_amp_invert'] = flaw_peak_amp_invert_all
    
    # flag high error cases
    if config['FLAGGING']:
        flagging['cost_norm'] = cost_normalized_all
        try:
            flagging['BM'] = ['BM' in i for i in df['flaw_type']]
        except:
            flagging['BM'] = ['BM' in i for i in df['Indication']]
        
        try:
            flag_df = pd.DataFrame(flagging)
            flagged_cases = flag_cases(flag_df, config)
            results_df['flag_high_error'] = flagged_cases
            
            flag_high_error_true = results_df['flag_high_error'] == True
            results_df.loc[flag_high_error_true, 'pred_depth'] = results_df.loc[flag_high_error_true, 'pred_depth'].astype('str') + ' (2)'
            results_df.loc[flag_high_error_true, 'note_2'] = 'Measured depth is flagged, need to be reviewed'
                
        except:
            y = 'Not able to flag'
            file_logger.error(y, exc_info=True)
            flag_df['flag_high_error'] = y 
            results_df['flag_high_error'] = y
        
    results_df.rename(columns = {'pred_depth' : 'flaw_depth'}, inplace = True)

    # save all results
    if save_files:
        save_path = os.path.join(save_location, 'Depth Sizing')
        os.makedirs(save_path, exist_ok=True)
        flag_df.to_excel(os.path.join( save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_nb_stats.xlsx'), index = False)  
        results_df.to_excel(os.path.join(save_path,  f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_nb_results.xlsx'), index = False)    
        missing_df.to_excel(os.path.join(save_path,  f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_nb_missing_df.xlsx'), index = False)
        
    return results_df, missing_df, flag_df
