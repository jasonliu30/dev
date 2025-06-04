"""Find the depth and flaw-info, such as flaw bottom location, max amp, etc., for Axial & Circ Scrapes"""

import os
import numpy as np
import pandas as pd
from depth_sizing.utils.flaw_information import Flaw_Information
from depth_sizing.utils.chatter_magnitude import measure_chatter
from depth_sizing.utils.utils import flaw_loc_file
from inferencing_script import InferenceCharOutput
from depth_sizing.utils.plot_utils import plot_depth_profile_single_flaw
from utils.logger_init import create_dual_loggers

# create loggers
_, file_logger  = create_dual_loggers()

def remove_outlier(x, method='percentile_2'):
    """
    Removes outliers from an array.

    Args:
        x: numpy array
        method: the outlier method to be used

    Returns:
        x (with outliers removed)
    """
    if method.lower() == 'none' or method is None:
        return x

    elif "percentile" in method:
        x = x.astype('float')
        p = int(method.split("_")[1])
        lower_limit = np.percentile(x, p)
        upper_limit = np.percentile(x, 100 - p)
        x[x < lower_limit] = np.nan
        x[x > upper_limit] = np.nan

    elif method == "iqr":
        q75, q25 = np.percentile(x, [75, 25])
        iqr = q75 - q25
        lower_limit = q25 - iqr * 1.5
        upper_limit = q75 + iqr * 1.5
        x[x < lower_limit] = np.nan
        x[x > upper_limit] = np.nan

    elif method == "std":
        x_std = np.std(x)
        x_mean = np.mean(x)
        lower_limit = x_mean - 3 * x_std
        upper_limit = x_mean + 3 * x_std
        x[x < lower_limit] = np.nan
        x[x > upper_limit] = np.nan

    else:
        raise LookupError(f"Unknown Outlier Method: {method}")

    return x


def find_tof_bottom(cscan_flaw, cscan_type, depth_method, outlier_method):
    """
    Calculates the ToF of the bottom of the flaw.

    Args:
        cscan_flaw: CScan, cropped to only contain the flaw of interest.
        cscan_type: String indicating the type of CScan
        depth_method: String indicating the depth method to use to calculate the peak ToF
        outlier_method: String indicating the outlier method to use

    Returns:
        tof_bottom - peak ToF, indicating the bottom of the flaw
        tof_bottom_indices: location of the bottom of the flaw
    """
    
    if depth_method == 'global':
        if 'APC' in cscan_type or 'CPC' in cscan_type:
            tof_bottom = np.nanmin(cscan_flaw)
            tof_bottom_indices = np.unravel_index(np.nanargmin(cscan_flaw), cscan_flaw.shape)
        else:
            tof_bottom = np.nanmax(cscan_flaw)
            tof_bottom_indices = np.unravel_index(np.nanargmax(cscan_flaw), cscan_flaw.shape)

        return tof_bottom, tof_bottom_indices

    elif depth_method == 'top_n_median':

        cscan_flaw_1d = cscan_flaw.ravel()

        top_n_percent = cscan_flaw_1d <= np.nanpercentile(cscan_flaw_1d, 1) if ('APC' in cscan_type or 'CPC' in cscan_type) else cscan_flaw_1d >= np.nanpercentile(cscan_flaw_1d, 99)
        flaw_top = cscan_flaw_1d[top_n_percent]
        tof_bottom = np.median(flaw_top)

        if len(cscan_flaw_1d[top_n_percent]) % 2 != 0:
            tof_bottom_indices = [np.where(cscan_flaw == tof_bottom)[0][0], np.where(cscan_flaw == tof_bottom)[1][0]]
        else:
            idx_0 = np.argsort(flaw_top)[len(flaw_top) // 2]
            idx_1 = np.argsort(flaw_top)[len(flaw_top) // 2 - 1]
            tof_bottom_indices = [[np.where(cscan_flaw == flaw_top[idx_0])[0][0], np.where(cscan_flaw == flaw_top[idx_0])[1][0]],
                                  [np.where(cscan_flaw == flaw_top[idx_1])[0][0], np.where(cscan_flaw == flaw_top[idx_1])[1][0]]]
            
        return tof_bottom, tof_bottom_indices


def calculate_depth_ax_circ(depth, depth_method):
    """
    Picks the best depth measurement from an array of depth measurements.

    Args:
        depth: numpy array of calculated depths in one axis
        depth_method: depth method to use to calculate the overall depth

    Returns:
        depth: predicted depth
        loc_selected: flaw bottom location
    """
    if depth_method == 'global':
        pred_depth = np.nanmax(depth)
        loc_selected = [np.unravel_index(np.nanargmax(np.array(depth)), np.array(depth).shape)[0]]

    elif depth_method == 'top_n_median':
        top_n_perc = depth >= np.nanpercentile(depth, 99)
        depth_top = depth[top_n_perc]
        pred_depth = np.median(depth_top)

        if len(depth[top_n_perc]) % 2 != 0:
            loc_selected = [np.where(depth == pred_depth)[0][0]]
        else:
            idx = np.argsort(depth_top)[len(depth_top) // 2]
            idx_1 = np.argsort(depth_top)[len(depth_top) // 2 - 1]
            loc_selected = [np.where(depth == depth_top[idx])[0][0], np.where(depth == depth_top[idx_1])[0][0]]

    else:
        raise LookupError(f"Unknown depth method: {depth_method}. Expected: 'global' or 'top_n_median'")

    return pred_depth, loc_selected


def find_tof_surface(surface, outlier_method):
    """
    Calculates the ToF of the surface surrounding a flaw.

    Args:
        surface: numpy array indicating the surface around the flaw
        outlier_method: outlier method to use

    Returns:
        tof_surface_median: ToF of the surface
    """
    # Filter out any zero-values. These are values covered up by the mask.
    surface = surface[surface != 0]

    surface = remove_outlier(surface, outlier_method)

    tof_surface_median = np.nanmedian(surface.ravel())

    return tof_surface_median


def tof_depth(tof_bottom, tof_surface, cscan_type, tof_constant_shear, tof_constant_nb):
    """
    Converts the TOF to the depth.

    Args:
        tof_bottom: The ToF to the bottom of the flaw (peak depth)
        tof_surface: The ToF of the area surrounding the flaw
        cscan_type: name of the cscan used
        tof_constant_shear: constant to use to calculate the depth for APC and CPC probes
        tof_constant_nb: constant to use to calculate the depth for NB_ probes

    Returns:
        depth
    """
    if 'APC' in cscan_type or 'CPC' in cscan_type:
        depth = (tof_surface - tof_bottom) * tof_constant_shear
    else:
        depth = (tof_bottom - tof_surface) * tof_constant_nb
    return depth

def mask_flaw(cscan, y1, x1, y2, x2, buffer_size, boundary_size, crop=True, extra_buffer=0):
    """
    Masks the flaw by settings the ToF of the flaw area to zero, to allow the code to find the
    ToF of the surrounding surface.

    Args:
        cscan: CScan containing the flaw
        y1: flaw axial starting location
        x1: flaw rotary starting location
        y2: flaw axial ending location
        x2: flaw rotary ending location
        buffer_size: number of pixels around the flaw to include in the mask
        boundary_size: number of pixels around the buffered mask to include in the surface
        crop (default: True): whether to crop the cscan to the buffered area
        extra_buffer (default: 0): Amount of extra buffer to add to the ending locations

    Returns:
        cscan_surface
    """
    # Create a mask and set the values inside the mask to zero
    y1_mask = max(y1 - buffer_size, 0)
    y2_mask = min(y2 + buffer_size + extra_buffer, cscan.shape[0])
    x1_mask = max(x1 - buffer_size, 0)
    x2_mask = min(x2 + buffer_size + extra_buffer, cscan.shape[1])

    cscan_surface = cscan.copy()
    cscan_surface[y1_mask: y2_mask, x1_mask: x2_mask] = 0

    # Calculate buffer ROI
    y1_surface = max(y1 - buffer_size - boundary_size, 0)
    y2_surface = min(y2 + buffer_size + boundary_size + extra_buffer, cscan.shape[0])
    x1_surface = max(x1 - buffer_size - boundary_size, 0)
    x2_surface = min(x2 + buffer_size + boundary_size + extra_buffer, cscan.shape[1])

    # Crop if asked
    if crop:
        cscan_surface = cscan_surface[y1_surface: y2_surface, x1_surface: x2_surface]
    return cscan_surface

def find_depth_1d(row, cscan, flaw_config, general_config, FLAW_TYPE, save_files,save_location, cscan_type):
    """
    Calculates the flaw depth by iterating through the flaw row-by-row or column-by-column

    Args:
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        cscan: numpy array of the chosen CScan
        flaw_config: object containing configuration values
        FLAW_TYPE: Type of the flaw
        save_files: to save intermediate results
        save_location: location to save the results

    Returns:
        depth - Depth of the flaw
    """
    # Calculate flaw ROI and crop
    pred_ax_start, pred_ax_end = row['Ax Start'], row['Ax End']
    pred_circ_start, pred_circ_end = row['Ro Start'], row['Ro End']

    cscan_flaw = cscan.copy()

    cscan_surface = mask_flaw(cscan, pred_ax_start, pred_circ_start, pred_ax_end,\
                               pred_circ_end, general_config['buffer_size'],\
                                  general_config['boundary_size'], False, 0)

    # Set up the iterables based on the flaw type.
    if FLAW_TYPE == "Circ_Scrape":
        
        # Circular = Iterate along the columns, crop along the rows
        flaw_iter = range(pred_circ_start, pred_circ_end)
        flaw_extent = range(pred_ax_start, pred_ax_end)
        surface_extent = range(max(pred_ax_start - general_config['buffer_size'] - general_config['boundary_size'], 0),
                               min(pred_ax_end + general_config['buffer_size'] + general_config['boundary_size'], cscan.shape[0]))
        axis = 1
    else:

        # Axial = Iterate along the rows, crop along the columns
        flaw_iter = range(pred_ax_start, pred_ax_end)
        flaw_extent = range(pred_circ_start, pred_circ_end)
        surface_extent = range(max(pred_circ_start - general_config['buffer_size'] - general_config['boundary_size'], 0),
                               min(pred_circ_end + general_config['buffer_size'] + general_config['boundary_size'], cscan.shape[1]))
        axis = 0
    # Initialize empty arrays
    tof_bottoms = np.zeros(len(flaw_iter))
    bottom_indices = []
    tof_surfaces = np.zeros(len(flaw_iter))
    depths = np.zeros(len(flaw_iter))
    depth_map = np.zeros([len(flaw_iter), len(flaw_extent)])

    # Iterate through the flaw ax/circ
    j = 0
    for i in flaw_iter:
        # Get a 1d slice of the flaw and of the surrounding surface area
        flaw_1d = cscan_flaw.take(i, axis=axis)[flaw_extent]
        surface_1d = cscan_surface.take(i, axis=axis)[surface_extent]

        # Calculate ToF and the index of the peak ToF
        tof_bottom, tof_bottom_indices = find_tof_bottom(flaw_1d, cscan_type, 'global', flaw_config['outlier_method'])
        tof_bottoms[j] = tof_bottom
        bottom_indices.append(tof_bottom_indices)

        # Calculate the ToF of the surface
        tof_surface = find_tof_surface(surface_1d, flaw_config['outlier_method'])
        tof_surfaces[j] = tof_surface

        # Calculate depth
        depth = tof_depth(tof_bottom, tof_surface, cscan_type, general_config['tof_constant_shear'], general_config['tof_constant_nb'])
        depths[j] = depth

        # Remove outliers, and then create a depth map.
        depth_1d = tof_depth(flaw_1d, tof_surface, cscan_type, general_config['tof_constant_shear'], general_config['tof_constant_nb'])
        depth_map[j] = depth_1d

        j += 1

    bottom_indices = np.asarray(bottom_indices)

    # indices the bottom of the flaw
    if FLAW_TYPE == 'Circ_Scrape':
        plot_depth_profile_single_flaw(row, depth_map.transpose(), 'normal_beam', SAVE_ROOT)
        depth, circular_loc = calculate_depth_ax_circ(depths, flaw_config['depth_method'])
        axial_loc = bottom_indices[circular_loc].ravel()
    else:
        plot_depth_profile_single_flaw(row, depth_map, 'normal_beam', SAVE_ROOT)
        depth, axial_loc = calculate_depth_ax_circ(depths, flaw_config['depth_method'])
        circular_loc = bottom_indices[axial_loc].ravel()

    axial_loc = np.asarray(axial_loc) + pred_ax_start
    circular_loc = np.asarray(circular_loc) + pred_circ_start
    flaw_location = np.stack((circular_loc, axial_loc), axis=-1)

    # save depth for whole b-scan
    if save_files:
        save_path = os.path.join(save_location, 'Depth Sizing', 'Pred depth whole b-scan', str(row['Indication']))
        os.makedirs(os.path.join(save_path), exist_ok=True)
        pd.DataFrame(pd.DataFrame(depth_map)).to_excel(os.path.join(save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_pred_depth_whole_bscan.xlsx'))

    return depth, flaw_location


def find_depth_single_flaw(row, cscan, flaw_config, general_config, FLAW_TYPE, save_files,save_location, cscan_type):
    """
    Calculates the depth of a flaw.

    Args:
        row(dataseries): dataseries containing a single flaw to size the depth of, required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        cscan: 2D numpy array representing a specific cscan
        flaw_config: object containing configuration values
        FLAW_TYPE: Type of the flaw
        save_files: to save intermediate results
        save_location: location to save the results

    Returns:
        flaw_information
    """
    depth, flaw_location = find_depth_1d(row, cscan, flaw_config, general_config, FLAW_TYPE, save_files,save_location, cscan_type)

    flaw_information = Flaw_Information(FLAW_TYPE,
                                        row['Ax Start'],
                                        row['Ax End'] - row['Ax Start'],
                                        row['Ro Start'],
                                        row['Ro End'] - row['Ro Start'],
                                        depth,
                                        flaw_location)
    return flaw_information

def pred_ax_circ_scrape_depth(df, cscans, bscans, lags, backup_lags, config, FLAW_TYPE, run_name, save_files,save_location, cscan_type, out_root):

    """
    Main function to calculate the depth of the axial/circumferential scrapes

    Args:
        df(dataframe): dataframe of every flaw to size the depth of, must all be within the same B-scan file. required columns are:
            -'Ax Start' (frame number of start of flaw starting with 0)
            -'Ax End' (frame number of end of flaw starting with 0, flaw includes this frame)
            -'Ro Start' (rotary location of start of flaw starting with 0)
            - 'Ro End' (rotary location of end of flaw starting with 0, flaw includes this location)
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'Filename' (name of B-scan file, used to track the flaw locations)
        cscan: 2D numpy array representing a specific cscan
        b_scans(list): List of NB1 and NB2 b-scans
        lags : List of the NB1 and NB2 lags from performing cross-correlations on the entire A-scan
        backup_lags : List of the NB1 and NB2 lags from performing cross-correlations on the g4 of A-scan
        config: object containing configuration values
        run_name(str): Name of the run
        save_files: to save intermediate results
        save_location: location to save the results

    Returns:
        results_df(dataframe): results of depth calculation with columns:
            -'Indication' (Number of indication, as a string ) for example 'Ind 3' 
            -'pred_depth_nb1_nb2' (predicted depth in mm)
            -'flaw_ax' (frame location of deepest location,starts with 1)
            -'flaw_circ(circ location of deepest location,starts with 0)
            -'surface_ax' (surface frame location,starts with 1)
            -'surface_circ(surface circ location,starts with 0)
            -'Flaw Maximum Amplitude' (the maximum amplitude of the peaks used to measure the flaw)
    """

    global SAVE_ROOT
    global CHANNEL
    global OUTAGE_NUMBER
    global FILENAME

    # dataframe to store the output
    results_df = pd.DataFrame({})
    
    # predicted depth
    pred_depth_all = []
    
    # Flaw location
    flaw_ax_all = []
    flaw_circ_all = []
    
    # flaw feature amp
    flaw_peak_amp_all = []

    # flaw max amp
    flaw_max_amp_all = []
    
    # indications
    indication = []
    
    # chatter measurements
    chatter_all = []

    # probe used for depth calculations
    probe_depth_all = []

    # initialize dataframe to save results
    results_df = pd.DataFrame(columns = ['Indication', 'flaw_depth', 'flaw_max_amp', 'flaw_feature_amp',
                                          'flag_high_error', 'note_2', 'chatter_amplitude', 'probe_depth',
                                          'flaw_ax', 'flaw_circ'])
    file_logger.info(f"Initialising the depth sizing using C-scan.")

    # Flaw info
    FILENAME_FULL = df['Filename'].unique()[0]
    FILENAME = FILENAME_FULL.split('.')[0]
    OUTAGE_NUMBER = df['Outage Number'].unique()[0]
    CHANNEL = df['Channel'].unique()[0]
    SAVE_ROOT = os.path.join(out_root, run_name)    

    # loop through all flaw instances
    for index, row in df.iterrows():

        # file_logger.info(f"Calculating depth for Indication: {row['Indication']}, Filename: {row['Filename']}")

        # get c-scan
        general_config = config['Default'].__dict__
        flaw_config = config[FLAW_TYPE].__dict__
        [probe, cscan_reprsnt] = flaw_config['cscan'].split('_')
        probe_to_id = general_config['probe_to_id'].__dict__
        cscan = cscans[probe_to_id[cscan_type]][cscan_reprsnt]
        # get b-scan
        bscan = bscans[cscan_type].data

        # try to predict depth, otherwise predict nan
        #try:
        results = find_depth_single_flaw(row, cscan, flaw_config, general_config, FLAW_TYPE, save_files,save_location, cscan_type)
        pred_depth = results.depth_mm
        [flaw_circ, flaw_ax] = results.flaw_location[0]
        # except Exception as e:
        #     file_logger.error(f'Error in calculating depth for flaw instance: {row["Indication"]}: {e}. Continuing to next instance.', exc_info=True)
        #     pred_depth = np.nan
        #     [flaw_circ, flaw_ax] = [np.nan, np.nan]

        # measure chatter
        try:
            file_logger.info('Measuring Chatter')
            # Find the loc of the flaw w.r.t ax & circ loc
            flaw_dic_ax, _ = flaw_loc_file(df, bscan)
            # predicted flaw extent coordinates
            pred_ax_start, pred_ax_end = row['Ax Start'], row['Ax End']
            pred_circ_start, pred_circ_end = row['Ro Start'], row['Ro End']
            chatter = measure_chatter(FILENAME_FULL, lags, backup_lags, flaw_dic_ax,
                                pred_ax_start, pred_ax_end, pred_circ_start, pred_circ_end,\
                                flaw_config, FLAW_TYPE)
        except:
            file_logger.error('Chatter amp cannot be calculated')
            chatter = 'N/A'

        # Flaw max and feature amp
        try:
            flaw_max_amp = np.nanmax(bscan)
            flaw_peak_amp = np.nanmax(bscan[flaw_ax - pred_ax_start, flaw_circ - pred_circ_start])
        except:
            flaw_max_amp = np.nan
            flaw_peak_amp = np.nan
        
        # Save the results
        indication.append(row['Indication'])
        pred_depth_all.append(pred_depth)
        flaw_ax_all.append(flaw_ax)
        flaw_circ_all.append(flaw_circ)
        probe_depth_all.append(probe)
        flaw_max_amp_all.append(flaw_max_amp)
        flaw_peak_amp_all.append(flaw_peak_amp)
        chatter_all.append(chatter)

    # Results dataframe
    results_df['Indication'] = indication
    results_df['probe_depth'] = probe_depth_all
    results_df['flaw_ax'] = flaw_ax_all
    results_df['flaw_circ'] = flaw_circ_all
    results_df['flaw_feature_amp'] = flaw_peak_amp_all
    results_df['flaw_max_amp'] = flaw_max_amp_all
    results_df['chatter_amplitude'] = chatter_all
    results_df['flaw_depth'] = pred_depth_all
    results_df['flag_high_error'] = ''
    results_df['note_2'] = ''

    # save all results
    if save_files:
        file_logger.info('Saving intermediate files.')
        stats_df = pd.merge(left = df, right = results_df, how = 'left', on = ['Indication']) 
        save_path = os.path.join(save_location, 'Depth Sizing')
        os.makedirs(save_path, exist_ok=True)
        results_df.to_excel(os.path.join( save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_results.xlsx'), index = False) 
        stats_df.to_excel(os.path.join( save_path, f'{OUTAGE_NUMBER}_{CHANNEL}_{FLAW_TYPE}_stats.xlsx'), index = False) 

    return results_df
