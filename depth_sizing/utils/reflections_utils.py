import numpy as np
from scipy import signal


def rmv_fwg(a_scan, start, stop):
    
    """This function removes the fwg of a_scan
    Args:
        a_scan(np.array): a-scan
        start(int): start of the fwg
        stop(int): : stop of the fwg
            
    Returns:
        a_scan(np.array): a-scan after removing the fwg
           
    """
    
    a_scan_fwg_rmv = a_scan.copy()
    a_scan_fwg_rmv[: stop] = np.median(a_scan)
    
    a_scan_fwg = a_scan[start : stop]
    
    return a_scan_fwg_rmv, a_scan_fwg

def check_multi_reflections(flaw_a_scan_fwg_rmv, amp_thresh):
    
    """This function checks the number of multi features in the reflections

    Args:
        flaw_a_scan_fwg_rmv(np.array): Flaw a-scan after removing the fwg
        amp_thresh(int): Min refelction feature amp threshold
        
    Returns:
        peak_indices: indices of the multiple features found in the reflection
        
    """
    
    peak_indices, _ = signal.find_peaks(flaw_a_scan_fwg_rmv, width = 2, height = amp_thresh)
    
    return peak_indices

def find_consecutive_true(lst, consecutive_count):
    
    """This function removes the Trues from list if these are not consecutive with the given consecutive_count

    Args:
        lst(list): list of True and False
        consecutive_count(int): min no of consecutive True requires in a list to not to be removed
        
    Returns:
        result(list): list after removing non-consecutive Trues
        
    Example:
        Input: lst = [T, F, F, T, T, T, T, F, F], consecutive_count = 3
    Output: result = [F, F, F, T, T, T, T, F, F]
                
       
    """
    
    # use convolution to find consecutive True
    arr = np.array(lst)
    ones_list = [1] * consecutive_count
    mask = np.convolve(arr, ones_list, mode='same') >= consecutive_count
    result = np.zeros_like(arr, dtype=bool)
    result[np.where(mask)] = True
    result[np.where(np.roll(mask, 1))] = True
    result[np.where(np.roll(mask, -1))] = True
    
    return result

def check_reflections_main_fwg(flaw_a_scan, surface_a_scan, surface_fwg_start, surface_fwg_stop, ref_std, config, ITERATION, OPEN_SURFACE_AMP_DIFF_THRSH):
    
    """This function checks the reflections based on the fwg

    Args:
        flaw_a_scan(np.array): Flaw a-scan
        surface_a_scan(np.array): Surface a-scan
        surface_fwg_start(int): Surface FWG start position
        surface_fwg_stop(int): Surface FWG stop position
        ref_std(float): STD of Reflections of reflection window in Flaw a-scan
        
    Returns:
        reflections(bool): True, if flaw a-scan has reflections
        open_surface(bool): True: if flaw a-scan is open to surface
        fwg_amp_diff(int): Amp difference between the surface fwg and flaw fwg
       
    """
    
    reflections = False
    open_surface = False
    
    # threshold to find reflections
    amp_diff_thesh = config[f'REFLECTION_FWG_AMP_DIFF_THRSH_{ITERATION}'] # min amp diff between flaw and surface fwg
    ratio_thresh = config['REFLECTION_FWG_AMP_RATIO_THRSH'] # min amp ratio between flaw and surface fwg
    buffer = 20
    
    
    # if ref std is high lower your thresh
    if ref_std > 1.6:
        amp_diff_thesh-=15
        ratio_thresh+=0.06
    
    # FWG
    surface_a_scan_fwg = surface_a_scan[surface_fwg_start - buffer : surface_fwg_stop + buffer]
    flaw_a_scan_fwg = flaw_a_scan[surface_fwg_start - buffer : surface_fwg_stop + buffer]
    
    # find amp diff between the FWGs
    # find max to max/ min to min difference between the Surface and Flaw FWGs
    surface_max_amp = np.max(surface_a_scan_fwg)
    flaw_max_amp = np.max(flaw_a_scan_fwg)
    surface_min_amp = np.min(surface_a_scan_fwg)
    flaw_min_amp = np.min(flaw_a_scan_fwg)
    flaw_amp_pos_diff = surface_max_amp - flaw_max_amp
    flaw_amp_neg_diff = flaw_min_amp - surface_min_amp
    fwg_amp_diff = max(flaw_amp_pos_diff, flaw_amp_neg_diff)
    fwg_amp_ratio = max(flaw_max_amp / surface_max_amp,  surface_min_amp / (flaw_min_amp+1))
    
    # if the diff is very large, could be open to surface
    if fwg_amp_diff >= OPEN_SURFACE_AMP_DIFF_THRSH:
        open_surface = True

    # use ratio as reflection condition if the surface do not have saturated peak
    if surface_max_amp < 220 and surface_min_amp > 30:
        reflection_condn = fwg_amp_ratio < ratio_thresh
        # print('condn: ratio', fwg_amp_ratio)
    else:
        reflection_condn = fwg_amp_diff >= amp_diff_thesh
        # print('condn: amp', fwg_amp_diff, amp_thesh, reflection_condn)

    # for reflections = True, there should be high var/std and either the amp diff or amp ratio should be high
    if reflection_condn and ref_std >= config[f'REFLECTION_MIN_STD_{ITERATION}']:
        reflections = True
        
    # print(np.max(flaw_a_scan_fwg), np.max(surface_a_scan_fwg), np.min(flaw_a_scan_fwg), np.min(surface_a_scan_fwg), fwg_amp_diff, fwg_amp_ratio, ref_std)
    return reflections, open_surface, fwg_amp_diff

def ignore_reflections_open_surface(reflections_fwg_frame_list, open_surface_frame_list, open_surface_near_circ_loc):
    
    """This function makes reflection_fwg flag as False for open surface locations

    Args:
        reflections_fwg_frame_list(list): True, if circ loc has reflections, shape: (1 * no of circ locs)
        open_surface_frame_list(list): True, if circ loc is open to surface, shape: (1 * no of circ locs)
        open_surface_near_circ_loc(int): no nearby circ loc for which reflections will be ignored along with open surface
        
    Returns:
        reflections_fwg_frame_list(list): reflection list, after making reflections = False for open surface and nearby locations
       
    """
    
    # find indices of open surface
    open_surface_idxs = np.where(open_surface_frame_list)[0]
    
    # if open surface are there
    if len(open_surface_idxs) != 0:
        open_surface_min_idx, open_surface_max_idx = np.min(open_surface_idxs), np.max(open_surface_idxs)
        
        # if open surface is at bottom edge, ignore nearby locations on the bottom edge
        if open_surface_max_idx + open_surface_near_circ_loc + 1 >= len(reflections_fwg_frame_list):
            circ_locs_greater = np.arange(open_surface_max_idx + 1, len(reflections_fwg_frame_list))
        else:
            circ_locs_greater = np.arange(open_surface_max_idx + 1, open_surface_max_idx + open_surface_near_circ_loc + 1)
        
        # if open surface is at top edge, ignore nearby locations on the top edge 
        if open_surface_min_idx - open_surface_near_circ_loc < 0:
            circ_locs_smaller = np.arange(0, open_surface_min_idx)
        else:
            circ_locs_smaller = np.arange(open_surface_min_idx - open_surface_near_circ_loc, open_surface_min_idx)
        
        # make reflections = False for the open surface and nearby locations
        open_surface_idxs = np.concatenate([circ_locs_smaller, open_surface_idxs, circ_locs_greater])
        reflections_fwg_frame_list[open_surface_idxs] = False
        
    return reflections_fwg_frame_list


def open_surface_middle_depth(open_surface_idxs, depth_ax_list, depth_ax_fwg_rmv_list):
    
    """This function helps to select the depth near to middle of open surface

    Args:
        open_surface_idxs(list): indexes of open to surface circ locs inside a single frame
        depth_ax_list(list): contains depth for all circ locs inside a single frame
        depth_ax_fwg_rmv_list(list): contains depth after renoving fwg for all circ locs inside a single frame
        
    Returns:
        depth_ax_list: contains depth for all circ locs inside a single frame, all other circ locs depth will be zero except those near to open to surface
        depth_ax_fwg_rmv_list: contains depth for all circ locs inside a single frame, all other circ locs depth will be zero except those near to open to surface
       
    """
    
    depth_ax_list_temp = np.zeros_like(depth_ax_list) * np.nan
    depth_ax_fwg_rmv_list_temp = np.zeros_like(depth_ax_list) * np.nan
    
    # find middle of the open surface
    open_surface_mid_idx = np.median(open_surface_idxs).astype('int')
    i = open_surface_mid_idx
    
    # make depth zero for  all circ except the locs near to open to surface
    depth_ax_list_temp[i-1:i+2] = depth_ax_list[i-1:i+2]
    depth_ax_fwg_rmv_list_temp[i-1:i+2] = depth_ax_fwg_rmv_list[i-1:i+2]
    depth_ax_list = depth_ax_list_temp
    depth_ax_fwg_rmv_list = depth_ax_fwg_rmv_list_temp
    
    return depth_ax_list, depth_ax_fwg_rmv_list