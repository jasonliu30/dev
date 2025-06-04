import numpy as np
from scipy import signal
from depth_sizing.utils.flattening import Smoothlags_remove_flaws


def measure_chatter(Filename, lags, backup_lags, flaw_dic_ax,
                     pred_ax_start ,pred_ax_end ,pred_ro_start , pred_ro_end ,config, flaw_type):
        
    """This function measures the chatter for NB probes. 

    Args:
        Filename (name of B-scan file, used to track the flaw locations)
        lags(array): The lags from performing cross-correlations on the entire A-scan
        backup_lags(array): The lags from performing cross-correlations on the focus wave group
        flaw_dic_ax(dic): contains circ locations of all flaws inside a frame
        pred_ax_start: Flaw axial start position
        pred_ax_end: Flaw axial end position
        pred_ro_start: Flaw rotary start position
        pred_ro_end: Flaw rotary end position
        config(dictionary): a dictionary containing constants required to run the code. can be found in sizing_config.yaml under 'config'
            
    Returns:
            chatter_mm: measured chatter in mm
    """
    all_peaks = []
    for frame in range(pred_ax_start,pred_ax_end+1):
      
        used_lags = [lags[frame],backup_lags[frame]][np.argmin([np.sum(np.abs(np.gradient(lags[frame]))),
                                                                np.sum(np.abs(np.gradient(backup_lags[frame])))])]
        indications = np.zeros(len(used_lags),dtype=bool)
        if frame in flaw_dic_ax[Filename].keys():
            locations = flaw_dic_ax[Filename][frame][0]
            for i in locations:
                indications[i] = True
                        
        pressure_tube_surface = Smoothlags_remove_flaws(used_lags, config['SD_LIMIT_CHATTER_MEASUREMENT'],
                                                         config['MAXITER_CHATTER_MEASUREMENT'],indications,
                                                           config['SG_ORD_CHATTER_MEASUREMENT'],
                                                             config['SG_FLEN_CHATTER_MEASUREMENT'],
                                                               config['OUTLIER_GRADIENT_CHATTER_MEASUREMENT'])[0]
        flat_lags = used_lags - pressure_tube_surface
        acceptable_range = np.arange(max(0,pred_ro_start - config['CIRC_RANGE_CHATTER_MEASUREMENT']),
                                     min(pred_ro_end+config['CIRC_RANGE_CHATTER_MEASUREMENT'],len(used_lags)) ) # which circ locations are used

        #POSITIVE PEAKS
        a,_ = signal.find_peaks(flat_lags,height = config['MIN_HEIGHT_CHATTER_MEASUREMENT'],
                                width = config['MIN_WIDTH_CHATTER_MEASUREMENT'])
        for i in a:
            if i in acceptable_range and i not in locations:
                all_peaks.append(flat_lags[i])
            
        #NEGATIVE PEAKS
        a,_ = signal.find_peaks(flat_lags*-1,height = config['MIN_HEIGHT_CHATTER_MEASUREMENT'],
                                width = config['MIN_WIDTH_CHATTER_MEASUREMENT'])
        for i in a:
            if i in acceptable_range and i not in locations:
                all_peaks.append((flat_lags*-1)[i])

    if flaw_type == 'CC':
        lag = np.nanmean(all_peaks)
        time_depth_constant = config['MICRO_SEC_DEPTH_SHEAR']
    else:
        lag = np.nanmax(all_peaks)    
        time_depth_constant = config['MICRO_SEC_depth_NB']

    chatter_mm = max(np.round(lag * config['UNIT_MICRO_SEC'] * time_depth_constant, 4), 0)

    return chatter_mm