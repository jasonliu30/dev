import numpy as np
from numpy import unravel_index
from depth_sizing.utils.CC_depth_utils import swap_axes, find_largest_consecutive_changes

def criteria_flagging(pred_circ_start,pred_circ_end,pred_ax_start,
                      pred_ax_end,flaw_circ, flaw_ax,APC_cors,all_frame_peaks,
                      all_curvature_corrections, dtw_range_flaw,
                      dtw_range_flaw_difference, amplitudes_flaw, amp_ratio_flaw, config):
    
    flaw_loc_DTW_range = []
    flaw_loc_amplitude = []
    flaw_loc_ampratio = []
    average_correlation = []
    max_correlation = []
    frame_largest_width_change = []
    largest_circ_width_change = []
    flaw_frame_width_change = []
    flaw_circ_width_change = []
    average_width = []
    min_width = []
    max_width = []
    flaw_loc_width = []
    frame_largest_prominence_change = []
    flaw_frame_prominence_change = []
    flaw_circ_prominence_change = []
    flaw_loc_prominence = []
    flaw_loc_curvature = []
    max_amplitude = []
    Frame_width_largest_percent_increase_and_decrease = []
    Circ_width_largest_percent_increase_and_decrease = []
    Circ_width_largest_percent_increase_and_decrease_width_increase = []
    Circ_width_largest_percent_increase = []
    Circ_width_largest_percent_decrease = []
    Percent_circ_width_percent_passed_thresh = []
    Number_circ_width_percent_passed_thresh = []
    Average_circ_width_largest_percent_increase_and_decrease = []
    Percent_circ_width_high_correlation_with_neighbors = []
    Number_circ_width_high_correlation_with_neighbors = []
    Average_circ_width_correlation_with_neighbors = []
    Frame_width_largest_percent_increase_and_decrease_width_increase = []
    Frame_width_largest_percent_increase = []
    Frame_width_largest_percent_decrease = []
    Percent_frames_width_percent_passed_thresh = []
    Number_frames_width_percent_passed_thresh = []
    Average_frame_width_largest_percent_increase_and_decrease = []
    Percent_frames_width_high_correlation_with_neighbors = []
    Number_frames_width_high_correlation_with_neighbors = []
    Average_frame_width_correlation_with_neighbors = []
    flaw_loc_correlation = []

    dtw_range_flaw = np.array(dtw_range_flaw)
    dtw_range_flaw_difference = np.array(dtw_range_flaw_difference)
    flaw_loc_DTW_range.append(dtw_range_flaw_difference[flaw_ax,flaw_circ])

    # Amplitude
    amplitudes_flaw = np.array(amplitudes_flaw)
    flaw_loc_amplitude.append(amplitudes_flaw[flaw_ax,flaw_circ])
    max_amplitude.append(np.max(amplitudes_flaw.flatten()))
    # Amplitude ratio
    amp_ratio_flaw = np.array(amp_ratio_flaw)
    flaw_loc_ampratio.append(amp_ratio_flaw[flaw_ax,flaw_circ])
    amp_ratio_flaw = amp_ratio_flaw < config['FLAW_AMP_RATIO_THRSH']
            
    # Correlations
    average_correlation.append(np.mean(APC_cors))
    max_correlation.append(np.max(APC_cors))
    flaw_loc_correlation.append(APC_cors[flaw_ax,flaw_circ])
    flaw_peaks_info =  all_frame_peaks
    flaw_peaks_info_circ= swap_axes(flaw_peaks_info)

    #filtering widths to remove large differences
    all_widths = []
    for circ in range(pred_circ_start,pred_circ_end+1):
        widths = [i['width'] for i in flaw_peaks_info_circ[circ-pred_circ_start]]
        diff=np.diff(widths,prepend=widths[0])
        for frame in range(len(diff)):
            if np.abs(diff[frame])>config['FLAW_PEAK_WIDTH_DIFFERENCE_LIMIT']:
                widths[frame] = widths[frame-1]
        all_widths.append(widths)

    widths_frames = np.swapaxes(all_widths,0,1)

    for frame in range(len(widths_frames)):
        widths = widths_frames[frame]
        diff=np.diff(widths,prepend=widths[0])
        for circ in range(len(diff)):
            if np.abs(diff[circ])>config['FLAW_PEAK_WIDTH_DIFFERENCE_LIMIT']:
                widths[circ] = widths[circ-1]
        widths_frames[frame] = widths

    all_widths = np.swapaxes(widths_frames,0,1)

    frame_width_changes = np.array([np.max(i)- np.min(i) for i in widths_frames])
    circ_width_changes = np.array([np.max(i)- np.min(i) for i in all_widths])

    frame_largest_width_change.append(np.arange(pred_ax_start+1,pred_ax_end+2)[np.argmax(frame_width_changes)])
    largest_circ_width_change.append(np.max(circ_width_changes))
    flaw_frame_width_change.append(frame_width_changes[flaw_ax])
    flaw_circ_width_change.append(circ_width_changes[flaw_circ])
    average_width.append(np.mean(all_widths))
    min_width.append(np.min(all_widths))
    max_width.append(np.max(all_widths))
    flaw_loc_width.append(all_widths[flaw_circ][flaw_ax])

    # Width changes in main flaw peak
    width_increases=[]
    width_decreases = []
    increase_first = []
    cors = []
    old_widths= widths_frames[0]
    a,b,c=find_largest_consecutive_changes(old_widths)
    width_increases.append(a)
    width_decreases.append(b)
    increase_first.append(c)

    for frame in range(1,len(widths_frames)):
        a,b,c=find_largest_consecutive_changes(widths_frames[frame])
        width_increases.append(a)
        width_decreases.append(b)
        increase_first.append(c)
        cors.append(np.corrcoef(widths_frames[frame],old_widths)[0][1])
        old_widths = widths_frames[frame]


    percent_increases_and_decreases = []
    for frame in range(len(width_increases)):
        if increase_first[frame]:
            percent_increases_and_decreases.append((width_increases[frame]+width_decreases[frame])/len(widths_frames[frame]))
        else:
            percent_increases_and_decreases.append(0)
    Frame_width_largest_percent_increase_and_decrease.append(100*np.max(percent_increases_and_decreases))
    Frame_width_largest_percent_increase_and_decrease_width_increase.append(frame_width_changes[np.argmax(percent_increases_and_decreases)])
    Frame_width_largest_percent_increase.append(100*(np.max(width_increases)/len(widths_frames[0])))
    Frame_width_largest_percent_decrease.append(100*(np.max(width_decreases)/len(widths_frames[0])))
    Percent_frames_width_percent_passed_thresh.append(100*(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH'])/len(percent_increases_and_decreases)))
    Number_frames_width_percent_passed_thresh.append(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH']))
    Average_frame_width_largest_percent_increase_and_decrease.append(100*np.mean(percent_increases_and_decreases))
    Percent_frames_width_high_correlation_with_neighbors.append(100*(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON'])/len(cors)))
    Number_frames_width_high_correlation_with_neighbors.append(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON']))
    Average_frame_width_correlation_with_neighbors.append(100*np.mean(cors))

    width_increases=[]
    width_decreases = []
    increase_first = []
    cors = []
    old_widths= all_widths[0]
    a,b,c=find_largest_consecutive_changes(old_widths)
    width_increases.append(a)
    width_decreases.append(b)
    increase_first.append(c)
    for circ in range(1,len(all_widths)):
        a,b,c=find_largest_consecutive_changes(all_widths[circ])
        width_increases.append(a)
        width_decreases.append(b)
        increase_first.append(c)
        cors.append(np.corrcoef(all_widths[circ],old_widths)[0][1])
        old_widths = all_widths[circ]

    percent_increases_and_decreases = []
    for circ in range(len(width_increases)):
        if increase_first[circ]:
            percent_increases_and_decreases.append((width_increases[circ]+width_decreases[circ])/len(all_widths[circ]))
        else:
            percent_increases_and_decreases.append(0)
    Circ_width_largest_percent_increase_and_decrease.append(100*np.max(percent_increases_and_decreases))
    Circ_width_largest_percent_increase_and_decrease_width_increase.append(circ_width_changes[np.argmax(percent_increases_and_decreases)])
    Circ_width_largest_percent_increase.append(100*(np.max(width_increases)/len(all_widths[0])))
    Circ_width_largest_percent_decrease.append(100*(np.max(width_decreases)/len(all_widths[0])))
    Percent_circ_width_percent_passed_thresh.append(100*(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH'])/len(percent_increases_and_decreases)))
    Number_circ_width_percent_passed_thresh.append(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH']))
    Average_circ_width_largest_percent_increase_and_decrease.append(100*np.mean(percent_increases_and_decreases))
    Percent_circ_width_high_correlation_with_neighbors.append(100*(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON'])/len(cors)))
    Number_circ_width_high_correlation_with_neighbors.append(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON']))
    Average_circ_width_correlation_with_neighbors.append(100*np.mean(cors))
    # -- Peak prominence changes in main flaw peak
    all_prominences = []
    for circ in range(pred_circ_start,pred_circ_end+1):
        prominences = [i['prominence'] for i in flaw_peaks_info_circ[circ-pred_circ_start]]
        all_prominences.append(prominences)
    all_prominences=np.array(all_prominences)
    prominences_frames = np.swapaxes(all_prominences,0,1)

    frame_prominence_changes = np.array([np.max(i)- np.min(i) for i in prominences_frames])
    circ_prominence_changes = np.array([np.max(i)- np.min(i) for i in all_prominences])
    frame_largest_prominence_change.append(np.arange(pred_ax_start+1,pred_ax_end+2)[np.argmax(frame_prominence_changes)])
    flaw_frame_prominence_change.append(frame_prominence_changes[flaw_ax])
    flaw_circ_prominence_change.append(circ_prominence_changes[flaw_circ])
    flaw_loc_prominence.append(all_prominences[flaw_circ][flaw_ax])
    flaw_loc_curvature.append(all_curvature_corrections[flaw_ax][flaw_circ])

    results = {
    'flaw_loc_DTW_range': flaw_loc_DTW_range,
    'flaw_loc_amplitude': flaw_loc_amplitude,
    'flaw_loc_ampratio': flaw_loc_ampratio,
    'average_correlation': average_correlation,
    'max_correlation': max_correlation,
    ' flaw_loc_correlation': flaw_loc_correlation,
    'frame_largest_width_change': frame_largest_width_change,
    'largest_circ_width_change': largest_circ_width_change,
    'flaw_frame_width_change': flaw_frame_width_change,
    'flaw_circ_width_change': flaw_circ_width_change,
    'average_width': average_width,
    'min_width': min_width,
    'max_width': max_width,
    'flaw_loc_width': flaw_loc_width,
    'frame_largest_prominence_change': frame_largest_prominence_change,
    'flaw_frame_prominence_change': flaw_frame_prominence_change,
    'flaw_circ_prominence_change': flaw_circ_prominence_change,
    'flaw_loc_prominence': flaw_loc_prominence,
    'flaw_loc curvature correction': flaw_loc_curvature,
    'max_amplitude': max_amplitude,
    'Frame_width_largest_percent_increase_and_decrease':Frame_width_largest_percent_increase_and_decrease,
    'Circ_width_largest_percent_increase_and_decrease':Circ_width_largest_percent_increase_and_decrease,
    'Frame_width_largest_percent_increase_and_decrease_width_increase':Frame_width_largest_percent_increase_and_decrease_width_increase,
    'Circ_width_largest_percent_increase_and_decrease_width_increase':Circ_width_largest_percent_increase_and_decrease_width_increase,
    'Frame_width_largest_percent_increase':Frame_width_largest_percent_increase,
    'Circ_width_largest_percent_increase':Circ_width_largest_percent_increase,
    'Frame_width_largest_percent_decrease':Frame_width_largest_percent_decrease,
    'Circ_width_largest_percent_decrease':Circ_width_largest_percent_decrease,
    'Average_frame_width_largest_percent_increase_and_decrease':Average_frame_width_largest_percent_increase_and_decrease,
    'Average_circ_width_largest_percent_increase_and_decrease':Average_circ_width_largest_percent_increase_and_decrease,
    'Percent_frames_width_high_correlation_with_neighbors':Percent_frames_width_high_correlation_with_neighbors,
    'Percent_circ_width_high_correlation_with_neighbors':Percent_circ_width_high_correlation_with_neighbors,
    'Average_circ_width_correlation_with_neighbors':Average_circ_width_correlation_with_neighbors,
    'Average_frame_width_correlation_with_neighbors':Average_frame_width_correlation_with_neighbors,
    }
    return results


def flagging_issues(pred_circ_start, pred_circ_end, pred_ax_start, pred_ax_end, flaw_circ, 
                    flaw_ax, config, depth_flaw_arr_unfiltered, pred_depth_flaw, depth_flaw_arr,
                    depth_flaw_start_of_peak_arr):

    (flaw_ax_sop, flaw_circ_sop) = unravel_index(np.nanargmax(depth_flaw_start_of_peak_arr), depth_flaw_start_of_peak_arr.shape)
    flaw_ax_start_of_peak=flaw_ax_sop
    flaw_circ_start_of_peak=flaw_circ_sop
    
    length = pred_ax_end - pred_ax_start
    width = pred_circ_end - pred_circ_start
    max_depths = np.nanmax(depth_flaw_arr, axis=1)
    max_depth_lags = max_depths / (config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'])
    criteria_max_depth_lags = np.nan_to_num(
        np.sum(np.abs(np.gradient(max_depth_lags))[np.abs(np.gradient(max_depth_lags)) >= config['LAG_GRADIENT_THRESH']]), 
        copy=False, nan=0)
    flattened_depth_flaw_unfiltered = depth_flaw_arr_unfiltered.flatten()

    # Flagging issues -- lag gradients
    flaw_depth_lags = depth_flaw_arr / (config['UNIT_MICRO_SEC'] * config['MICRO_SEC_DEPTH_SHEAR'])
    criteria_lags = [np.sum(np.abs(np.gradient(i))[np.abs(np.gradient(i)) >= config['LAG_GRADIENT_THRESH']]) for i in flaw_depth_lags]
    criteria_lags = np.nan_to_num(criteria_lags, copy=False, nan=0)
    gradient_flaw = criteria_lags[flaw_ax]

    # Frame distribution flagging
    frames_below_reportable = 100 * (np.sum(max_depths <= config['REPORTABLE_THRESH']) / len(max_depths))
    frames_close_to_max_depth = 100 * (np.sum(np.abs(max_depths - pred_depth_flaw) <= config['REPORTABLE_THRESH']) / len(max_depths))

    # Unfiltered flagging
    circ_locations_close_to_max_depth_number = np.sum(np.abs(depth_flaw_arr_unfiltered.flatten()-pred_depth_flaw)<=config['REPORTABLE_THRESH'])
    circ_locations_close_to_max_depth = np.sum(np.abs(flattened_depth_flaw_unfiltered - pred_depth_flaw) <= config['REPORTABLE_THRESH']) / len(flattened_depth_flaw_unfiltered)
    circ_locations_above_max_depth_number=sum(depth_flaw_arr_unfiltered.flatten()>pred_depth_flaw)
    # Gather results
    results = {
        'deepest_location_percent_ax': 100 * (flaw_ax / length),
        'deepest_location_percent_ro': 100 * (flaw_circ / width),
        'Frame Max Depth Gradient': criteria_max_depth_lags,
        'Frames Below Reportable %': frames_below_reportable,
        'Frames Close to Max Depth %': frames_close_to_max_depth,
        'circ_locations_close_to_max_depth': circ_locations_close_to_max_depth,
        'circ_locations_close_to_max_depth_number': circ_locations_close_to_max_depth_number,
        'Flaw Gradient Index': gradient_flaw,
        'Number Circ. Locations Above Max Depth':circ_locations_above_max_depth_number,
        'flaw_ax_start_of_peak':flaw_ax_start_of_peak,
        'flaw_circ_start_of_peak':flaw_circ_start_of_peak,
    }

    return results

def bonus_peak_info(flaw_circ,flaw_ax,pred_ax_start,pred_ax_end,pred_circ_start,pred_circ_end,flaw_bonus_peaks_info_circ,config):

    bonus_frame_largest_amp_change = []
    bonus_largest_frame_amp_change = []
    bonus_circ_with_largest_amp_change = []
    bonus_largest_circ_amp_change = []
    bonus_percent_frames_large_amp_changes = []
    bonus_percent_circ_large_amp_changes = []
    bonus_flaw_frame_amp_change = []
    bonus_flaw_circ_amp_change = []
    bonus_average_amp = []
    bonus_min_amp = []
    bonus_max_amp = []
    bonus_flaw_loc_amp = []
    Frame_bonus_amp_largest_percent_increase_and_decrease = []
    Frame_bonus_amp_largest_percent_increase_and_decrease_amp_increase = []
    Frame_bonus_amp_largest_percent_increase = []
    Frame_bonus_amp_largest_percent_decrease = []
    Percent_frames_bonus_amp_percent_passed_thresh = []
    Number_frames_bonus_amp_percent_passed_thresh = []
    Average_frame_bonus_amp_largest_percent_increase_and_decrease = []
    Percent_frames_bonus_amp_high_correlation_with_neighbors = []
    Number_frames_bonus_amp_high_correlation_with_neighbors = []
    Average_frame_bonus_amp_correlation_with_neighbors = []
    Circ_bonus_amp_largest_percent_increase_and_decrease = []
    Circ_bonus_amp_largest_percent_increase_and_decrease_amp_increase = []
    Circ_bonus_amp_largest_percent_increase = []
    Circ_bonus_amp_largest_percent_decrease = []
    Percent_circ_bonus_amp_percent_passed_thresh = []
    Number_circ_bonus_amp_percent_passed_thresh = []
    Average_circ_bonus_amp_largest_percent_increase_and_decrease = []
    Percent_circ_bonus_amp_high_correlation_with_neighbors = []
    Number_circ_bonus_amp_high_correlation_with_neighbors = []
    Average_circ_bonus_amp_correlation_with_neighbors = []
    bonus_frame_largest_prominence_change = []
    bonus_largest_frame_prominence_change = []
    bonus_circ_with_largest_prominence_change = []
    bonus_largest_circ_prominence_change = []
    bonus_percent_frames_large_prominence_changes = []
    bonus_percent_circ_large_prominence_changes = []
    bonus_flaw_frame_prominence_change = []
    bonus_flaw_circ_prominence_change = []
    bonus_average_prominence = []
    bonus_min_prominence = []
    bonus_max_prominence = []
    bonus_flaw_loc_prominence = []
    Frame_bonus_prominence_largest_percent_increase_and_decrease = []
    Frame_bonus_prominence_largest_percent_increase_and_decrease_prominence_increase = []
    Frame_bonus_prominence_largest_percent_increase = []
    Frame_bonus_prominence_largest_percent_decrease = []
    Percent_frames_bonus_prominence_percent_passed_thresh = []
    Number_frames_bonus_prominence_percent_passed_thresh = []
    Average_frame_bonus_prominence_largest_percent_increase_and_decrease = []
    Percent_frames_bonus_prominence_high_correlation_with_neighbors = []
    Number_frames_bonus_prominence_high_correlation_with_neighbors = []
    Average_frame_bonus_prominence_correlation_with_neighbors = []
    Circ_bonus_prominence_largest_percent_increase_and_decrease = []
    Circ_bonus_prominence_largest_percent_increase_and_decrease_prominence_increase = []
    Circ_bonus_prominence_largest_percent_increase = []
    Circ_bonus_prominence_largest_percent_decrease = []
    Percent_circ_bonus_prominence_percent_passed_thresh = []
    Number_circ_bonus_prominence_percent_passed_thresh = []
    Average_circ_bonus_prominence_largest_percent_increase_and_decrease = []
    Percent_circ_bonus_prominence_high_correlation_with_neighbors = []
    Number_circ_bonus_prominence_high_correlation_with_neighbors = []
    Average_circ_bonus_prominence_correlation_with_neighbors = []

    # IE. changes in largest peak earlier than the main flaw peak
    all_amps = []
    for circ in range(pred_circ_start,pred_circ_end+1):
        amps = [i['height'] for i in flaw_bonus_peaks_info_circ[circ-pred_circ_start]]
        all_amps.append(amps)
    all_amps=np.array(all_amps)
    amps_frames = np.swapaxes(all_amps,0,1)

    
    frame_amp_changes = np.array([np.max(i)- np.min(i) for i in amps_frames])
    circ_amp_changes = np.array([np.max(i)- np.min(i) for i in all_amps])
    bonus_frame_largest_amp_change.append(np.arange(pred_ax_start+1,pred_ax_end+2)[np.argmax(frame_amp_changes)])
    bonus_largest_frame_amp_change.append(np.max(frame_amp_changes))
    bonus_circ_with_largest_amp_change.append(np.arange(pred_circ_start,pred_circ_end+1)[np.argmax(circ_amp_changes)])
    bonus_largest_circ_amp_change.append(np.max(circ_amp_changes))
    bonus_percent_frames_large_amp_changes.append(100*(sum(frame_amp_changes>config['BONUS_FLAW_PEAK_AMP_DIFFERENCE_LIMIT'])/frame_amp_changes.shape[0]))
    bonus_percent_circ_large_amp_changes.append(100*(sum(circ_amp_changes>config['BONUS_FLAW_PEAK_AMP_DIFFERENCE_LIMIT'])/circ_amp_changes.shape[0]))
    bonus_flaw_frame_amp_change.append(frame_amp_changes[flaw_ax])
    bonus_flaw_circ_amp_change.append(circ_amp_changes[flaw_circ])
    bonus_average_amp.append(np.mean(all_amps))
    bonus_min_amp.append(np.min(all_amps))
    bonus_max_amp.append(np.max(all_amps))
    bonus_flaw_loc_amp.append(all_amps[flaw_circ][flaw_ax])

    amp_increases=[]
    amp_decreases = []
    increase_first = []
    cors = []
    old_amps= amps_frames[0]
    a,b,c=find_largest_consecutive_changes(old_amps)
    amp_increases.append(a)
    amp_decreases.append(b)
    increase_first.append(c)

    for frame in range(1,len(amps_frames)):
        a,b,c=find_largest_consecutive_changes(amps_frames[frame])
        amp_increases.append(a)
        amp_decreases.append(b)
        increase_first.append(c)
        cors.append(np.corrcoef(amps_frames[frame],old_amps)[0][1])
        old_amps = amps_frames[frame]


    percent_increases_and_decreases = []
    for frame in range(len(amp_increases)):
        if increase_first[frame]:
            percent_increases_and_decreases.append((amp_increases[frame]+amp_decreases[frame])/len(amps_frames[frame]))
        else:
            percent_increases_and_decreases.append(0)

    Frame_bonus_amp_largest_percent_increase_and_decrease.append(100*np.max(percent_increases_and_decreases))
    Frame_bonus_amp_largest_percent_increase_and_decrease_amp_increase.append(frame_amp_changes[np.argmax(percent_increases_and_decreases)])
    Frame_bonus_amp_largest_percent_increase.append(100*(np.max(amp_increases)/len(amps_frames[0])))
    Frame_bonus_amp_largest_percent_decrease.append(100*(np.max(amp_decreases)/len(amps_frames[0])))
    Percent_frames_bonus_amp_percent_passed_thresh.append(100*(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH'])/len(percent_increases_and_decreases)))
    Number_frames_bonus_amp_percent_passed_thresh.append(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH']))
    Average_frame_bonus_amp_largest_percent_increase_and_decrease.append(100*np.mean(percent_increases_and_decreases))
    Percent_frames_bonus_amp_high_correlation_with_neighbors.append(100*(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON'])/len(cors)))
    Number_frames_bonus_amp_high_correlation_with_neighbors.append(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON']))
    Average_frame_bonus_amp_correlation_with_neighbors.append(100*np.mean(cors))

    amp_increases=[]
    amp_decreases = []
    increase_first = []
    cors = []
    old_amps= all_amps[0]
    a,b,c=find_largest_consecutive_changes(old_amps)
    amp_increases.append(a)
    amp_decreases.append(b)
    increase_first.append(c)
    for circ in range(1,len(all_amps)):
        a,b,c=find_largest_consecutive_changes(all_amps[circ])
        amp_increases.append(a)
        amp_decreases.append(b)
        increase_first.append(c)
        cors.append(np.corrcoef(all_amps[circ],old_amps)[0][1])
        old_amps = all_amps[circ]


    percent_increases_and_decreases = []
    for circ in range(len(amp_increases)):
        if increase_first[circ]:
            percent_increases_and_decreases.append((amp_increases[circ]+amp_decreases[circ])/len(all_amps[circ]))
        else:
            percent_increases_and_decreases.append(0)
    Circ_bonus_amp_largest_percent_increase_and_decrease.append(100*np.max(percent_increases_and_decreases))
    Circ_bonus_amp_largest_percent_increase_and_decrease_amp_increase.append(circ_amp_changes[np.argmax(percent_increases_and_decreases)])
    Circ_bonus_amp_largest_percent_increase.append(100*(np.max(amp_increases)/len(all_amps[0])))
    Circ_bonus_amp_largest_percent_decrease.append(100*(np.max(amp_decreases)/len(all_amps[0])))
    Percent_circ_bonus_amp_percent_passed_thresh.append(100*(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH'])/len(percent_increases_and_decreases)))
    Number_circ_bonus_amp_percent_passed_thresh.append(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH']))
    Average_circ_bonus_amp_largest_percent_increase_and_decrease.append(100*np.mean(percent_increases_and_decreases))
    Percent_circ_bonus_amp_high_correlation_with_neighbors.append(100*(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON'])/len(cors)))
    Number_circ_bonus_amp_high_correlation_with_neighbors.append(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON']))
    Average_circ_bonus_amp_correlation_with_neighbors.append(100*np.mean(cors))
    

    all_prominences = []
    for circ in range(pred_circ_start,pred_circ_end+1):
        prominences = [i['prominence'] for i in flaw_bonus_peaks_info_circ[circ-pred_circ_start]]
        all_prominences.append(prominences)
    all_prominences=np.array(all_prominences)
    prominences_frames = np.swapaxes(all_prominences,0,1)

    frame_prominence_changes = np.array([np.max(i)- np.min(i) for i in prominences_frames])
    circ_prominence_changes = np.array([np.max(i)- np.min(i) for i in all_prominences])
    bonus_frame_largest_prominence_change.append(np.arange(pred_ax_start+1,pred_ax_end+2)[np.argmax(frame_prominence_changes)])
    bonus_largest_frame_prominence_change.append(np.max(frame_prominence_changes))
    bonus_circ_with_largest_prominence_change.append(np.arange(pred_circ_start,pred_circ_end+1)[np.argmax(circ_prominence_changes)])
    bonus_largest_circ_prominence_change.append(np.max(circ_prominence_changes))
    bonus_percent_frames_large_prominence_changes.append(100*(sum(frame_prominence_changes>config['FLAW_PEAK_PROMINENCE_DIFFERENCE_LIMIT'])/frame_prominence_changes.shape[0]))
    bonus_percent_circ_large_prominence_changes.append(100*(sum(circ_prominence_changes>config['FLAW_PEAK_PROMINENCE_DIFFERENCE_LIMIT'])/circ_prominence_changes.shape[0]))
    bonus_flaw_frame_prominence_change.append(frame_prominence_changes[flaw_ax])
    bonus_flaw_circ_prominence_change.append(circ_prominence_changes[flaw_circ])
    bonus_average_prominence.append(np.mean(all_prominences))
    bonus_min_prominence.append(np.min(all_prominences))
    bonus_max_prominence.append(np.max(all_prominences))
    bonus_flaw_loc_prominence.append(all_prominences[flaw_circ][flaw_ax])
    
    prominence_increases=[]
    prominence_decreases = []
    increase_first = []
    cors = []
    old_prominences= prominences_frames[0]
    a,b,c=find_largest_consecutive_changes(old_prominences)
    prominence_increases.append(a)
    prominence_decreases.append(b)
    increase_first.append(c)

    for frame in range(1,len(prominences_frames)):
        a,b,c=find_largest_consecutive_changes(prominences_frames[frame])
        prominence_increases.append(a)
        prominence_decreases.append(b)
        increase_first.append(c)
        cors.append(np.corrcoef(prominences_frames[frame],old_prominences)[0][1])
        old_prominences = prominences_frames[frame]


    percent_increases_and_decreases = []
    for frame in range(len(prominence_increases)):
        if increase_first[frame]:
            percent_increases_and_decreases.append((prominence_increases[frame]+prominence_decreases[frame])/len(prominences_frames[frame]))
        else:
            percent_increases_and_decreases.append(0)

    Frame_bonus_prominence_largest_percent_increase_and_decrease.append(100*np.max(percent_increases_and_decreases))
    Frame_bonus_prominence_largest_percent_increase_and_decrease_prominence_increase.append(frame_prominence_changes[np.argmax(percent_increases_and_decreases)])
    Frame_bonus_prominence_largest_percent_increase.append(100*(np.max(prominence_increases)/len(prominences_frames[0])))
    Frame_bonus_prominence_largest_percent_decrease.append(100*(np.max(prominence_decreases)/len(prominences_frames[0])))
    Percent_frames_bonus_prominence_percent_passed_thresh.append(100*(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH'])/len(percent_increases_and_decreases)))
    Number_frames_bonus_prominence_percent_passed_thresh.append(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH']))
    Average_frame_bonus_prominence_largest_percent_increase_and_decrease.append(100*np.mean(percent_increases_and_decreases))
    Percent_frames_bonus_prominence_high_correlation_with_neighbors.append(100*(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON'])/len(cors)))
    Number_frames_bonus_prominence_high_correlation_with_neighbors.append(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON']))
    Average_frame_bonus_prominence_correlation_with_neighbors.append(100*np.mean(cors))

    prominence_increases=[]
    prominence_decreases = []
    increase_first = []
    cors = []
    old_prominences= all_prominences[0]
    a,b,c=find_largest_consecutive_changes(old_prominences)
    prominence_increases.append(a)
    prominence_decreases.append(b)
    increase_first.append(c)
    for circ in range(1,len(all_prominences)):
        a,b,c=find_largest_consecutive_changes(all_prominences[circ])
        prominence_increases.append(a)
        prominence_decreases.append(b)
        increase_first.append(c)
        cors.append(np.corrcoef(all_prominences[circ],old_prominences)[0][1])
        old_prominences = all_prominences[circ]


    percent_increases_and_decreases = []
    for circ in range(len(prominence_increases)):
        if increase_first[circ]:
            percent_increases_and_decreases.append((prominence_increases[circ]+prominence_decreases[circ])/len(all_prominences[circ]))
        else:
            percent_increases_and_decreases.append(0)
    Circ_bonus_prominence_largest_percent_increase_and_decrease.append(100*np.max(percent_increases_and_decreases))
    Circ_bonus_prominence_largest_percent_increase_and_decrease_prominence_increase.append(circ_prominence_changes[np.argmax(percent_increases_and_decreases)])
    Circ_bonus_prominence_largest_percent_increase.append(100*(np.max(prominence_increases)/len(all_prominences[0])))
    Circ_bonus_prominence_largest_percent_decrease.append(100*(np.max(prominence_decreases)/len(all_prominences[0])))
    Percent_circ_bonus_prominence_percent_passed_thresh.append(100*(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH'])/len(percent_increases_and_decreases)))
    Number_circ_bonus_prominence_percent_passed_thresh.append(sum(np.array(percent_increases_and_decreases)>=config['INCREASE_DECREASE_PERCENT_THRESH']))
    Average_circ_bonus_prominence_largest_percent_increase_and_decrease.append(100*np.mean(percent_increases_and_decreases))
    Percent_circ_bonus_prominence_high_correlation_with_neighbors.append(100*(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON'])/len(cors)))
    Number_circ_bonus_prominence_high_correlation_with_neighbors.append(sum(np.array(cors)>=config['HIGH_COR_THRESH_COMPARISON']))
    Average_circ_bonus_prominence_correlation_with_neighbors.append(100*np.mean(cors))

    return {'Frame_bonus_amp_largest_percent_increase_and_decrease':Frame_bonus_amp_largest_percent_increase_and_decrease,
            'Frame_bonus_amp_largest_percent_increase_and_decrease_amp_increase':Frame_bonus_amp_largest_percent_increase_and_decrease_amp_increase,
            'Frame_bonus_amp_largest_percent_increase':Frame_bonus_amp_largest_percent_increase,
            'Frame_bonus_amp_largest_percent_decrease':Frame_bonus_amp_largest_percent_decrease,
            'Average_frame_bonus_amp_largest_percent_increase_and_decrease':Average_frame_bonus_amp_largest_percent_increase_and_decrease,
            'Average_circ_bonus_amp_largest_percent_increase_and_decrease':Average_circ_bonus_amp_largest_percent_increase_and_decrease,
            'Percent_circ_bonus_amp_high_correlation_with_neighbors':Percent_circ_bonus_amp_high_correlation_with_neighbors,
            'Average_circ_bonus_amp_correlation_with_neighbors':Average_circ_bonus_amp_correlation_with_neighbors,
            'Frame_bonus_prominence_largest_percent_increase_and_decrease':Frame_bonus_prominence_largest_percent_increase_and_decrease,
            'Frame_bonus_prominence_largest_percent_increase_and_decrease_prominence_increase':Frame_bonus_prominence_largest_percent_increase_and_decrease_prominence_increase,
            'Frame_bonus_prominence_largest_percent_increase':Frame_bonus_prominence_largest_percent_increase,
            'Frame_bonus_prominence_largest_percent_decrease':Frame_bonus_prominence_largest_percent_decrease,
            'Circ_bonus_prominence_largest_percent_decrease':Circ_bonus_prominence_largest_percent_decrease,
            'Average_frame_bonus_prominence_largest_percent_increase_and_decrease':Average_frame_bonus_prominence_largest_percent_increase_and_decrease,
            'Average_circ_bonus_prominence_largest_percent_increase_and_decrease':Average_circ_bonus_prominence_largest_percent_increase_and_decrease,
            'Percent_frames_bonus_prominence_high_correlation_with_neighbors':Percent_frames_bonus_prominence_high_correlation_with_neighbors,
            'Number_circ_bonus_prominence_high_correlation_with_neighbors':Number_circ_bonus_prominence_high_correlation_with_neighbors,
            'Average_circ_bonus_prominence_correlation_with_neighbors':Average_circ_bonus_prominence_correlation_with_neighbors,
            'Average_frame_bonus_prominence_correlation_with_neighbors':Average_frame_bonus_prominence_correlation_with_neighbors,
            'bonus_flaw_frame_amp_change':bonus_flaw_frame_amp_change,
            'bonus_average_amp':bonus_average_amp,
            'bonus_flaw_loc_amp':bonus_flaw_loc_amp,
            'bonus_frame_largest_prominence_change':bonus_frame_largest_prominence_change,
            'bonus_average_prominence':bonus_average_prominence
            }

def deepest_flaw_bonus_peak(all_frame_bonus_peaks,find_deepest_location,config,pred_ax_start):
    
    bonus_flaw_ax = []
    bonus_flaw_ax_circ = []
    bonus_longest_sequence_circ = []
    flaw_bonus_peaks_info = all_frame_bonus_peaks
    flaw_bonus_peaks_info_circ= swap_axes(flaw_bonus_peaks_info)
    try:
        found_location,frame_deepest,circ_deepest,longest_sequence=find_deepest_location(flaw_bonus_peaks_info,config['EXTRA_PEAK_SAME_CIRC_THRESH'],config['EXTRA_PEAK_SEQUENCE_THRESH'])
    except Exception:
        found_location,frame_deepest,circ_deepest,longest_sequence = False,0,0,0
    bonus_flaw_ax.append(frame_deepest+pred_ax_start+1)

    # Try the same but with circumferential locations
    try:
        found_location_circ,circ_deepest_circ,frame_deepest_circ,longest_sequence_circ=find_deepest_location(flaw_bonus_peaks_info_circ,config['EXTRA_PEAK_SAME_CIRC_THRESH'],config['EXTRA_PEAK_SEQUENCE_THRESH'])
    except Exception:
        found_location_circ,circ_deepest_circ,frame_deepest_circ,longest_sequence_circ = False,0,0,0
    bonus_flaw_ax_circ.append(frame_deepest_circ+pred_ax_start+1)
    bonus_longest_sequence_circ.append(longest_sequence_circ/len(all_frame_bonus_peaks[0]))
    
    return  {'bonus_flaw_ax': bonus_flaw_ax,
            'bonus_flaw_ax_circ': bonus_flaw_ax_circ,
            ' bonus_longest_sequence_circ': bonus_longest_sequence_circ}