from typing import Dict, List, Tuple, Optional, Any
import numpy as np
from numpy.polynomial import Polynomial

def find_surface(flaw, signal, bscan_data_frame, flaw_position):
    
    flaw_circ_start = max(0, flaw.x_start)
    flaw_circ_end = flaw.x_end

    for flaw_position in range(flaw_circ_start, flaw_circ_end+1):
        flaw_position = 159
        left_ref, right_ref, trend = find_balanced_reference_points(signal, flaw_position, flaw_circ_start, flaw_circ_end)
        left_surface = bscan_data_frame[left_ref,:]
        right_surface = bscan_data_frame[right_ref,:]

        plot_surfaces_and_flaw_subplots(bscan_data_frame, left_surface, right_surface, flaw_position)

def find_balanced_reference_points(signal, flaw_position, flaw_start, flaw_end, search_range=15, balance_factor=0.7, poly_degree=2):
    # Define search ranges
    left_search_start = max(0, flaw_start - search_range)
    left_search_end = flaw_start
    right_search_start = flaw_end
    right_search_end = min(len(signal), flaw_end + search_range)
    
    # Calculate overall trend of the signal using polynomial fit
    x = np.arange(len(signal))
    mask = np.ones(len(signal), dtype=bool)
    mask[flaw_start:flaw_end+1] = False  # Exclude flaw region from trend calculation
    poly = Polynomial.fit(x[mask], signal[mask], poly_degree)
    trend = poly(x)
    
    # Define function to detrend the signal
    detrend = lambda val: val - trend
    
    # Calculate mean of detrended signal excluding flaw and search ranges
    detrended_signal = detrend(signal)
    mean_signal = np.mean(detrended_signal[np.logical_or(
        np.arange(len(signal)) < left_search_start,
        np.arange(len(signal)) > right_search_end
    )])
    
    # Find points closest to detrended mean in each search range
    left_candidates = np.abs(detrended_signal[left_search_start:left_search_end] - mean_signal)
    right_candidates = np.abs(detrended_signal[right_search_start:right_search_end] - mean_signal)
    
    # Calculate ideal distances based on available space
    left_ideal = min(flaw_position - left_search_start, right_search_end - flaw_position)
    right_ideal = min(flaw_position - left_search_start, right_search_end - flaw_position)
    
    # Combine distance from mean and distance from ideal position
    left_scores = left_candidates + balance_factor * np.abs(np.arange(len(left_candidates)) - left_ideal)
    right_scores = right_candidates + balance_factor * np.abs(np.arange(len(right_candidates)) - right_ideal)
    
    left_ref = left_search_start + np.argmin(left_scores)
    right_ref = right_search_start + np.argmin(right_scores)
    return left_ref, right_ref, trend

def plot_bscan_with_references_and_trend(bscan_data, signal, flaw_start, flaw_end, flaw_position, left_ref, right_ref, trend, search_range=15):
 
    # Calculate search ranges
    left_search_start = max(0, flaw_start - search_range)
    left_search_end = flaw_start
    right_search_start = flaw_end
    right_search_end = min(len(signal), flaw_end + search_range)

    plt.figure(figsize=(15, 8))
    
    # Plot B-scan
    plt.imshow(bscan_data, aspect='auto', cmap='gray')
    
    # Plot horizontal lines
    plt.axhline(y=flaw_start, color='r', linestyle='--', label='Flaw start')
    plt.axhline(y=flaw_end, color='r', linestyle='--', label='Flaw end')
    plt.axhline(y=flaw_position, color='m', linestyle='-', label='Flaw position')
    plt.axhline(y=left_ref, color='g', linestyle=':', label='Left reference')
    plt.axhline(y=right_ref, color='g', linestyle=':', label='Right reference')
    
    # Plot search ranges
    plt.axhline(y=left_search_start, color='y', linestyle='-.', label='Left search start')
    plt.axhline(y=left_search_end, color='y', linestyle='-.', label='Left search end')
    plt.axhline(y=right_search_start, color='y', linestyle='-.', label='Right search start')
    plt.axhline(y=right_search_end, color='y', linestyle='-.', label='Right search end')
    
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.title('B-scan with Flaw, Search Ranges, and Reference Points')
    plt.xlabel('A-scan')
    plt.ylabel('Depth')
    
    # Invert y-axis to have 0 at the top
    plt.gca().invert_yaxis()
    
    plt.tight_layout()
    plt.show()

    # Plot signal and trend separately
    plt.figure(figsize=(15, 4))
    plt.plot(signal, label='Signal')
    plt.plot(trend, 'r--', label='Overall trend')
    plt.legend()
    plt.title('Signal and Overall Trend')
    plt.xlabel('Depth')
    plt.ylabel('Amplitude')
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.tight_layout()
    plt.show()
# if __name__ == '__main__':

    # import matplotlib.pyplot as plt
    # bscan_data_frame = probes_data['NB1'].data[25, :, :]
    # signal = cscans_output[2]['average_amplitudes'][25]
    # flaw = scan.flaws[0]

    # amp = cscans_output[2]['average_amplitudes']
    # b_amp = amp[22]
    # signal = b_amp

    # flaw_start, flaw_end = 152, 176
    # flaw_position = 162
    # left_ref, right_ref, trend = find_balanced_reference_points(signal, flaw_position, flaw_start, flaw_end)

    # print(f"Left reference point: {left_ref}")
    # print(f"Right reference point: {right_ref}")
    # print(f"Distance to left: {flaw_position - left_ref}")
    # print(f"Distance to right: {right_ref - flaw_position}")
    
    # plot_bscan_with_references_and_trend(bscan_data_frame, signal, flaw_circ_start, flaw_circ_end, flaw_position, left_ref, right_ref, trend, search_range=15)
    # """
    # Initially, it aimed to find reference points outside a flaw region in a signal.
    # T`he function was improved to balance finding undisturbed points and maintaining equidistance from a specific flaw position.
    # It was then updated to account for linear trends in the signal by detrending before selecting reference points.
    # Finally, it was enhanced to handle non-linear trends using polynomial fitting, allowing for more accurate reference point selection in signals with complex shapes.
    # Throughout its development, the function maintained a focus on balancing three key factors: proximity to the local mean (undisturbed signal), equidistance from the flaw position, and consideration of the overall signal trend.
    # The function now uses a scoring system that combines deviation from the detrended mean and distance from the ideal equidistant position, with an adjustable balance factor.`
    # Visualization capabilities were added to help users understand and verify the reference point selection process.
    # """

    