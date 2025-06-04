from find_surface import find_surface, find_balanced_reference_points
from detect_peak import detect_all_peaks
from typing import List, Tuple
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np

def calculate_flaw_depth(flaw, cscans_output, bscan_data, config):
    """
    Calculate debris depth for each position within the flaw's range across multiple frames.

    Parameters:
    -----------
    flaw : object
        Flaw object containing x_start, x_end, y_start, and y_end attributes.
    signal : np.array
        1D array representing the overall signal.
    bscan_data : np.array
        3D array representing the B-scan data (frame, position, depth).

    Returns:
    --------
    depth_array : np.array
        2D array containing depth values for each frame and circumferential position.
    max_depth : int
        Maximum depth found across all flaw positions and frames.
    max_depth_info : dict
        Dictionary containing information about the maximum depth (frame, position, depth, reference_peak, flaw_peak).
    """
    flaw_circ_start = max(0, flaw.x_start)
    flaw_circ_end = flaw.x_end
    flaw_axial_start = max(0, flaw.y_start)
    flaw_axial_end = flaw.y_end

    num_frames = flaw_axial_end - flaw_axial_start + 1
    num_positions = flaw_circ_end - flaw_circ_start + 1

    depth_array = np.zeros((num_frames, num_positions), dtype=np.float32)
    max_depth = 0
    max_depth_info = None

    for i, frame in enumerate(range(flaw_axial_start, flaw_axial_end + 1)):
        print(f"Processing frame {frame}...")
        bscan_data_frame = bscan_data[frame, :, :]
        signal = cscans_output[2]['average_amplitudes'][frame]
        for j, flaw_position in enumerate(range(flaw_circ_start, flaw_circ_end + 1)):
            left_ref, right_ref, trend = find_balanced_reference_points(signal, flaw_position, flaw_circ_start, flaw_circ_end)
            left_surface = bscan_data_frame[left_ref, :]
            right_surface = bscan_data_frame[right_ref, :]
            major_peaks, potential_flaw_peaks = detect_all_peaks(left_surface, right_surface, bscan_data_frame[flaw_position, :], config)
            depth, reference_peak, flaw_peak = calculate_depth(major_peaks, potential_flaw_peaks)

            if depth is not None:
                depth_array[i, j] = depth

                if depth > max_depth:
                    max_depth = depth
                    max_depth_info = {
                        'frame': frame,
                        'position': flaw_position,
                        'depth': depth,
                        'reference_peak': reference_peak,
                        'flaw_peak': flaw_peak,
                        'potential_flaw_peaks': potential_flaw_peaks
                    }
    
    # fig = plot_flaw_peaks(bscan_data[max_depth_info['frame'], :, :], 
    #                 max_depth_info['potential_flaw_peaks'], 
    #                 max_depth_info['position'])
    
    # visualize_depth_array(depth_array, flaw_circ_start, flaw_axial_start)
    return depth_array, max_depth, max_depth_info

def convert_units(depth):
    micro_dec_per_index = 0.710
    unit_micro_sec = 0.008
    return max(np.round(depth * micro_dec_per_index * unit_micro_sec, 4), 0)

def calculate_depth(major_peaks, potential_flaw_peaks):
    """
    Calculate the depth based on the reference surface peaks and potential flaw peaks.
    
    Parameters:
    -----------
    major_peaks : dict
        Dictionary containing major peaks from left and right surfaces.
    potential_flaw_peaks : list
        List of tuples (index, amplitude) for potential flaw peaks.
    
    Returns:
    --------
    depth : int
        Calculated depth (difference between reference and flaw peak indices).
    reference_peak : int
        Index of the reference peak used.
    flaw_peak : int
        Index of the flaw peak used.
    """
    # Find the largest peak index from reference surfaces
    all_surface_peaks = major_peaks['left'] + major_peaks['right']
    reference_peak = max(all_surface_peaks, key=lambda x: x[0])[0]
    
    # Round to the nearest even number
    reference_peak = round(reference_peak / 2) * 2
    
    # Sort flaw peaks by index (ascending order)
    sorted_flaw_peaks = sorted(potential_flaw_peaks, key=lambda x: x[0])
    
    # Find the rightmost flaw peak that's to the right of the reference peak
    flaw_peak = None
    for peak in reversed(sorted_flaw_peaks):
        if peak[0] > reference_peak:
            flaw_peak = peak[0]
            break
    
    # If no suitable flaw peak found, use the rightmost peak
    if flaw_peak is None and sorted_flaw_peaks:
        flaw_peak = sorted_flaw_peaks[-1][0]
    
    # Calculate depth
    if flaw_peak is not None:
        depth = flaw_peak - reference_peak
        
        return convert_units(depth), reference_peak, flaw_peak
    else:
        return None, reference_peak, None

def visualize_depth_array(depth_array, flaw_circ_start, flaw_axial_start):
    """
    Visualize the depth_array using multiple plot types in a single large figure.
    
    Parameters:
    -----------
    depth_array : np.array
        2D array containing depth values for each frame and circumferential position.
    flaw_circ_start : int
        Starting circumferential position of the flaw.
    flaw_axial_start : int
        Starting axial position (frame) of the flaw.

    Returns:
    --------
    matplotlib.figure.Figure
        The figure containing all plots.
    """
    
    fig = plt.figure(figsize=(24, 20))
    
    # 1. Heatmap
    ax1 = fig.add_subplot(231)
    im = ax1.imshow(depth_array, aspect='auto', cmap='viridis')
    cbar1 = fig.colorbar(im, ax=ax1, label='Depth')
    cbar1.ax.tick_params(labelsize=10)
    ax1.set_title('Debris Depth Heatmap', fontsize=14)
    ax1.set_xlabel('Circumferential Position', fontsize=12)
    ax1.set_ylabel('Frame', fontsize=12)
    
    # Add actual position labels
    x_ticks = np.arange(0, depth_array.shape[1], depth_array.shape[1]//5)
    y_ticks = np.arange(0, depth_array.shape[0], depth_array.shape[0]//5)
    ax1.set_xticks(x_ticks)
    ax1.set_xticklabels([str(i + flaw_circ_start) for i in x_ticks], rotation=45, ha='right')
    ax1.set_yticks(y_ticks)
    ax1.set_yticklabels([str(i + flaw_axial_start) for i in y_ticks])
    ax1.tick_params(axis='both', which='major', labelsize=10)

    # 2. 3D Surface Plot
    ax2 = fig.add_subplot(232, projection='3d')
    x = np.arange(depth_array.shape[1]) + flaw_circ_start
    y = np.arange(depth_array.shape[0]) + flaw_axial_start
    X, Y = np.meshgrid(x, y)
    surf = ax2.plot_surface(X, Y, depth_array, cmap='viridis')
    cbar2 = fig.colorbar(surf, ax=ax2, label='Depth')
    cbar2.ax.tick_params(labelsize=10)
    ax2.set_title('3D Surface Plot of Debris Depth', fontsize=14)
    ax2.set_xlabel('Circumferential Position', fontsize=12)
    ax2.set_ylabel('Frame', fontsize=12)
    ax2.set_zlabel('Depth', fontsize=12)
    ax2.tick_params(axis='both', which='major', labelsize=10)

    # 3. Contour Plot
    ax3 = fig.add_subplot(233)
    contour = ax3.contourf(x, y, depth_array, levels=20, cmap='viridis')
    cbar3 = fig.colorbar(contour, ax=ax3, label='Depth')
    cbar3.ax.tick_params(labelsize=10)
    ax3.set_title('Contour Plot of Debris Depth', fontsize=14)
    ax3.set_xlabel('Circumferential Position', fontsize=12)
    ax3.set_ylabel('Frame', fontsize=12)
    ax3.tick_params(axis='both', which='major', labelsize=10)

    # 4. Line Plots
    ax4 = fig.add_subplot(234)
    ax5 = fig.add_subplot(235)
    
    # Average depth across frames for each circumferential position
    avg_depth_circ = np.mean(depth_array, axis=0)
    ax4.plot(x, avg_depth_circ)
    ax4.set_title('Avg Depth Across Frames\nfor Each Circ Position', fontsize=14)
    ax4.set_xlabel('Circumferential Position', fontsize=12)
    ax4.set_ylabel('Average Depth', fontsize=12)
    ax4.tick_params(axis='both', which='major', labelsize=10)
    
    # Average depth across circumferential positions for each frame
    avg_depth_frame = np.mean(depth_array, axis=1)
    ax5.plot(y, avg_depth_frame)
    ax5.set_title('Avg Depth Across Circ Positions\nfor Each Frame', fontsize=14)
    ax5.set_xlabel('Frame', fontsize=12)
    ax5.set_ylabel('Average Depth', fontsize=12)
    ax5.tick_params(axis='both', which='major', labelsize=10)

    # 5. Box Plot
    ax6 = fig.add_subplot(236)
    sns.boxplot(data=depth_array, ax=ax6)
    ax6.set_title('Distribution of Depth Values\nfor Each Circ Position', fontsize=14)
    ax6.set_xlabel('Circumferential Position', fontsize=12)
    ax6.set_ylabel('Depth', fontsize=12)
    ax6.set_xticks(np.arange(0, depth_array.shape[1], depth_array.shape[1]//5))
    ax6.set_xticklabels([str(i + flaw_circ_start) for i in range(0, depth_array.shape[1], depth_array.shape[1]//5)], rotation=45, ha='right')
    ax6.tick_params(axis='both', which='major', labelsize=10)

    plt.tight_layout()
    return fig

def plot_flaw_peaks(
    bscan_data_frame: List[List[float]],
    potential_flaw_peaks: List[Tuple[int, float]],
    flaw_position: int
) -> plt.Figure:
    """
    Plot the flaw signal and print detected peaks.

    Parameters
    ----------
    bscan_data_frame : List[List[float]]
        B-scan data frame.
    potential_flaw_peaks : List[Tuple[int, float]]
        List of potential flaw peaks.
    flaw_position : int
        Position of the flaw in the B-scan.

    Returns
    -------
    plt.Figure
        The matplotlib Figure object containing the plot.
    """
    fig, ax = plt.subplots(figsize=(12, 6))

    # Plot flaw signal
    ax.plot(bscan_data_frame[flaw_position], color='green')
    ax.set_title(f'Flaw Position A-scan (position: {flaw_position})')
    ax.set_xlabel('A-scan index')
    ax.set_ylabel('Amplitude')
    
    for i, (peak_index, peak_amplitude) in enumerate(potential_flaw_peaks):
        ax.axvline(x=peak_index, color=f'C{i}', linestyle='--', 
                   label=f'Potential flaw peak {i+1} (index: {peak_index}, amplitude: {peak_amplitude:.2f})')
    ax.legend()

    plt.tight_layout()

    # Print detected peaks
    print("\nPotential flaw peaks (index, amplitude):")
    for i, peak in enumerate(potential_flaw_peaks):
        print(f"Peak {i+1}: {peak}")

    return fig
# if __name__ == '__main__':
    # Usage example
    
    # bscan_data = probes_data['NB1'].data
    # depth, ref_peak, flaw_peak = calculate_depth(major_peaks, potential_flaw_peaks)

  
    # visualize_depth_array(depth_array, flaw.x_start, flaw.y_start)