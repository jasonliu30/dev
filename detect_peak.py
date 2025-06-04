import numpy as np
from scipy.signal import find_peaks
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

def detect_all_peaks(
    left_surface: np.ndarray,
    right_surface: np.ndarray,
    flaw_signal: np.ndarray,
    config: Dict
) -> Tuple[Dict[str, List[Tuple[int, float]]], List[Tuple[int, float]]]:
    """
    Detect major peaks in reference signals within a limited range and all significant peaks in the flaw signal.

    Parameters
    ----------
    left_surface : np.ndarray
        The left surface signal.
    right_surface : np.ndarray
        The right surface signal.
    flaw_signal : np.ndarray
        The flaw signal.
    config : Dict
        Configuration dictionary containing peak detection parameters.

    Returns
    -------
    Tuple[Dict[str, List[Tuple[int, float]]], List[Tuple[int, float]]]
        A tuple containing:
        - A dictionary with 'left' and 'right' keys, each containing a list of (index, amplitude) tuples for surface peaks.
        - A list of (index, amplitude) tuples for potential flaw peaks.
    """
    # Extract configuration parameters
    surface_window = config.surface_window
    num_flaw_peaks = config.num_flaw_peaks
    min_amplitude = config.min_amplitude
    ratio_to_max = config.ratio_to_max

    def find_surface_peaks(signal: np.ndarray, window: int) -> List[Tuple[int, float]]:
        main_peak = np.argmax(signal)
        start = max(0, main_peak - window)
        end = min(len(signal), main_peak + window + 1)
        window_signal = signal[start:end]
        
        # Find the highest peak within the window
        highest_peak = start + np.argmax(window_signal)
        
        # Find additional peaks
        peaks, _ = find_peaks(window_signal, height=max(window_signal.max()*0.5, min_amplitude), distance=5, prominence=10)
        peaks += start  # Adjust peak indices to original signal
        
        # Combine highest peak with other peaks, remove duplicates, and sort by amplitude
        all_peaks = sorted(set([highest_peak] + list(peaks)), key=lambda x: signal[x], reverse=True)
        
        # Filter peaks below min_amplitude
        filtered_peaks = [(int(peak), float(signal[peak])) for peak in all_peaks if signal[peak] >= min_amplitude]
        
        return filtered_peaks[:3]

    # Find the major peaks in the reference signals
    major_peaks = {
        'left': find_surface_peaks(left_surface, surface_window),
        'right': find_surface_peaks(right_surface, surface_window)
    }

    # Find peaks in the flaw signal
    normalized_flaw_signal = flaw_signal - 128
    max_normalized_amplitude = np.max(np.abs(normalized_flaw_signal))
    threshold_normalized = max(max_normalized_amplitude * ratio_to_max, min_amplitude - 128)
    threshold = threshold_normalized + 128  # Convert back to original scale
    peaks, _ = find_peaks(flaw_signal, height=threshold, distance=5, prominence=1)

    # Sort peaks by amplitude, filter those below min_amplitude, and return the top num_flaw_peaks
    potential_flaw_peaks = sorted(
        [(int(peak), float(flaw_signal[peak])) for peak in peaks if flaw_signal[peak] >= min_amplitude],
        key=lambda x: x[1],
        reverse=True
    )[:num_flaw_peaks]

    return major_peaks, potential_flaw_peaks


def plot_surface_and_flaw_peaks(
    left_surface: List[float],
    right_surface: List[float],
    bscan_data_frame: List[List[float]],
    major_peaks: Dict[str, List[Tuple[int, float]]],
    potential_flaw_peaks: List[Tuple[int, float]],
    flaw_position: int
) -> None:
    """
    Plot the left and right surfaces, flaw signal, and print detected peaks.

    Parameters
    ----------
    left_surface : List[float]
        Data for the left surface.
    right_surface : List[float]
        Data for the right surface.
    bscan_data_frame : List[List[float]]
        B-scan data frame.
    major_peaks : Dict[str, List[Tuple[int, float]]]
        Dictionary containing major peaks for 'left' and 'right' surfaces.
    potential_flaw_peaks : List[Tuple[int, float]]
        List of potential flaw peaks.
    flaw_position : int
        Position of the flaw in the B-scan.

    Returns
    -------
    None
    """
    fig, (ax1, ax2, ax3) = plt.subplots(3, 1, figsize=(12, 15), sharex=True)

    def plot_surface(ax, data, title, color, peaks):
        ax.plot(data, color=color)
        ax.set_title(title)
        ax.set_ylabel('Amplitude')
        for i, (peak_index, peak_amplitude) in enumerate(peaks):
            ax.axvline(x=peak_index, color=f'C{i+1}', linestyle='--', 
                       label=f'Peak {i+1} (index: {peak_index}, amplitude: {peak_amplitude:.2f})')
        ax.legend()

    plot_surface(ax1, left_surface, 'Left Surface', 'blue', major_peaks['left'])
    plot_surface(ax2, right_surface, 'Right Surface', 'red', major_peaks['right'])

    # Plot flaw signal
    ax3.plot(bscan_data_frame[flaw_position], color='green')
    ax3.set_title(f'Flaw Position A-scan (position: {flaw_position})')
    ax3.set_xlabel('A-scan index')
    ax3.set_ylabel('Amplitude')
    for i, (peak_index, peak_amplitude) in enumerate(potential_flaw_peaks):
        ax3.axvline(x=peak_index, color=f'C{i+1}', linestyle='--', 
                    label=f'Potential flaw peak {i+1} (index: {peak_index}, amplitude: {peak_amplitude:.2f})')
    ax3.legend()

    plt.tight_layout()
    plt.show()

    # Print detected peaks
    def print_peaks(title, peaks):
        print(f"\n{title}:")
        for i, peak in enumerate(peaks):
            print(f"Peak {i+1}: {peak}")

    print_peaks("Major peaks in left surface (index, amplitude)", major_peaks['left'])
    print_peaks("Major peaks in right surface (index, amplitude)", major_peaks['right'])
    print_peaks("Potential flaw peaks (index, amplitude)", potential_flaw_peaks)

# if __name__ == '__main__':
#     # Usage example
#     major_peaks, potential_flaw_peaks = detect_all_peaks(left_surface, right_surface, bscan_data_frame[flaw_position, :])

#     plot_surface_and_flaw_peaks(
#     left_surface,
#     right_surface,
#     bscan_data_frame,
#     major_peaks,
#     potential_flaw_peaks,
#     flaw_position
# )
    