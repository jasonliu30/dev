from pathlib import Path
import pandas as pd
import numpy as np
from b_scan_reader.BScan import BScan
from b_scan_reader import bscan_structure_definitions as bssd
from utils.folder_utils import mkdir
import cv2
from utils import datamap_utils
from datamap_utils import FlawLocation
from matplotlib import pyplot as plt
from matplotlib import patches
import math

files_to_save = ['tof', 'tof_flat', 'cors', 'cors_whole', 'amp', 'average_amplitudes', 'area_under_curve', 'lags_whole', 'lags', 'cors_g2g3', 'stddev_before_wg',
         'stddev_in_wg', 'stddev_after_wg', 'stddev_full_scan']

def flaw_coords_pyplot(flaw: FlawLocation) -> tuple[tuple[float, float], float, float]:
    """
    Converts FlawLocation coordinates to a coord style required by plt (left, bottom), width, height.

    Args:
        flaw: current flaw

    Returns:
        Flaw coordinates in the style expected by pyplot for drawing a rectangle - (left, bottom), width, height
    """
    left = flaw.rotary_start
    width = flaw.rotary_end - flaw.rotary_start

    bottom = flaw.frame_start
    height = flaw.frame_end - flaw.frame_start

    return (left, bottom), width, height


def nice_number(value, round_=False) -> float:
    """Converts any number into a 'nice' number.

    Args:
        value: value to be rounded
        round_: whether to round

    Returns:
        nice number
    """
    exponent = math.floor(math.log(value, 10))
    fraction = value / 10 ** exponent

    if round_:
        if fraction < 1.5:
            nice_fraction = 1.
        elif fraction < 3.:
            nice_fraction = 2.
        elif fraction < 7.:
            nice_fraction = 5.
        else:
            nice_fraction = 10.
    else:
        if fraction <= 1:
            nice_fraction = 1.
        elif fraction <= 2:
            nice_fraction = 2.
        elif fraction <= 5:
            nice_fraction = 5.
        else:
            nice_fraction = 10.

    return nice_fraction * 10 ** exponent


def nice_bounds(axis_start, axis_end, num_ticks=10):
    """Calculates axis ticks with guaranteed minimum number of ticks.
    
    Args:
        axis_start: Actual start value of the axis / bound
        axis_end: Actual end value of the axis / bound
        min_ticks: Minimum desired number of ticks
        
    Returns:
        Tuple in the form (nice_axis_start, nice_axis_end, nice_tick_width)
    """
    min_ticks=10
    axis_width = max(axis_start, axis_end) - min(axis_start, axis_end)
    
    if axis_width == 0:
        return axis_start, axis_end, 0
    
    # Try decreasing powers of nice numbers until we get enough ticks
    for factor in [1.0, 0.5, 0.25, 0.2, 0.1, 0.05, 0.025, 0.01]:
        nice_range = nice_number(axis_width)
        nice_tick = nice_number(nice_range / (min_ticks - 1), round_=True) * factor
        
        # Calculate how many ticks this would give
        potential_ticks = math.floor(axis_width / nice_tick) + 1
        
        if potential_ticks >= min_ticks:
            # We found a good tick size
            nice_start = math.floor(axis_start / nice_tick) * nice_tick
            nice_end = math.ceil(axis_end / nice_tick) * nice_tick
            return nice_start, nice_end, nice_tick
    
    # If we get here, just force a division of the range
    nice_tick = axis_width / (min_ticks - 1)
    return axis_start, axis_end, nice_tick


def nearest(array: np.ndarray, value):
    """
    Returns the index of the element in the array closest to the supplied value.
    Args:
        array: numpy array
        value (int | float): closest value

    Returns:
        int: index of the element that is closest to value
    """
    idx = (np.abs(array - value)).argmin()
    nearest_value = array[idx]
    return idx


def make_x_axis(axis):
    """
    Creates the x-axis / rotary axis for a CScan.

    Args:
        axis (np.ndarray): rotary axis of the CScan

    Returns
        - indices - Indices along the X-Axis to place tick marks
        - Values - Values, in degrees, of the corresponding tick marks

    """
    start = axis[0]
    end = axis[-1]
    if axis[0] > axis[-1]:
        end += 360

    axis_start, axis_end, tick = nice_bounds(start, end, num_ticks=18)
    values = np.arange(axis_start, axis_end, tick/2)
    values = np.append(values, axis_end)
    values %= 360

    indices = np.zeros(len(values))
    for i in range(len(values)):
        indices[i] = nearest(axis, values[i])

    return indices, values


def make_y_axis(axis):
    """
    Creates the y-axis / linear axis for a CScan.

    Args:
        axis (np.ndarray): linear axis of the CScan

    Returns
        - indices - Indices along the Y-Axis to place tick marks
        - Values - Values, in mm, of the corresponding tick marks

    """
    axis_start, axis_end, tick = nice_bounds(axis[0], axis[-1], num_ticks=12)
    values = np.arange(axis_start, axis_end, tick)
    values = np.append(values, axis_end)

    indices = np.zeros(len(values))
    for i in range(len(values)):
        indices[i] = nearest(axis, values[i])

    return indices, values


def plot_c_scan(c_scan: np.ndarray, flaws, axes: bssd.BScanAxis, title, flaw_ids):
    """
    Plots a CScan using matplotlib. Draws the axes and overlays flaw locations.

    Args:
        c_scan: CScan to plot
        flaws: List of flaws that exist in the CScan
        axes: BScanAxis of the currently loaded BScan File
        title: Title to be added to the CScan plot. Usually the filename.

    Returns:
        fig - matplotlib figure, so the figure can either be displayed or saved.
    """
    plt.ioff()
    ratio = len(axes.axial_pos) / len(axes.rotary_pos)
    
    if len(axes.rotary_pos) < 1000 and len(flaws) < 30:
        plt_size = 23
    elif len(flaws) > 100:
        plt_size = 100
    elif len(flaws) > 50:
        plt_size = 60
    else:
        plt_size = 35

    fig = plt.figure(figsize=(plt_size, plt_size*ratio + 1))
    plt.imshow(c_scan)
    plt.title(title)

    if len(flaws) > len(flaw_ids):
        extra_length = len(flaws) - len(flaw_ids)
        flaw_ids.extend([""] * extra_length)

    for flaw, flaw_id in zip(flaws, flaw_ids):
        overlay_flaw(fig, flaw, flaw_id)
    plt.tight_layout()

    plt.xlabel('circumferencial location (deg)')
    plt.grid(linestyle='-.')
    x_indices, x_values = make_x_axis(axes.rotary_pos)
    plt.xticks(x_indices, x_values)

    plt.ylabel('axial location (mm)')
    y_indices, y_values = make_y_axis(axes.axial_pos)
    plt.yticks(y_indices, y_values)
    plt.tight_layout()

    return fig


def overlay_flaw(fig: plt.Figure, flaw: FlawLocation, flaw_id):
    """
    Overlays flaw on the figure
    Args:
        fig: matplotlib figure
        flaw: flaw to be overlaid, with datatype FlawLocation

    Returns:
        None
    """
    plt.ioff()
    color = 'red' if flaw.dr == 'D' else 'green'

    (x, y), width, height = flaw_coords_pyplot(flaw)

    ax = fig.gca()

    rect = patches.Rectangle((x, y), width, height, fill=False, color=color)
    ax.add_artist(rect)
    if flaw.dr == 'D':
        if "merged into" in flaw_id:
            ax.text(flaw.rotary_start, flaw.frame_start, ' ', horizontalalignment='left', verticalalignment='bottom', color=color)
        else:
            ax.text(flaw.rotary_start, flaw.frame_start, flaw_id + " " + flaw.name, 
                   horizontalalignment='left', 
                   verticalalignment='bottom', 
                   color=color,
                   fontsize=7,  # Half of default size (10)
                   bbox=dict(
                       facecolor='white', 
                       alpha=0.7, 
                       edgecolor=color, 
                       boxstyle='round,pad=0.07',  # Half of the default padding
                       mutation_scale=7  # Half of default scale (typically 10)
                   ),
                   zorder=100)
    else:
        ax.text(flaw.rotary_start, flaw.frame_end, flaw.name, 
               horizontalalignment='left', 
               verticalalignment='top', 
               color=color,
               fontsize=7,  # Half of default size
               bbox=dict(
                   facecolor='white', 
                   alpha=0.7, 
                   edgecolor=color, 
                   boxstyle='round,pad=0.07',  # Half of the default padding
                   mutation_scale=7  # Half of default scale
               ),
               zorder=100)

def save_c_scans_as_csv(c_scans: dict, save_path: Path, probe: bssd.Probe):
    """
    Saves the dictionary of C Scan results to CSVs

    Args:
        c_scans (dict): output of Autoanalysis
        save_path (Path): Path pointing to the folder to save the csvs to
        probe (bssd.Probe): Probe. The probe name will be prepended to the filenames

    Returns:
        None

    """
    mkdir(save_path)
    debug_couldnt_save = []
    for name, c_scan in c_scans.items():
        try:
            filename = f"{probe.name}_{name}.csv"
            pd.DataFrame(c_scan).to_csv(save_path / filename, index=False, header=False)
        except:
            debug_couldnt_save.append(name)


def save_header_to_csv(b_scan: BScan, save_path: Path):
    """
    Saves the header of the provided BScan to csv

    Args:
        b_scan (BScan): BScan file to take the header from
        save_path (Path): Path pointing to the folder to save the csvs to

    Returns:
        None

    """
    mkdir(save_path)
    header_text = b_scan.get_header_text()
    location_info = []

    for i in range(1, len(header_text)):
        row = header_text[i].split(",")
        row1 = [x.replace(" ", "") for x in row]
        location_info.append(row1)

    header_loc = pd.DataFrame(location_info, dtype=np.dtype("float"))
    header_loc = header_loc.round(2)
    header_loc.iloc[:, 0:3].to_csv(save_path / "header_loc.csv", index=False, header=False)


def save_flaws_to_csv(all_flaws: pd.DataFrame, save_path: Path):
    """
    Saves the all_flaws datamap to the provided path

    Args:
        all_flaws:
        save_path:

    Returns:

    """
    mkdir(save_path)
    all_flaws.to_csv(save_path / "rows_with_flaw.csv", index=False)


def save_flaw_locations_to_csv(flaw_locations, save_path: Path, probe: bssd.Probe):
    """
    Saves the flaw locations to CSV
    Args:
        flaw_locations:
        save_path:
        probe:

    Returns:

    """
    mkdir(save_path)
    column_names = ["Flaw Type", "AX1", "AX2", "RO1", "RO2", "DR"]
    flaw_loc_df = pd.DataFrame(flaw_locations, columns=column_names)
    flaw_loc_df.to_csv(save_path / f"flaw_loc_{probe.name}.csv", index=False)


def normalize(data):
    if np.min(data) == np.max(data):
        normalized = data / np.min(data)
    else:
        normalized = (data - np.min(data)) / (np.max(data) - np.min(data))
    return normalized


def save_c_scans(c_scans: dict, save_name: str, save_path: Path, probe_name: str):
    """
    Saves the raw CScan to save_path/original-cscans, and saves a copy of the CScan with overlays & axes to save_path.

    Args:
        c_scans: Dictionary of CScans
        save_name: name that will be used for the filename and the plot title
        save_path: root save directory
        probe_data: BScanData() class of the current probe

    Returns:
        None
    """

    for name, c_scan in c_scans.items():
        if name.lower() in files_to_save:
            filename = f'{save_name}-{name}-{probe_name}.png'
            # Save the un-marked up CScan
            image = normalize(c_scan) * 255
            cv2.imwrite(str(save_path / filename), image)
            
# if __name__ == "__main__":  # pragma: no cover
#     # Run with example code
#     import pickle
#     flaws = [['Ind 196_Debris', 15, 18, 84, 97, 'D'], ['Ind 197_CC', 49, 64, 663, 679, 'R'], ['Ind 199_CC', 54, 65, 761, 778, 'R'], ['Ind 200_CC (M)', 58, 83, 854, 898, 'D'],
#      ['Ind 198_CC', 63, 76, 738, 764, 'R'], ['Ind 6_ID Scratch', 0, 97, 649, 668, 'R']]
#
#     with open('..\\probe_axes.pkl', 'rb') as f:
#         axes = pickle.load(f)
#
#     for i in range(len(flaws)):
#         flaws[i] = FlawLocation(*flaws[i])
#
#     c_scan = pd.read_csv(r'C:\gitrepo\OPG-Auto-Analysis\C_scan_results\1555-M05-P1861-[A8178-8216][R1060-2450] (Debris)\csv-results\CPC_ToF.csv', header=None).values
#
#     fig = plot_c_scan(c_scan, flaws, axes)


# if __name__ == '__main__':  # pragma: no cover
#     import pickle
#     save_path = Path(r'C:\gitrepo\OPG-Auto-Analysis\auto-analysis-results\BSCAN Type A  M-05 Pickering B Unit-6 west 12-Feb-2018 055830 [A8178-8216][R1060-2450] - Copy')
#     names = ['flaw_locations', 'c_scans', 'save_path', 'probe_data', 'flaw_locations']
#     vardict = {}
#     for idx, var in enumerate(names):
#         fn = names[idx] + '.pkl'
#         with open(save_path / fn, 'rb') as f:
#             vardict[var] = pickle.load(f)
#     save_c_scans(vardict['c_scans'], save_path, vardict['probe_data'], vardict['flaw_locations'])