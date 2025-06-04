from typing import List, Tuple
import itertools
import pandas as pd

def closest_to_0_or_360(rotary_position: float) -> int:
    """
    Determines if a given rotary position is closer to 0 or 360 degrees.
    
    Parameters:
    ----------
    rotary_position : float
        The rotary position in degrees to evaluate.

    Returns:
    -------
    int
        Returns 0 if the position is closer to 0 degrees, otherwise returns 360.
    """
    return 0 if rotary_position % 360 < 180 else 360

def merge(config, auto_sizing_summary: pd.DataFrame) -> pd.DataFrame:
    """
    Merges rows in the DataFrame that represent flaws close to each other based on axial and rotary thresholds
    specified in the config object. It first calculates the 'flaw_axial_end' and 'flaw_rotary_end' for each flaw,
    sorts the DataFrame by 'flaw_rotary_start', and then iterates over all possible pairs of flaws to determine
    if they should be merged based on their axial and rotary closeness.

    Parameters:
    ----------
    config : Config
        Configuration object that holds parameters including axial and rotary thresholds for merging flaws.
        
    auto_sizing_summary : pd.DataFrame
        DataFrame containing flaw information with columns for flaw axial start, flaw length, flaw rotary start, and flaw width.
        
    Returns:
    -------
    pd.DataFrame
        A modified DataFrame after merging close flaws based on the specified thresholds.
    """
    axial_threshold = config.characterization.flaw_merging.axial_threshold
    rotary_threshold_to_0 = config.characterization.flaw_merging.rotary_threshold_to_0
    auto_sizing_summary = auto_sizing_summary.copy()
    auto_sizing_summary['original_order'] = range(len(auto_sizing_summary))
    auto_sizing_summary['flaw_axial_end'] = auto_sizing_summary['flaw_axial_start'] + auto_sizing_summary['flaw_length']
    auto_sizing_summary['flaw_rotary_end'] = auto_sizing_summary['flaw_rotary_start'] + auto_sizing_summary['flaw_width']
    df_sorted = auto_sizing_summary.sort_values(by='flaw_rotary_start').reset_index(drop=True)
    ind_to_drop: List[int] = []
    
    for i1, i2 in itertools.combinations(range(len(df_sorted)), 2):
        # Axial check
        current_start = df_sorted.at[i1, 'flaw_axial_start']
        current_end = df_sorted.at[i1, 'flaw_axial_end']
        next_start = df_sorted.at[i2, 'flaw_axial_start']
        next_end = df_sorted.at[i2, 'flaw_axial_end']
        if abs(next_start - current_start) <= axial_threshold and abs(next_end - current_end) <= axial_threshold and (df_sorted.at[i1, 'flaw_type'] == df_sorted.at[i2, 'flaw_type']):
            
            current_rotary_start = df_sorted.at[i1, 'flaw_rotary_start']
            current_rotary_end = df_sorted.at[i1, 'flaw_rotary_end']
            next_rotary_start = df_sorted.at[i2, 'flaw_rotary_start']
            next_rotary_end = df_sorted.at[i2, 'flaw_rotary_end']
            # Rotary check
            if closest_to_0_or_360(current_rotary_start) == 0:
                if abs(current_rotary_start - 0) <= rotary_threshold_to_0 and abs(next_rotary_end - 360) <= rotary_threshold_to_0:
                    df_sorted, ind_to_drop = merge_flaws_based_on_depth(df_sorted, i1, i2, ind_to_drop, current_rotary_end, next_rotary_start, current_start, next_start)
            elif closest_to_0_or_360(current_rotary_start) == 360:
                if abs(current_rotary_end - 360) <= rotary_threshold_to_0 and abs(next_rotary_start - 0) <= rotary_threshold_to_0:
                    df_sorted, ind_to_drop = merge_flaws_based_on_depth(df_sorted, i2, i1, ind_to_drop, current_rotary_end, next_rotary_start, current_start, next_start)

    cols_to_nan = df_sorted.columns.difference(['scan_name', 'scan_unit', 'scan_station', 'scan_channel', 'scan_axial_pitch', 'flaw_id', 'flaw_type','original_order'])
    df_sorted.loc[ind_to_drop, cols_to_nan] = ''

    # remove the columns that were added for the merging process
    df_sorted = df_sorted.sort_values(by='original_order').drop(columns=['flaw_axial_end', 'flaw_rotary_end', 'original_order'])

    return df_sorted


def merge_flaws_based_on_depth(df_sorted: pd.DataFrame, i1: int, i2: int, ind_to_drop: List[int], 
                               current_rotary_end: float, next_rotary_start: float, 
                               current_start: float, next_start: float) -> Tuple[pd.DataFrame, List[int]]:
    """
    Merges two consecutive flaw entries in a DataFrame based on their depth, retaining the entry with the maximum depth.
    It updates the flaw's width, length, rotary start, and axial start in the DataFrame based on the comparison.
    
    Parameters:
    ----------
    df_sorted : pd.DataFrame
        DataFrame containing flaw entries sorted by some criteria.
        
    i1, i2 : int
        Indices of the flaw entries to be compared and potentially merged.
        
    ind_to_drop : List[int]
        A list that tracks indices of the DataFrame rows to be removed.
        
    current_rotary_end : float
        Rotary end position of the current (first) flaw entry.
        
    next_rotary_start : float
        Rotary start position of the next (second) flaw entry.
        
    current_start : float
        Axial start position of the current (first) flaw entry.
        
    next_start : float
        Axial start position of the next (second) flaw entry.

    Returns:
    -------
    Tuple[pd.DataFrame, List[int]]
        The updated DataFrame and the updated list of indices marked for deletion.
    """
    # Determine which flaw has the greater depth and store relevant info
    if df_sorted.at[i1, 'flaw_depth'] > df_sorted.at[i2, 'flaw_depth']:
        chosen_index, delete_index = i1, i2
    elif df_sorted.at[i1, 'flaw_depth'] < df_sorted.at[i2, 'flaw_depth']:
        chosen_index, delete_index = i2, i1
    else:
        # If there are strings, choose whichever one is a float
        if isinstance(df_sorted.at[i1, 'flaw_depth'], float) and not isinstance(df_sorted.at[i2, 'flaw_depth'], float):
            chosen_index, delete_index = i1, i2
        elif not isinstance(df_sorted.at[i1, 'flaw_depth'], float) and isinstance(df_sorted.at[i2, 'flaw_depth'], float):
            chosen_index, delete_index = i2, i1
        else:
            # Default to i1 if both are floats or neither is a float
            chosen_index, delete_index = i1, i2

    # Update the flaw at index with merged information
    df_sorted.at[chosen_index, 'flaw_rotary_start'] = next_rotary_start
    df_sorted.at[chosen_index, 'flaw_axial_start'] = min(current_start, next_start)
    df_sorted.at[chosen_index, 'flaw_width'] = (current_rotary_end - 0) + 360 - next_rotary_start
    df_sorted.at[chosen_index, 'flaw_length'] = max(df_sorted.at[i1, 'flaw_axial_end'], df_sorted.at[i2, 'flaw_axial_end']) - df_sorted.at[chosen_index, 'flaw_axial_start']
    df_sorted.at[chosen_index, 'frame_start'] = min(df_sorted.at[i1, 'frame_start'], df_sorted.at[i2, 'frame_start'])
    df_sorted.at[chosen_index, 'frame_end'] = max(df_sorted.at[i1, 'frame_end'], df_sorted.at[i2, 'frame_end'])
    df_sorted.at[chosen_index, 'confidence'] = (df_sorted.at[chosen_index, 'confidence'] + df_sorted.at[delete_index, 'confidence']) / 2
    df_sorted.at[delete_index, 'flaw_id'] = df_sorted.at[delete_index, 'flaw_id'] + ' (merged into ' + df_sorted.at[chosen_index, 'flaw_id'] + ')'

    ind_to_drop.append(delete_index)
    return df_sorted, ind_to_drop