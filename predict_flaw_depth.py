from re import X
from b_scan_reader import bscan_structure_definitions as bssd
import numpy as np
import pandas as pd
from depth_sizing.CC_depth_sizing import pred_CC_depth
from depth_sizing.debris_depth_sizing import pred_debris_depth
from depth_sizing.fbbpf_depth_sizing import pred_FBBPF_depth
from depth_sizing.ax_circ_scrape_depth_sizing import pred_ax_circ_scrape_depth
from utils.logger_init import create_dual_loggers
from scan_data_structure import Scan
from typing import Dict, List, Tuple, Optional, Any
from tqdm import tqdm

# create loggers
dual_logger, file_logger  = create_dual_loggers()

all_flaw_type_probes = {'CC' : 'pc', 'Debris' : 'nb',
                        'FBBPF' : 'nb', 'BM_FBBPF' : 'nb',
                        'Axial_Scrape' : 'nb', 'Circ_Scrape' : 'nb',
                          'Others' : 'nb', "ID_scratch" : 'nb'}

def process_auto_sizing_summary(auto_sizing_summary, bscan):
    """
    This function will process the auto sizing summary dataframe to be in the appropriate format for depth sizing
    Parameters
    ----------
        auto_sizing_summary : dataframe 
        bscan : BScan object
    Returns
    -------
        df2 : dataframe with columns ['Ax Start', 'Ax End', 'Ro Start', 'Ro End', 'Indication', 'ID', 'Filename','flaw_type', 'possible_category']
    """
    bscan_axes = bscan.get_channel_axes(bssd.Probe['APC'])
    df_col = ['Ax Start', 'Ax End', 'Ro Start', 'Ro End',
              'Indication', 'Filename', 'flaw_type', 'possible_category',
              'Channel']
    df = auto_sizing_summary
    df2 = df[['frame_start', 'frame_end', 'flaw_rotary_start', 'flaw_width',
              'flaw_id', 'scan_name', 'flaw_type', 'possible_category',
              'scan_channel']]
        
    rows_to_drop = []
    for index, row in df2.iterrows():

        matched_indices = np.where(np.array([round(i,1) for i in bscan_axes.rotary_pos]) == round(float(row['flaw_rotary_start']),1))[0]
        if len(matched_indices) > 0:
            matched_index = matched_indices[0]
            df2.at[index, 'flaw_rotary_start'] = matched_index
            df2.at[index, 'flaw_width'] = (matched_index + row['flaw_width'] * 10)
        else:
            dual_logger.info(f"Warning: No match found for row {index} with flaw_rotary_start {row['flaw_rotary_start']}. Excluding this row.")
            rows_to_drop.append(index)
    df2.drop(rows_to_drop, inplace=True)
    df2.columns = df_col
    df2['Ro Start'] = df2['Ro Start'].astype(int)
    df2['Ro End'] = df2['Ro End'].astype(int)
    df2['Outage Number'] = df['scan_station'].str[0] + str(bscan.get_header().Year) + df['scan_unit'].astype('str') + '1'
    return df2.reset_index(drop=True)

def modify_flaw_depth_poss_cat(dict_data):
    """
    Modifies the flaw depth possibility category based on the values in the dictionary.
    Parameters:
    ----------
    dict_data (dict): Dictionary containing depth values.
    Returns:
    -------
    str: Modified depth possibility category or NaN.
    """
    
    if dict_data:
        value_list = [float(value.split()[0]) if isinstance(value, str) else value 
                      for value in dict_data.values()]
        flagged_case = any([True if '(2)' in str(value) else False 
                      for value in dict_data.values()])
        modified_value = "< 0.1 (1)(2)" if flagged_case else "< 0.1 (1)"
        
        return  modified_value if all(value < 0.1 for value in value_list) else "see possible category flaw depth"
    return np.nan

def modify_flaw_depth(depth):
    """
    Modifies the flaw depth value.
    Parameters:
    ----------
    depth (str or float): Original depth value.
    Returns:
    -------
    str or float: Modified depth value.
    """

    if isinstance(depth, str):
        if depth == 'nan (2)':
            return 'See note_2'
        try:
            return "< 0.1 (2)" if float(depth.split()[0]) < 0.1 else depth
        except ValueError:
            return depth

    return "< 0.1" if float(depth) < 0.1 else depth

def dict_to_sentence(dict_data):
    """
    Converts a dictionary into a readable string of sentences.
    Each key-value pair in the dictionary is translated into a sentence 
    Parameters:
    ----------
        dict_data (dict): The dictionary to convert. Keys are interpreted as the flaw type
                        and values as the predicted depth.
    Returns:
    -------
        str: A string representation of the dictionary, with each key-value pair 
            converted into a sentence.
    """
    # Create a list to hold all sentences
    sentences = []
    
    # Loop over the items in the dictionary
    for key, value in dict_data.items():
        # Convert each key-value pair to a sentence and append to the list
        sentences.append(f"{key}: {value}.")
    
    # Join all sentences with a space and return the resulting string
    return " ".join(sentences)

def pred_depth(scan_type, type_df, flaw_type, depth_method, cscans_output, probes_data, config, save_path, run_name, output_root):
     # Helper function to extract float from string like "0.11 (2)"
    def extract_float(value):
        if isinstance(value, str):
            return float(value.split()[0])
        return float(value)
    results_df = pd.DataFrame()
    if scan_type == 'bscan':
        if depth_method == 'nb_debris':
            results_df, _, stats = pred_debris_depth(df = type_df,
                                            b_scans = [probes_data['NB1'].data, probes_data['NB2'].data],
                                            NB_lags_whole = [cscans_output[2]['lags_whole'], cscans_output[3]['lags_whole']],
                                            NB_lags_g4 = [cscans_output[2]['lags'], cscans_output[3]['lags']],
                                            conf = config.sizing.Debris.__dict__,
                                            probes = ['NB1', 'NB2'], run_name = run_name,
                                            save_files=False, plotting=False,save_location=save_path, out_root=output_root)
        elif depth_method == 'nb_fbbpf':
            results_df, _, flag_nb1 = pred_FBBPF_depth(config.sizing.FBBPF.__dict__,
                                    type_df,
                                    cscans_output[2]['lags_whole'],
                                    cscans_output[2]['lags'],
                                    probes_data['NB1'].data,
                                    run_name = run_name,
                                    save_files=False,
                                    plotting=False,save_location=save_path, out_root=output_root)
            results_df2, _, flag_nb2 = pred_FBBPF_depth(config.sizing.FBBPF.__dict__,
                                        type_df,
                                        cscans_output[3]['lags_whole'],
                                        cscans_output[3]['lags'],
                                        probes_data['NB2'].data,
                                        run_name = run_name,
                                        save_files=False,
                                        plotting=False,save_location=save_path, out_root=output_root)
            
            # Extract float values for comparison
            depth1 = extract_float(results_df['flaw_depth'][0])
            depth2 = extract_float(results_df2['flaw_depth'][0])
            
            # Add the depth values as new columns to results_df
            results_df['nb1_depth'] = depth1
            results_df['nb2_depth'] = depth2
            
            # Compare depths and take the larger value
            if depth2 > depth1:
                # Use results from NB2 as they show a deeper flaw
                results_df = results_df2.copy()
                results_df['nb1_depth'] = depth1
                results_df['nb2_depth'] = depth2    
                results_df['probe_used'] = "NB2"
            else:
                # Keep original results_df and stats (from NB1)
                results_df['probe_used'] = "NB1"
        elif depth_method == 'pc_cc':
            results_df, _, stats = pred_CC_depth(config.sizing.CC.__dict__, type_df, cscans_output[0]['lags_whole'], cscans_output[0]['cors'],
                                    cscans_output[0]['lags'], cscans_output[0]['wavegroups'], probes_data['APC'].data, run_name = run_name,
                                    save_location=save_path, save_name='flaw', save_files=True, plotting=False, out_root=output_root)
            
            results_df_c, _, stats = pred_CC_depth(config.sizing.CC.__dict__, type_df, cscans_output[1]['lags_whole'], cscans_output[1]['cors'],
                                    cscans_output[1]['lags'], cscans_output[1]['wavegroups'], probes_data['CPC'].data, run_name = run_name,
                                    save_location=save_path, save_name='flaw', save_files=True, plotting=False, out_root=output_root)
             # Extract float values for comparison
            depth1 = extract_float(results_df['flaw_depth'][0])
            depth2 = extract_float(results_df_c['flaw_depth'][0])

            # Add the depth values as new columns to results_df
            results_df['apc_depth'] = depth1
            results_df['cpc_depth'] = depth2
            # Compare depths and take the larger value
            if depth2 > depth1:
                # Use results from NB2 as they show a deeper flaw
                results_df = results_df_c.copy()
                results_df['apc_depth'] = depth1
                results_df['cpc_depth'] = depth2  
                results_df['probe_used'] = "CPC"
            else:
                results_df['probe_used'] = "APC"
                
    elif scan_type == 'cscan':
        if depth_method == 'nb_other':
            results_df  = pred_ax_circ_scrape_depth(type_df,
                                            cscans_output,
                                            probes_data,
                                            cscans_output[2]['lags_whole'],
                                            cscans_output[2]['lags'],
                                            config.sizing.__dict__,
                                            flaw_type,
                                            run_name,
                                            save_files = False,
                                            save_location=save_path,
                                            cscan_type = 'NB1',
                                            out_root=output_root
                                            )
        elif depth_method == 'pc_other':
            type_df.to_excel('test.xlsx', index=False)
            results_df  = pred_ax_circ_scrape_depth(type_df,
                                            cscans_output,
                                            probes_data,
                                            cscans_output[0]['lags_whole'],
                                            cscans_output[0]['lags'],
                                            config.sizing.__dict__,
                                            flaw_type,
                                            run_name,
                                            save_files = False,
                                            save_location=save_path,
                                            cscan_type = 'APC',
                                            out_root=output_root
                                            )
            
            if not results_df.empty:
                depth1 = extract_float(results_df['flaw_depth'][0])
                results_df['apc_depth'] = depth1
                results_df['probe_used'] = "APC"
                
    return results_df


def pred_flaw_depth_for_type(df, flaw_type, cscans_output, probes_data, config, save_path, run_name, output_root):
    """
    Predict flaw depth for a given flaw type using appropriate probes.
    Always uses both primary and secondary methods when available.
    
    Args:
        df (pd.DataFrame): DataFrame containing flaw data
        flaw_type (str): Type of flaw to process
        cscans_output (list): List of dictionaries with C-scan outputs
        probes_data (dict): Dictionary containing probe data
        config (object): Configuration object
        save_path (str): Path to save results
        run_name (str): Name of the analysis run
        output_root (str): Root directory for outputs
        
    Returns:
        pd.DataFrame: DataFrame containing depth prediction results
    """
    # Filter rows of appropriate type - both exact match and in possible categories
    type_df = df[((df['flaw_type'] == flaw_type) | df['possible_category'].apply(lambda x: flaw_type in x))]
    
    if type_df.empty:
        return pd.DataFrame()  # Return empty DataFrame if no matching flaws
    
    # Configuration for different flaw types
    flaw_config = {
        'CC': {
            'primary': {'scan_type': 'bscan', 'method': 'pc_cc', 'depth_col': 'flaw_depth_pc'},
            'secondary': None,  # No secondary for CC
        },
        'Debris': {
            'primary': None,  # No primary for Debris
            'secondary': {'scan_type': 'bscan', 'method': 'pc_cc', 'depth_col': 'flaw_depth_pc'}
        },
        'FBBPF': {
            'primary': {'scan_type': 'bscan', 'method': 'nb_fbbpf', 'depth_col': 'flaw_depth_nb'},
            'secondary': {'scan_type': 'cscan', 'method': 'pc_other', 'depth_col': 'flaw_depth_pc'}
        },
        'BM_FBBPF': {
            'primary': {'scan_type': 'bscan', 'method': 'nb_fbbpf', 'depth_col': 'flaw_depth_nb'},
            'secondary': {'scan_type': 'bscan', 'method': 'pc_cc', 'depth_col': 'flaw_depth_pc'}
        },
        'scrape_or_other': {
            'primary': {'scan_type': 'cscan', 'method': 'nb_other', 'depth_col': 'flaw_depth_nb'},
            'secondary': {'scan_type': 'cscan', 'method': 'pc_other', 'depth_col': 'flaw_depth_pc'}
        }
    }
    
    # Handle scrape and other flaw types with the same config
    if flaw_type in ['Axial_Scrape', 'Circ_Scrape', 'Others', 'ID_scratch']:
        config_key = 'scrape_or_other'
    else:
        config_key = flaw_type
    
    # Get configuration for this flaw type
    config_entry = flaw_config.get(config_key)
    if not config_entry:
        return pd.DataFrame()  # Return empty DataFrame if flaw_type not supported
    
    results_df = pd.DataFrame()
    
    # Process primary probe measurement if available
    if config_entry['primary']:
        primary = config_entry['primary']
        try:
            results_df = pred_depth(
                primary['scan_type'], type_df, flaw_type, primary['method'],
                cscans_output, probes_data, config, save_path, run_name, output_root
            )
            results_df[primary['depth_col']] = results_df['flaw_depth']
        except Exception as e:
            results_df = pd.DataFrame()
            print(f"Error in primary sizing for {flaw_type}: {str(e)}")
            file_logger.error(f'Error in sizing depth for {flaw_type} using primary method: {e}', exc_info=True)
    
    # Process secondary probe measurement if applicable
    if config_entry['secondary']:
        secondary = config_entry['secondary']
        try:
            secondary_df = pred_depth(
                secondary['scan_type'], type_df, flaw_type, secondary['method'],
                cscans_output, probes_data, config, save_path, run_name, output_root
            )
            secondary_df = secondary_df.rename(columns={'flaw_depth': secondary['depth_col']})
            
            # Either merge with primary results or use secondary as primary if no primary exists
            if not results_df.empty and 'Indication' in results_df.columns:
                # Get all the columns from secondary_df that we want to keep
                keep_cols = ['Indication', secondary['depth_col']]
                
                # Add any probe specific columns if they exist
                for col in secondary_df.columns:
                    if col.endswith('_depth') and col != secondary['depth_col'] and col != 'flaw_depth':
                        keep_cols.append(col)
                
                if secondary['method'] == 'pc_cc':
                    if 'apc_depth' in secondary_df.columns:
                        keep_cols.append('apc_depth')
                    if 'cpc_depth' in secondary_df.columns:
                        keep_cols.append('cpc_depth')
                
                # Only use columns that actually exist in secondary_df
                existing_cols = [col for col in keep_cols if col in secondary_df.columns]
                
                # Merge with primary results
                results_df = pd.merge(
                    results_df, 
                    secondary_df[existing_cols], 
                    on='Indication', 
                    how='left'
                )
            else:
                results_df = secondary_df
        except Exception as e:
            if not results_df.empty:
                results_df[secondary['depth_col']] = np.nan
            print(f"Error in secondary sizing for {flaw_type}: {str(e)}")
            file_logger.error(f'Error in sizing depth for {flaw_type} using secondary method: {e}', exc_info=True)
    
    return results_df

def predict_flaw_depth(scan: Scan, cscans_output: List[Dict], probes_data: Dict, config: object) -> pd.DataFrame:
    """
    Predict flaw depths and update flaws in the scan object.
    Always runs both primary and secondary analysis for all flaw types when available.

    Args:
        scan: Scan object containing flaws
        cscans_output: List of dictionaries with C-scan outputs
        probes_data: Dictionary containing probe data
        config: Configuration object

    Returns:
        pd.DataFrame: Results dataframe with depth predictions
    """
    
    # Create flaw lookup dictionary for faster access
    flaw_dict = {flaw.ind_num: flaw for flaw in scan.flaws}
    
    # Extract unique flaw types present in the scan
    pred_flaw_types = {flaw.flaw_type for flaw in scan.flaws}
    
    # Initialize base DataFrame with comprehension for better performance
    base_data = [
        {
            'frame_start': flaw.frame_start-1,
            'frame_end': flaw.frame_end-1,
            'flaw_rotary_start': flaw.rotary_start,
            'flaw_width': flaw.width,
            'flaw_id': flaw.ind_num,
            'scan_name': scan.scan_name,
            'scan_channel': scan.scan_channel,
            'scan_station': scan.scan_station,
            'scan_unit': scan.scan_unit,
            'flaw_type': flaw.flaw_type,
            'possible_category': ' '.join(flaw.possible_category) if flaw.possible_category else ''
        } 
        for flaw in scan.flaws
    ]
    
    # Skip processing if no flaws to analyze
    if not base_data:
        print("No flaws found in scan, returning empty DataFrame")
        return pd.DataFrame()
    
    # Create and process base DataFrame
    base_df = pd.DataFrame(base_data)
    processed_df = process_auto_sizing_summary(base_df, scan.bscan)
    
    # We don't need the multi-probe logic anymore since we're always running both methods
    # but we'll keep this field for backward compatibility, setting all to True
    processed_df['multi_probe'] = processed_df['possible_category'].str.split(' ')
    processed_df['calc_depth_multiple_probe'] = True  # Always set to True so both methods run
    
    # Track predicted depths for each flaw category
    possible_cats = {
        flaw.ind_num: {cat: None for cat in (flaw.possible_category or [])} 
        for flaw in scan.flaws
    }
    
    # Define flaw types to skip (can be extended as needed)
    skip_types = set()
    
    # Use collector pattern for results
    all_results = []
    
    # Constants for function arguments
    save_path = 'runs'
    run_name = 'auto_analysis'
    output_root = 'output'
    
    # Get total flaws to process for progress tracking
    total_flaws = sum(
        len(processed_df[processed_df['flaw_type'] == flaw_type])
        for flaw_type in pred_flaw_types
        if flaw_type not in skip_types
    )
    
    # Process each flaw type
    with tqdm(total=total_flaws, desc="Calculating flaw depth", ncols=100) as pbar:
        # Process each relevant flaw type
        for flaw_type in pred_flaw_types:
            if flaw_type in skip_types:
                print(f"Skipping flaw type {flaw_type} (in skip_types)")
                continue
            
            # Filter dataframe by flaw type once for efficiency
            type_df = processed_df[processed_df['flaw_type'] == flaw_type]
            
            if type_df.empty:
                print(f"No flaws of type {flaw_type} found in processed data")
                continue
            
            
            # Process flaws of this type
            for idx, row in type_df.iterrows():
                try:
                    # Create single flaw dataframe
                    single_flaw_df = pd.DataFrame([row])
                    
                    # Predict depth for this flaw
                    results = pred_flaw_depth_for_type(
                        df=single_flaw_df,
                        flaw_type=flaw_type,
                        cscans_output=cscans_output,
                        probes_data=probes_data,
                        config=config,
                        save_path=save_path,
                        run_name=run_name,
                        output_root=output_root
                    )
                    
                    # Process results if available
                    if results is not None and not results.empty:
                        all_results.append(results)
                        result_row = results.iloc[0]
                        flaw_id = result_row['Indication']
                        # Update flaw attributes using dictionary lookup
                        flaw = flaw_dict.get(flaw_id)
                        if flaw:
                            # Update depth attributes if not debris
                            if result_row.get('flaw_type') != 'Debris':
                                if result_row.get('flaw_depth') is not None:
                                    flaw.depth = result_row.get('flaw_depth')
                                if result_row.get('flaw_type') != 'CC':   
                                    # Update normal beam probe measurements
                                    if result_row.get('flaw_depth_nb') is not None:
                                        flaw.depth_nb = result_row.get('flaw_depth_nb')
                                        
                                    # Update NB1 and NB2 specific depths if available
                                    if result_row.get('nb1_depth') is not None:
                                        flaw.depth_nb1 = result_row.get('nb1_depth')
                                    if result_row.get('nb2_depth') is not None:
                                        flaw.depth_nb2 = result_row.get('nb2_depth')
                                
                            # Update pitch-catch measurements
                            if result_row.get('flaw_depth_pc') is not None:
                                flaw.depth_pc = result_row.get('flaw_depth_pc')
                                
                            # Check if apc_depth exists and is valid
                            apc_depth = result_row.get('apc_depth')
                            if apc_depth is not None:
                                # Handle case where apc_depth is a Series
                                if hasattr(apc_depth, 'empty'):
                                    # It's a pandas Series or DataFrame
                                    if not apc_depth.empty:
                                        # Get the first value if it's a Series
                                        apc_value = apc_depth.iloc[0] if len(apc_depth) > 0 else None
                                        if apc_value is not None and apc_value != 0 and not (isinstance(apc_value, float) and np.isnan(apc_value)):
                                            flaw.depth_apc = apc_value
                                else:
                                    # It's a single value
                                    if apc_depth != 0 and not (isinstance(apc_depth, float) and np.isnan(apc_depth)):
                                        flaw.depth_apc = apc_depth

                            # Check if cpc_depth exists and is valid
                            cpc_depth = result_row.get('cpc_depth')
                            if cpc_depth is not None:
                                # Handle case where cpc_depth is a Series
                                if hasattr(cpc_depth, 'empty'):
                                    # It's a pandas Series or DataFrame
                                    if not cpc_depth.empty:
                                        # Get the first value if it's a Series
                                        cpc_value = cpc_depth.iloc[0] if len(cpc_depth) > 0 else None
                                        if cpc_value is not None and cpc_value != 0 and not (isinstance(cpc_value, float) and np.isnan(cpc_value)):
                                            flaw.depth_cpc = cpc_value
                                else:
                                    # It's a single value
                                    if cpc_depth != 0 and not (isinstance(cpc_depth, float) and np.isnan(cpc_depth)):
                                        flaw.depth_cpc = cpc_depth
                            
                            # Update amplitude measurements
                            flaw.feature_amp = result_row.get('flaw_feature_amp')
                            flaw.max_amp = result_row.get('flaw_max_amp')
                            flaw.chatter_amplitude = result_row.get('chatter_amplitude')
                            flaw.probe_depth = result_row.get('probe_depth')
                            # Track depth for this flaw category
                            if flaw_type in possible_cats.get(flaw_id, {}):
                                possible_cats[flaw_id][flaw_type] = result_row.get('flaw_depth')
                    else:
                        print(f"No results obtained")
                
                except Exception as e:
                    print(f"Error processing flaw ID {row.get('flaw_id', 'unknown')} of type {flaw_type}: {str(e)}")
                    file_logger.error(f"Error processing flaw ID {row.get('flaw_id', 'unknown')} of type {flaw_type}: {str(e)}")
                    import traceback
                    traceback.print_exc()
                
                # Update progress
                pbar.update(1)
    
    # Post-process special cases for Note 1 flaws
    process_note1_flaws(flaw_dict, possible_cats)
    
    # Combine all results or return empty DataFrame
    processed_results = []
    for i, df in enumerate(all_results):
        if df is None or not isinstance(df, pd.DataFrame) or df.empty:
            continue
            
        # Check for duplicate column names
        if len(df.columns) != len(set(df.columns)):
            # print(f"DataFrame at index {i} has duplicate column names: {df.columns.tolist()}")
            
            # Rename duplicate columns to make them unique
            cols = pd.Series(df.columns)
            for dup in cols[cols.duplicated()].unique(): 
                cols[cols.duplicated()] = [f"{dup}_{i}" for i in range(sum(cols == dup))]
            df.columns = cols
            
        # Reset index to ensure clean concatenation
        df = df.reset_index(drop=True)
        processed_results.append(df)

    # Now concatenate the processed results
    if processed_results:
        final_df = pd.concat(processed_results, ignore_index=True)
        return final_df
    
    print("No results generated for any flaws")
    return pd.DataFrame()


def process_note1_flaws(flaw_dict, possible_cats):
    """
    Process special cases for Note 1 flaws.
    
    Args:
        flaw_dict: Dictionary mapping flaw IDs to flaw objects
        possible_cats: Dictionary tracking depths for possible categories
    """
    for flaw_id, flaw in flaw_dict.items():
        if flaw.flaw_type in ['Note 1', 'Note 1 - overlapped']:
            flaw_cats = possible_cats.get(flaw_id, {})
            if flaw_cats:
                # Get the first non-None depth value
                depths = [d for d in flaw_cats.values() if d is not None]
                if depths:
                    flaw.depth = depths[0]