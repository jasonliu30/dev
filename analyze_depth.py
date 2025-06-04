import matplotlib.pyplot as plt
import numpy as np
import os
import shutil
import pandas as pd
from matplotlib.lines import Line2D
import seaborn as sns
    
def count_depth_categories(data):
    # Initialize a dictionary to store counts
    category_counts = {}
    
    # Iterate through each item and count categories
    for item in data:
        category = item.get('depth_category')
        if category:
            if category in category_counts:
                category_counts[category] += 1
            else:
                category_counts[category] = 1
    
    return category_counts
def plot_depth_delta_by_category(depth_info_list, depth_metric='depth1_delta', title_prefix='Depth1'):
    """
    Create a plot showing depth delta values for each depth category.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
        depth_metric (str): The depth metric to plot ('depth1_delta' or 'depth2_delta')
        title_prefix (str): Prefix for the title ('Depth1' or 'Depth2')
    """
    # Group depth delta values by category
    delta_by_category = {}
    for info in depth_info_list:
        category = info['depth_category']
        delta = info.get(depth_metric)
        
        # Skip None values
        if delta is None:
            continue
            
        if category not in delta_by_category:
            delta_by_category[category] = []
        
        delta_by_category[category].append(delta)
    
    # Prepare for plotting
    categories = sorted(delta_by_category.keys(), key=lambda cat: len(delta_by_category[cat]), reverse=True)
    
    # Create a boxplot to show distribution
    plt.figure(figsize=(32, 10))
    box_data = [delta_by_category[cat] for cat in categories]
    box_plot = plt.boxplot(box_data, labels=categories, patch_artist=True)
    
    # Color the boxplots
    for box in box_plot['boxes']:
        box.set(facecolor='lightblue', alpha=0.8)
    
    # Add scatter points to show individual measurements
    for i, cat in enumerate(categories):
        # Get the data points and their flagged status for this category
        deltas = []
        flags = []
        for info in depth_info_list:
            if info['depth_category'] == cat and info.get(depth_metric) is not None:
                deltas.append(info[depth_metric])
                flags.append(info['flagged'])
        
        # Add jitter to x-position for better visualization
        x = np.random.normal(i+1, 0.04, size=len(deltas))
        
        # Plot unflagged points in navy blue
        unflagged_indices = [idx for idx, flag in enumerate(flags) if flag is False]
        if unflagged_indices:
            plt.scatter(x[unflagged_indices], [deltas[j] for j in unflagged_indices], 
                       alpha=0.6, color='navy', s=30, label=None if i > 0 else 'Not Flagged')
        
        # Plot flagged points in red
        flagged_indices = [idx for idx, flag in enumerate(flags) if flag is True]
        if flagged_indices:
            plt.scatter(x[flagged_indices], [deltas[j] for j in flagged_indices], 
                       alpha=0.6, color='red', s=30, marker="o", label=None if i > 0 else 'Flagged')
    
    # Add a reference line at y=0
    plt.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Calculate and display statistics for each category and add 2-sigma lines
    for i, cat in enumerate(categories):
        values = delta_by_category[cat]
        if not values:  # Skip empty categories
            continue
            
        mean_val = np.mean(values)
        std_val = np.std(values)
        count = len(values)
        
        # Add 2-sigma range lines for each category
        upper_2sigma = mean_val + 2*std_val
        lower_2sigma = mean_val - 2*std_val
        
        # Plot the 2-sigma range lines
        plt.hlines(y=upper_2sigma, xmin=i+0.75, xmax=i+1.25, colors='green', linestyles='dashed', label='_nolegend_')
        plt.hlines(y=lower_2sigma, xmin=i+0.75, xmax=i+1.25, colors='green', linestyles='dashed', label='_nolegend_')
        
        # Add small vertical connectors at the ends of the 2-sigma lines
        plt.vlines(x=i+0.75, ymin=lower_2sigma, ymax=upper_2sigma, colors='green', linestyles='dashed', alpha=0.7)
        plt.vlines(x=i+1.25, ymin=lower_2sigma, ymax=upper_2sigma, colors='green', linestyles='dashed', alpha=0.7)
        
        # Display statistics near each box with more space between categories
        plt.text(i+1, plt.ylim()[0] + (plt.ylim()[1] - plt.ylim()[0])*0.05, 
                f"n={count}\nμ={mean_val:.3f}\nσ={std_val:.3f}\n2σ=[{lower_2sigma:.3f}, {upper_2sigma:.3f}]", 
                ha='center', va='bottom', fontsize=10, 
                bbox=dict(boxstyle="round,pad=0.5", facecolor='white', alpha=0.8))
    
    # Add labels and title
    plt.xlabel('Depth Categories', fontsize=12, fontweight='bold')
    plt.ylabel(f'{title_prefix} Delta (predicted - reported)', fontsize=12, fontweight='bold')
    plt.title(f'Distribution of {title_prefix} Delta by Category', fontsize=14, fontweight='bold')
    
    # Add legend for flagged status
    plt.legend(loc='upper right', title='Flagged Status')
    
    # Rotate x-axis labels for better readability and increase spacing
    plt.xticks(rotation=45, ha='right', fontsize=12)
    
    # Increase spacing between tick labels
    plt.gca().xaxis.set_tick_params(pad=10)
    
    # Add a grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Add more space between plot elements
    plt.subplots_adjust(bottom=0.15, left=0.1, right=0.95, top=0.95)
    
    # Adjust layout to make room for rotated labels and text
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    
    return plt

def plot_depth_delta_overall(depth_info_list, include_flagged=False, flaw_type=None):
    """
    Create a histogram plot showing the overall distribution of depth1_delta values.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
        include_flagged (bool): Whether to include flagged cases in the analysis
        flaw_type (str, optional): Filter for specific flaw type (e.g., 'CC', 'Debris', 'FBBPF')
        
    Returns:
        plt: Matplotlib figure with the histogram visualization
    """

    # Collect deltas based on filtering criteria
    filtered_deltas = []
    
    for info in depth_info_list:
 
        # Apply flagged case filtering based on parameter
        if not include_flagged and info.get('flagged', False):
            continue
            
        delta = info.get('depth1_delta')
        # Skip NaN values
        if delta is not None and not np.isnan(delta):
            filtered_deltas.append(delta)
    
    # Calculate overall statistics
    if filtered_deltas:
        overall_mean = np.mean(filtered_deltas)
        overall_std = np.std(filtered_deltas)
        overall_count = len(filtered_deltas)
        overall_upper_2sigma = overall_mean + 2*overall_std
        overall_lower_2sigma = overall_mean - 2*overall_std
    else:
        overall_mean, overall_std, overall_count = 0, 0, 0
        overall_upper_2sigma, overall_lower_2sigma = 0, 0
    
    # Create histogram plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10), sharex=True)
    
    # Define bin settings for histograms
    bins = np.linspace(-0.07, 0.08, 36)  # Adjust range based on your data
    
    # Calculate non-stochastic error (mean)
    non_stochastic_error = overall_mean
    
    # Determine title based on flaw type and inclusion of flagged cases
    flag_status = "Including Flagged Cases" if include_flagged else "Excluding Flagged Cases"
    display_type = flaw_type if flaw_type else "All Types"
    
    # Plot for Non-Stochastic Error (top)
    ax1.hist(filtered_deltas, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax1.axvline(x=non_stochastic_error, color='green', linestyle='dashed', linewidth=2)
    ax1.set_ylabel('Count', fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    ax1.set_title(f'Depth Difference Distribution - {display_type}: {overall_count} cases\nNon-Stochastic Error ({flag_status})', fontsize=16)
    ax1.text(0.05, 0.9, f'non-stochastic Error: {non_stochastic_error:.4f}', 
            transform=ax1.transAxes, color='green', fontsize=12)
    
    # Plot for Stochastic Error (bottom)
    ax2.hist(filtered_deltas, bins=bins, color='steelblue', edgecolor='black', alpha=0.7)
    ax2.axvline(x=overall_lower_2sigma, color='green', linestyle='dashed', linewidth=2)
    ax2.axvline(x=overall_upper_2sigma, color='green', linestyle='dashed', linewidth=2)
    ax2.set_xlabel('depth difference (mm)', fontsize=12)
    ax2.set_ylabel('Count', fontsize=12)
    ax2.grid(True, linestyle='--', alpha=0.7)
    ax2.set_title('Stochastic Error', fontsize=16)
    ax2.text(0.05, 0.9, f'2 sigma: {overall_lower_2sigma:.4f}, +{overall_upper_2sigma:.4f}', 
            transform=ax2.transAxes, color='green', fontsize=12)
    
    plt.tight_layout()
    
    return plt

def categorize_depth_category(depth_category):
    """
    Categorize the depth_category dictionary according to specific rules.
    
    Rules:
    - if the dictionary contains "Spaghetti" -> "s"
    - if {"V": 1, "default": 1} -> "V+d"
    - if {"V": 1, "MR": 1} -> "V+M"
    - if V with any other categories -> "V+[other categories joined with +]"
    - if only one category with one count -> name of that category
    - if multiple MR -> "M"
    - if V has a count of 1 -> "V"
    - if V has a count > 1 -> "MV"
    - For all other cases -> first letter of each category joined with "+"
    
    Args:
        depth_category (dict or str): A dictionary with depth categories as keys and counts as values,
                                      or a string value to categorize
        
    Returns:
        str: The categorized string based on the rules
    """
    # Handle string input case for "spaghetti"
    if isinstance(depth_category, str) and depth_category.lower() == "spaghetti":
        return "s"
    
    # Handle None or empty dictionaries
    if not depth_category:
        return 'dnd'
    
    # Handle any dictionary containing "Spaghetti" regardless of other keys
    if isinstance(depth_category, dict) and "Spaghetti" in depth_category:
        return "s"
    
    # Check for V+def case
    if "V" in depth_category and "default" in depth_category and len(depth_category) == 2:
        return "V+d"
    
    # Check for V+MR case (when only these two are present)
    if "V" in depth_category and "MR" in depth_category and len(depth_category) == 2:
        return "V+M"
    
    # If V is present with multiple other categories
    if "V" in depth_category and len(depth_category) > 1:
        # Get all other categories besides V
        other_categories = [cat for cat in depth_category.keys() if cat != "V"]
        if other_categories:
            # Join all categories with V first
            return "V+" + "+".join(cat[0] for cat in other_categories)
    
    # Count the number of MR categories
    mr_count = sum(1 for cat in depth_category if cat.startswith("MR"))
    
    # If multiple MR categories
    if mr_count >= 1:
        return "M"
    
    # Check the count of V category
    if "V" in depth_category:
        v_value = depth_category["V"]
        if v_value == 1:
            return "V"
        elif v_value > 1:
            return "MV"
    
    # If only one category with one count
    if len(depth_category) == 1:
        key = list(depth_category.keys())[0]
        return key[0]
    
    # Default case: join the first letter of each category with "+"
    first_letters = [cat[0] for cat in sorted(depth_category.keys())]
    return "+".join(first_letters)

def extract_and_clean_extra_flaws(query_result, index):
    """
    Extracts all flaws from the "extra" field of a scan and removes:
    1. Flaws with only 3 fields
    2. Flaws where "pred_depth" is less than 0.1
    3. Flaws where "pred_depth" is a string
    4. Flaws where "pred_depth" is null
    
    Args:
        query_result (dict): The scan data dictionary containing a "scan" field with an "extra" field
        index (int): The index number of the scan
        
    Returns:
        list: List of filtered flaws with scan information added
    """
    # Check if scan_data has an "extra" field
    if "extra" not in query_result['scan']:
        print(f"No 'extra' field found in scan {query_result['scan'].get('scan_id', 'unknown')}")
        return []
    
    # Get the scan name/ID
    scan_id = query_result['scan'].get('scan_id', 'unknown')
    
    # Get the list of extra flaws
    extra_flaws_copy = query_result['scan']["extra"]
    extra_flaws = extra_flaws_copy.copy()
    
    # Initialize categorized flaws lists
    valid_flaws = []
    removed_insufficient_fields = []
    removed_low_depth = []
    removed_string_depth = []
    removed_null_depth = []
    
    for flaw in extra_flaws:
        # Check if flaw has fewer than 4 fields
        if len(flaw) <= 3:
            removed_insufficient_fields.append(flaw)
            continue
        
        # Check pred_depth value
        pred_depth = flaw.get('pred_depth')
        flaw_type = flaw.get('pred_flaw_type')
        ind_num = flaw.get('ind_num')
        # Skip if pred_depth is None
        if pred_depth is None:
            removed_null_depth.append(flaw)
            continue
        
        # Check if pred_depth is a string
        if isinstance(pred_depth, str):
            # If it contains "<" or starts with a number less than 0.1
            if "<" in pred_depth or (pred_depth.split()[0].replace('.', '', 1).isdigit() and float(pred_depth.split()[0]) < 0.1):
                removed_low_depth.append(flaw)
            else:
                removed_string_depth.append(flaw)
            continue
        
        # Check if pred_depth is a number less than 0.1
        if isinstance(pred_depth, (int, float)) and pred_depth < 0.1:
            removed_low_depth.append(flaw)
            continue
        
        # If we get here, the flaw passes all filters
        # Add scan information to the flaw
        flaw_with_info = {
            'scan_id': scan_id,
            'ind_num': ind_num,
            'depth': pred_depth,
            'flaw_type': flaw_type,
            'original_flaw': flaw
        }
        valid_flaws.append(flaw_with_info)
    
    # Print summary
    print(f"Processed {len(extra_flaws)} flaws from 'extra' field in scan {scan_id} (index: {index})")
    print(f"  - Kept {len(valid_flaws)} valid flaws")
    print(f"  - Removed {len(removed_insufficient_fields)} flaws with 3 or fewer fields")
    print(f"  - Removed {len(removed_low_depth)} flaws with depth < 0.1")
    print(f"  - Removed {len(removed_string_depth)} flaws with string depth values")
    print(f"  - Removed {len(removed_null_depth)} flaws with null depth values")
    
    # Print removed flaws details (optional - can be commented out for brevity)
    if removed_insufficient_fields:
        print("\nRemoved flaws with insufficient fields:")
        for flaw in removed_insufficient_fields[:3]:  # Limit to first 3 for brevity
            print(f"  - {flaw}")
        if len(removed_insufficient_fields) > 3:
            print(f"  - ... and {len(removed_insufficient_fields) - 3} more")
    
    return valid_flaws

def extract_depth_info(query_result, depth_info_list, depth2_buffer=0):
    """
    Extract depth information from each flaw in the scan data.
    Excludes scans with outage number "P2251".
    
    Args:
        query_result (dict): A dictionary containing scan information with a 'flaws' list
        depth_info_list (list): List to accumulate depth information 
        depth2_buffer (float): Buffer value to add to depth2 when calculating depth2_delta
        
    Returns:
        list: A list of dictionaries, each containing the depth information for a flaw
    """
    
    # Skip processing if the scan has outage number "P2251"
    # if query_result.get('scan', {}).get('outage_number') == "P2251":
    #     return depth_info_list
   
    for flaw in query_result['scan']["reported_flaws"]:
        # Extract the required depth information
        depth_reported = flaw["depth"]
        depth_predicted = flaw["pred_depth"]
        flagged = None
        if flaw.get("pred_depth") != 'see possible category flaw depth':
            if "<" not in str(flaw.get('depth_reported', '')):
                if flaw.get("depth") is not None:
                  
                    if any(text in str(flaw.get("pred_depth", '')) for text in ["(2)", "See note_2"]):
                        flagged = True
                        # if "<" in str(flaw.get("pred_depth")):
                        #     depth = float(flaw.get("pred_depth").split()[1]) - 0.01
                        # else:
                        #     depth = float(flaw.get("pred_depth").split()[0])
                    else:
                        flagged = False
                        # if "<" in str(flaw.get("pred_depth")):
                        #     depth = float(flaw.get("pred_depth").split()[1]) - 0.01
                        # else:
                        #     depth = float(flaw.get("pred_depth"))
                            
                    # # Calculate depth1_delta
                    # depth1_delta = depth - float(flaw.get('depth_reported'))
                    
                    # # Prepare depth2 value with the minimum constraint
                    # depth2_value = flaw.get('depth2')
                    # depth2_delta = None
                    # if depth2_value is not None:
                    #     # Check if depth2_value is a string and convert to float if needed
                    #     if isinstance(depth2_value, str):
                    #         # Remove any non-numeric parts (like the "(2)" suffix)
                    #         clean_value = depth2_value.split()[0] if " " in depth2_value else depth2_value.split("(")[0]
                    #         try:
                    #             depth2_value = float(clean_value)
                    #         except ValueError:
                    #             # Handle case where conversion fails
                    #             depth2_value = None
                        
                    #     # Now perform the comparison with the numeric value
                    #     if depth2_value is not None and depth2_value < 0.1:
                    #         depth2_value = 0.09
                                            
                    #     # Calculate depth2_delta with the adjusted value plus buffer
                    #     depth2_delta = (depth2_value + depth2_buffer) - float(flaw.get('depth_reported'))
                        
                    depth_info = {
                        'scan_name': query_result['scan']['scan_id'],
                        'scan_num': query_result['scan']['numeric_id'],
                        'year': get_year_from_outage_number(query_result['scan']['outage_number']),
                        'channel': query_result['scan']['scan_channel'],
                        'ind_num': flaw.get('ind_num'),
                        'pred_ind_num': flaw.get('pred_ind_num'),
                        'depth': flaw.get("pred_depth"),
                        #'depth2': flaw.get('depth2'),
                        'depth_category': flaw.get('depth_category'),
                        'depth_reported': flaw.get('depth'),
                        'depth_pc': flaw.get('depth_pc'),
                        'depth2_delta': -flaw["metrics"]["depth_diff"],
                        'width_delta': flaw["metrics"]["width_diff"],
                        'length_delta': flaw["metrics"]["length_diff"],
                        'position_x_delta': flaw["metrics"]["position_x_diff"],
                        'position_y_delta': flaw["metrics"]["position_y_diff"],
                        # 'depth2_delta': depth2_delta,
                        'flagged': flagged,
                        'predicted': flaw.get('is_predicted'),
                        'classified': flaw.get('is_classified_correct'),
                    }
                
                    depth_info_list.append(depth_info)
    
    return depth_info_list

def get_year_from_outage_number(outage_number):
    """
    Extract the year from an outage number.
    Format is D followed by year (first 2 digits) and additional numbers.
    Handles special cases and invalid formats.
    
    Examples:
        D1321 -> 2013
        D2021 -> 2020
        None/invalid formats -> returns None
    
    Args:
        outage_number (str): The outage number in the format Dxxxx
        
    Returns:
        int or None: The extracted year, or None if format is invalid/unrecognized
    """
    # Check for None or empty string
    if outage_number is None or outage_number == '':
        return None
    
    # Strip any whitespace and ensure we're working with a string
    outage_number = str(outage_number).strip()
    
    # Check if the format is valid (starts with D followed by at least 2 digits)
    if not outage_number.startswith('D'):
        # Special case handling for non-standard formats
        if outage_number.lower().startswith('on'):
            # Handle "Ontario" or other ON prefixes - maybe use a default year or extract from elsewhere
            # For now, return None to indicate unknown year
            return None
        else:
            # Other unrecognized formats
            return None
    
    # Try to extract the numerical part
    try:
        # Extract the first two digits after 'D'
        numerical_part = outage_number[1:]
        
        # Find the first two consecutive digits
        year_digits = ""
        for i in range(len(numerical_part) - 1):
            if numerical_part[i].isdigit() and numerical_part[i+1].isdigit():
                year_digits = numerical_part[i:i+2]
                break
        
        # If we found two consecutive digits, convert to year
        if year_digits:
            # All years are in the 2000s
            return 2000 + int(year_digits)
        else:
            # Couldn't find two consecutive digits
            return None
            
    except (ValueError, IndexError):
        # Handle any parsing errors
        return None

def extract_detection_and_classification_info(query_result, detection_info_list):
    """
    Extract detection, characterization, and classification information.
    - Calculates delta values for predicted flaws compared to reported flaws
    - Tracks which reported flaws are successfully predicted and correctly classified
    
    Args:
        query_result (dict): A dictionary containing scan information with 'flaws' list
        detection_info_list (list): List to accumulate detection information
        
    Returns:
        list: A list of dictionaries with detection and classification information
    """
    # Process predicted flaws matched to reported flaws (similar to your original function)
    for flaw in query_result['scan']['flaws']:
        # Find the matching reported flaw using reported_id
        reported_flaw = next(
            (r_flaw for r_flaw in query_result['scan']['reported_flaws'] 
             if r_flaw['ind_num'] == flaw['reported_id']),
            None
        )
        
        # Skip if no matching reported flaw is found
        if reported_flaw is None:
            continue
        
        try:
            # Calculate delta values (predicted - reported)
            axial_start_delta = float(flaw['axial_start']) - float(reported_flaw['axial_start'])
            rotary_start_delta = float(flaw['rotary_start']) - float(reported_flaw['rotary_start'])
            length_delta = float(flaw['length']) - float(reported_flaw['length'])
            width_delta = float(flaw['width']) - float(reported_flaw['width'])
            
            # Determine if the flaw is flagged (based on depth notation or other criteria)
            flagged = False
            if isinstance(flaw.get('depth'), str):
                flagged = any(text in flaw.get('depth', '') for text in ["(2)", "See note_2"])
            
            # Check if classification is correct
            classification_correct = (flaw.get('flaw_type') == reported_flaw.get('flaw_type'))
                
            # Create detection info dictionary
            detection_info = {
                'scan_name': query_result['scan']['scan_id'],
                'scan_num': query_result['scan'].get('numeric_id', ''),
                'ind_num': flaw['ind_num'],
                'reported_ind_num': reported_flaw['ind_num'],
                'axial_start': flaw['axial_start'],
                'rotary_start': flaw['rotary_start'],
                'length': flaw['length'],
                'width': flaw['width'],
                'axial_start_reported': reported_flaw['axial_start'],
                'rotary_start_reported': reported_flaw['rotary_start'],
                'length_reported': reported_flaw['length'],
                'width_reported': reported_flaw['width'],
                'axial_start_delta': axial_start_delta,
                'rotary_start_delta': rotary_start_delta,
                'length_delta': length_delta,
                'width_delta': width_delta,
                'flaw_type': flaw.get('flaw_type'),
                'flaw_type_reported': reported_flaw.get('flaw_type'),
                'classification_correct': classification_correct,
                'depth': flaw.get('depth'),
                'depth_reported': reported_flaw.get('depth'),
                'flagged': flagged,
                'is_predicted': True,
                'detection_match_type': 'predicted_to_reported'
            }
            
            # If depth_category exists, include it
            if 'depth_category' in flaw:
                detection_info['depth_category'] = categorize_depth_category(flaw['depth_category'])
            
            detection_info_list.append(detection_info)
            
        except (ValueError, TypeError) as e:
            # Skip this flaw if there are conversion errors but log it
            print(f"Error processing flaw {flaw['ind_num']} in scan {query_result['scan']['scan_id']}: {e}")
            continue
    
    # Now check which reported flaws are successfully predicted and which are missed
    for reported_flaw in query_result['scan']['reported_flaws']:
        # Find matching predicted flaws for this reported flaw
        matching_predicted_flaws = [
            pred_flaw for pred_flaw in query_result['scan']['flaws']
            if pred_flaw['reported_id'] == reported_flaw['ind_num']
        ]
        
        # Record if this reported flaw has a corresponding prediction
        is_detected = len(matching_predicted_flaws) > 0
        
        # Check if classification is correct (if detected)
        classification_correct = False
        if is_detected:
            classification_correct = any(
                pred_flaw.get('flaw_type') == reported_flaw.get('flaw_type')
                for pred_flaw in matching_predicted_flaws
            )
        
        # Create a summary entry for this reported flaw
        reported_summary = {
            'scan_name': query_result['scan']['scan_id'],
            'scan_num': query_result['scan'].get('numeric_id', ''),
            'reported_ind_num': reported_flaw['ind_num'],
            'axial_start_reported': reported_flaw['axial_start'],
            'rotary_start_reported': reported_flaw['rotary_start'],
            'length_reported': reported_flaw['length'],
            'width_reported': reported_flaw['width'],
            'flaw_type_reported': reported_flaw.get('flaw_type'),
            'depth_reported': reported_flaw.get('depth'),
            'is_detected': is_detected,
            'classification_correct': classification_correct,
            'num_matching_predictions': len(matching_predicted_flaws),
            'detection_match_type': 'reported_summary'
        }
        
        detection_info_list.append(reported_summary)
    
    return detection_info_list

def plot_category_distribution(category_counts):
    """
    Create a bar chart visualization of category distributions with counts and percentages,
    consolidating categories into d, v (including mv), s, m, 7, and combined.
    
    Args:
        category_counts (dict): Dictionary with categories as keys and counts as values
    """
    # Create the consolidated groups
    consolidated_counts = {
        'd': 0,
        'v': 0,  # Will include both 'v' and 'mv'
        's': 0,
        'm': 0,
        '7': 0,
        'combined': 0  # For all other categories
    }
    
    # Consolidate the counts
    for category, count in category_counts.items():
        category_lower = category.lower()
        
        if category_lower == 'd':
            consolidated_counts['d'] += count
        elif category_lower == 'v' or category_lower == 'mv':
            consolidated_counts['v'] += count
        elif category_lower == 's':
            consolidated_counts['s'] += count
        elif category_lower == 'm':
            consolidated_counts['m'] += count
        elif category_lower == '7':
            consolidated_counts['7'] += count
        else:
            consolidated_counts['combined'] += count
    
    # Remove any empty categories (count = 0)
    consolidated_counts = {k: v for k, v in consolidated_counts.items() if v > 0}
    
    # Calculate the total for percentages
    total = sum(consolidated_counts.values())

    # Sort the categories by count in descending order
    sorted_categories = sorted(consolidated_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]

    # Calculate percentages
    percentages = [(count / total) * 100 for count in counts]

    # Create the bar chart
    plt.figure(figsize=(14, 6))
    bars = plt.bar(categories, counts, color='steelblue')

    # Add count and percentage labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{counts[i]} ({percentages[i]:.1f}%)', ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('Depth Categories (Consolidated)', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Distribution of Consolidated Depth Categories', fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add a grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to make room for rotated labels
    plt.tight_layout()
    
    return plt

def plot_category_distribution(category_counts):
    """
    Create a bar chart visualization of category distributions with counts and percentages,
    plotting each category separately.
    
    Args:
        category_counts (dict): Dictionary with categories as keys and counts as values
    """
    # Remove any empty categories (count = 0)
    category_counts = {k: v for k, v in category_counts.items() if v > 0}
    
    # Calculate the total for percentages
    total = sum(category_counts.values())

    # Sort the categories by count in descending order
    sorted_categories = sorted(category_counts.items(), key=lambda x: x[1], reverse=True)
    categories = [item[0] for item in sorted_categories]
    counts = [item[1] for item in sorted_categories]

    # Calculate percentages
    percentages = [(count / total) * 100 for count in counts]

    # Create the bar chart
    plt.figure(figsize=(14, 6))
    bars = plt.bar(categories, counts, color='steelblue')

    # Add count and percentage labels on top of each bar
    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                f'{counts[i]} ({percentages[i]:.1f}%)', ha='center', va='bottom')

    # Add labels and title
    plt.xlabel('Depth Categories', fontweight='bold')
    plt.ylabel('Count', fontweight='bold')
    plt.title('Distribution of Depth Categories', fontweight='bold')

    # Rotate x-axis labels for better readability
    plt.xticks(rotation=45, ha='right')

    # Add a grid for easier reading
    plt.grid(axis='y', linestyle='--', alpha=0.7)

    # Adjust layout to make room
    # for rotated labels
    plt.tight_layout()
    
    return plt
def get_category_outlier_scans(depth_info_list, categories=None, delta_threshold=0.04, delta_field='depth2_delta'):
    """
    Get all scan numbers where the depth category matches specified categories and the delta value is outside of ±threshold.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
        categories (list or str): Categories to filter for (default: None, which includes all categories)
                                  Can be a single string or a list of strings
        delta_threshold (float): The threshold for delta values (default: 0.04)
        delta_field (str): The field to check against the threshold (default: 'depth2_delta')
        
    Returns:
        dict: Dictionary with scan numbers as keys and lists of outlier information as values
    """
    # Handle categories parameter
    if categories is None:
        # Include all categories if None is provided
        category_filter = lambda x: True
    elif isinstance(categories, str):
        # If a single string is provided, convert to a set for faster lookups
        categories_set = {categories}
        category_filter = lambda x: x in categories_set
    else:
        # If a list or other iterable is provided, convert to a set for faster lookups
        categories_set = set(categories)
        category_filter = lambda x: x in categories_set
    
    # Initialize dictionary to store results
    outlier_scans = {}
    
    # Filter the depth_info_list for matching categories and delta outside ±threshold
    for info in depth_info_list:
        # Check if the category matches the filter
        category_match = category_filter(info['depth_category'])
        
        # Check if delta is outside the threshold
        delta_value = info.get(delta_field)
        if delta_value is not None:
            delta_outside_threshold = abs(delta_value) > delta_threshold
        else:
            delta_outside_threshold = True
        
        # If both conditions are met, add to results
        if category_match and delta_outside_threshold:
            scan_num = info['scan_num']
      
            # Initialize the list for this scan if it doesn't exist
            if scan_num not in outlier_scans:
                outlier_scans[scan_num] = []
            
            # Add the outlier information (including all relevant fields from the original info)
            outlier_info = {
                'ind_num': info['pred_ind_num'],
                'scan_name': info['scan_name'],
                'depth_category': info['depth_category'],
                'depth_reported': info['depth_reported'],
                #'depth1_delta': info['depth1_delta'],
                'depth2_delta': info.get('depth2_delta'),
                'flagged': info.get('flagged', False)
            }
            
            outlier_scans[scan_num].append(outlier_info)
    
    return outlier_scans
    """
    Create a plot comparing depth1_delta and depth2_delta values for each depth category.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
    
    Returns:
        matplotlib.figure.Figure: The figure containing the comparison plot
    """
    import numpy as np
    import matplotlib.pyplot as plt
    
    # Group depth delta values by category for both metrics
    depth1_by_category = {}
    depth2_by_category = {}
    
    for info in depth_info_list:
        category = info['depth_category']
        
        # Get depth1_delta and handle None values
        depth1_delta = info.get('depth1_delta')
        if depth1_delta is not None:
            if category not in depth1_by_category:
                depth1_by_category[category] = []
            depth1_by_category[category].append(depth1_delta)
        
        # Get depth2_delta and handle None values
        depth2_delta = info.get('depth2_delta')
        if depth2_delta is not None:
            if category not in depth2_by_category:
                depth2_by_category[category] = []
            depth2_by_category[category].append(depth2_delta)
    
    # Get unique categories from both metrics
    all_categories = sorted(set(list(depth1_by_category.keys()) + list(depth2_by_category.keys())),
                           key=lambda cat: len(depth1_by_category.get(cat, [])) + len(depth2_by_category.get(cat, [])),
                           reverse=True)
    
    # Create figure and subplots
    fig, ax = plt.subplots(figsize=(32, 10))
    
    # Position for each category on the x-axis
    category_positions = range(1, len(all_categories) + 1)
    
    # Width of each box
    box_width = 0.35
    
    # Prepare boxplot data
    depth1_data = [depth1_by_category.get(cat, []) for cat in all_categories]
    depth2_data = [depth2_by_category.get(cat, []) for cat in all_categories]
    
    # Create boxplots
    pos1 = [p - box_width/2 for p in category_positions]
    pos2 = [p + box_width/2 for p in category_positions]
    
    # Create boxplots
    bp1 = ax.boxplot(depth1_data, positions=pos1, widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor='lightblue'))
    bp2 = ax.boxplot(depth2_data, positions=pos2, widths=box_width, patch_artist=True,
                   boxprops=dict(facecolor='lightgreen'))
    
    # Add a reference line at y=0
    ax.axhline(y=0, color='r', linestyle='-', alpha=0.3)
    
    # Add scatter points for individual measurements
    for i, cat in enumerate(all_categories):
        # Depth1 scatter
        if cat in depth1_by_category:
            deltas = depth1_by_category[cat]
            x = np.random.normal(i+1-box_width/2, 0.04, size=len(deltas))
            ax.scatter(x, deltas, alpha=0.6, color='navy', s=30)
        
        # Depth2 scatter
        if cat in depth2_by_category:
            deltas = depth2_by_category[cat]
            x = np.random.normal(i+1+box_width/2, 0.04, size=len(deltas))
            ax.scatter(x, deltas, alpha=0.6, color='darkgreen', s=30)
    
    # Calculate and display statistics for each category
    for i, cat in enumerate(all_categories):
        # Depth1 statistics
        if cat in depth1_by_category and depth1_by_category[cat]:
            values = depth1_by_category[cat]
            mean_val = np.mean(values)
            std_val = np.std(values)
            count = len(values)
            
            # Add text for depth1 metrics
            ax.text(i+1-box_width/2, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.05, 
                    f"n={count}\nμ={mean_val:.3f}\nσ={std_val:.3f}", 
                    ha='center', va='bottom', fontsize=9, color='navy',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
        
        # Depth2 statistics
        if cat in depth2_by_category and depth2_by_category[cat]:
            values = depth2_by_category[cat]
            mean_val = np.mean(values)
            std_val = np.std(values)
            count = len(values)
            
            # Add text for depth2 metrics
            ax.text(i+1+box_width/2, ax.get_ylim()[0] + (ax.get_ylim()[1] - ax.get_ylim()[0])*0.05, 
                    f"n={count}\nμ={mean_val:.3f}\nσ={std_val:.3f}", 
                    ha='center', va='bottom', fontsize=9, color='darkgreen',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='white', alpha=0.8))
    
    # Set the labels and title
    ax.set_xlabel('Depth Categories', fontsize=12, fontweight='bold')
    ax.set_ylabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
    ax.set_title('Comparison of Depth1 vs Depth2 Delta by Category', fontsize=14, fontweight='bold')
    
    # Set x-tick positions and labels
    ax.set_xticks(category_positions)
    ax.set_xticklabels(all_categories, rotation=45, ha='right', fontsize=12)
    
    # Add legend
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='lightblue', edgecolor='black', label='Depth1 Delta'),
        Patch(facecolor='lightgreen', edgecolor='black', label='Depth2 Delta')
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    # Add a grid for easier reading
    ax.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Adjust layout
    plt.tight_layout(rect=[0, 0.06, 1, 0.95])
    
    return fig

def copy_outlier_images(depth_info_list, categories='V', delta_threshold=0.03, 
                       root_folder='depth_vision_snap', outfolder='S:\\snaps\\3.outlier',
                       delta_field='depth2_delta'):
    """
    Identify outlier scans and copy their images to the output folder.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
        categories (str or list): Categories to filter for
        delta_threshold (float): Threshold for considering a scan an outlier
        root_folder (str): Source folder containing the original images
        outfolder (str): Destination folder for copied images
    
    Returns:
        list: List of filenames that were copied
    """
    # Get outlier scans
    outlier_scans = get_category_outlier_scans(depth_info_list, categories=categories, 
                                              delta_threshold=delta_threshold, delta_field=delta_field)
    
    # Create output directory if it doesn't exist
    output_dir = os.path.join(outfolder, f'{categories.lower()}_outlier')
    os.makedirs(output_dir, exist_ok=True)
    
    # Generate filenames and copy files
    fnames = []
    copied_files = []
    not_found_files = []
    olist = []
    for ind, items in outlier_scans.items():
        for item in items:
            # Create filename
            olist.append(item['scan_name'])
            filename = f"{item['scan_name'].replace('Type D', 'Type A')}_{item['ind_num']}_depth.png"
            f2 = f"{item['scan_name'].replace('Type D', 'Type A')}_{item['ind_num']}_{item['depth_reported']}.png"
            fnames.append(filename)
            
            # Source and destination paths
            source_path = os.path.join(root_folder, filename)
            dest_path = os.path.join(output_dir, f2)
            
            # Copy file if it exists
            if os.path.exists(source_path):
                shutil.copy2(source_path, dest_path)
                copied_files.append(filename)
                print(f"Copied: {filename}")
            else:
                not_found_files.append(filename)
                print(f"File not found: {source_path}")
    
    # Print summary
    print(f"\nSummary:")
    print(f"Total outliers identified: {len(fnames)}")
    print(f"Files successfully copied: {len(copied_files)}")
    print(f"Files not found: {len(not_found_files)}")
    
    return fnames, olist

def plot_depth_delta_by_year_category(depth_info_list, use_categories=True):
    """
    Create a scatter plot showing depth1_delta across different years, 
    optionally color-coded by depth_category if present.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
        use_categories (bool): Whether to color points by category. If False, all points will be the same color.
        
    Returns:
        plt: Matplotlib figure with the visualization
    """
    # Filter out flagged cases and convert data to a format suitable for plotting
    valid_data = []
    
    for info in depth_info_list:
        # Skip flagged cases
        if info.get('flagged', False):
            continue
        
        # Skip entries with NaN values in required fields
        if info.get('year') is None or info.get('depth1_delta') is None:
            continue
            
        # Check if depth1_delta is a string and convert if necessary
        depth1_delta = info.get('depth1_delta')
        if isinstance(depth1_delta, str):
            try:
                depth1_delta = float(depth1_delta)
            except (ValueError, TypeError):
                continue
        
        # Skip NaN values
        if np.isnan(depth1_delta):
            continue
            
        # Add to valid data
        entry = {
            'year': info.get('year'),
            'depth1_delta': depth1_delta
        }
        
        # Only include category if we're using categories and it exists
        if use_categories and 'depth_category' in info:
            entry['category'] = info.get('depth_category', 'Unknown')
        
        valid_data.append(entry)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(valid_data)
    
    # Create figure
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # Plot scatter points
    if use_categories and 'category' in df.columns:
        # Get unique categories for color mapping
        categories = df['category'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        category_color_map = dict(zip(categories, colors))
        
        # Plot by category with different colors
        for category, category_df in df.groupby('category'):
            ax.scatter(
                category_df['year'], 
                category_df['depth1_delta'],
                color=category_color_map[category],
                alpha=0.7,
                s=50,
                label=category
            )
        title_suffix = "by Category"
    else:
        # Plot all points with the same color
        ax.scatter(
            df['year'], 
            df['depth1_delta'],
            color='steelblue',
            alpha=0.7,
            s=50,
            label='All Data Points'
        )
        title_suffix = "(All Data Points)"
    
    # Group by year for statistics
    yearly_stats = df.groupby('year')['depth1_delta'].agg(['mean', 'std']).reset_index()
    
    # Plot mean line
    ax.plot(
        yearly_stats['year'],
        yearly_stats['mean'],
        'k-',
        linewidth=2,
        label='Yearly Mean'
    )
    
    # Plot 2-sigma bands
    ax.fill_between(
        yearly_stats['year'],
        yearly_stats['mean'] - 2*yearly_stats['std'],
        yearly_stats['mean'] + 2*yearly_stats['std'],
        color='gray',
        alpha=0.2,
        label='±2σ Range'
    )
    
    # Add reference line at y=0
    ax.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Difference')
    
    # Set labels and title
    ax.set_xlabel('Year', fontsize=14)
    ax.set_ylabel('Depth1 Delta (mm)', fontsize=14)
    ax.set_title(f'Depth1 Delta Across Years {title_suffix}', fontsize=16)
    
    # Format x-axis to show years as integers
    ax.xaxis.set_major_locator(plt.MaxNLocator(integer=True))
    
    # Add grid
    ax.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend
    ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout to make room for the legend
    plt.tight_layout()
    
    return plt

def plot_depth_delta_by_channel(depth_info_list, use_categories=False, show_stats=False):
    """
    Create visualization showing depth1_delta across different channels,
    optionally color-coded by depth_category if present.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
        use_categories (bool): Whether to color points by category. If False, all points will be the same color.
        
    Returns:
        plt: Matplotlib figure with the visualization
    """
    # Filter out flagged cases and convert data to a format suitable for plotting
    valid_data = []
    
    for info in depth_info_list:
        # Skip flagged cases
        if info.get('flagged', False):
            continue
        
        # Skip entries with NaN values in required fields
        if info.get('channel') is None or info.get('depth1_delta') is None:
            continue
            
        # Check if depth1_delta is a string and convert if necessary
        depth1_delta = info.get('depth1_delta')
        if isinstance(depth1_delta, str):
            try:
                depth1_delta = float(depth1_delta)
            except (ValueError, TypeError):
                continue
        
        # Skip NaN values
        if np.isnan(depth1_delta):
            continue
            
        # Add to valid data
        entry = {
            'channel': info.get('channel'),
            'depth1_delta': depth1_delta
        }
        
        # Only include category if we're using categories and it exists
        if use_categories and 'depth_category' in info:
            entry['category'] = info.get('depth_category', 'Unknown')
        
        valid_data.append(entry)
    
    # Convert to DataFrame for easier manipulation
    df = pd.DataFrame(valid_data)
    
    if len(df) == 0:
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.text(0.5, 0.5, "No valid data to plot", ha='center', va='center', fontsize=14)
        ax.set_title("Depth Delta by Channel - No Data", fontsize=16)
        return plt
    
    # Create figure with subplots: boxplot and scatter plot
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 14), sharex=True, 
                                  gridspec_kw={'height_ratios': [1, 2]})
    
    # Sort channels alphabetically for consistent ordering
    channels = sorted(df['channel'].unique())
    
    # Create a boxplot on the first subplot
    sns.boxplot(x='channel', y='depth1_delta', data=df, ax=ax1, 
                order=channels, color='lightblue')
    ax1.axhline(y=0, color='r', linestyle='--', alpha=0.5)
    ax1.set_title("Distribution of Depth1 Delta by Channel", fontsize=14)
    ax1.set_xlabel('')  # Remove x-label from top plot
    ax1.set_ylabel("Depth1 Delta (mm)", fontsize=12)
    ax1.grid(True, linestyle='--', alpha=0.7)
    
    # Plot scatter points on the second subplot
    if use_categories and 'category' in df.columns:
        # Get unique categories for color mapping
        categories = df['category'].unique()
        colors = plt.cm.tab10(np.linspace(0, 1, len(categories)))
        category_color_map = dict(zip(categories, colors))
        
        # Plot by category with different colors
        for category, category_df in df.groupby('category'):
            # Add jitter to x positions for better visibility
            x_jitter = np.random.normal(0, 0.1, size=len(category_df))
            
            # Get numeric channel positions
            x_positions = [channels.index(ch) for ch in category_df['channel']]
            
            ax2.scatter(
                np.array(x_positions) + x_jitter, 
                category_df['depth1_delta'],
                color=category_color_map[category],
                alpha=0.7,
                s=50,
                label=category
            )
        title_suffix = "by Category"
    else:
        # Add jitter to x positions for better visibility
        for channel, channel_df in df.groupby('channel'):
            x_jitter = np.random.normal(0, 0.1, size=len(channel_df))
            x_position = channels.index(channel)
            
            ax2.scatter(
                np.array([x_position] * len(channel_df)) + x_jitter,
                channel_df['depth1_delta'],
                color='steelblue',
                alpha=0.7,
                s=50
            )
        title_suffix = "(All Data Points)"
    
    # Add statistics by channel
    for i, channel in enumerate(channels):
        channel_data = df[df['channel'] == channel]['depth1_delta']
        if len(channel_data) > 0:
            mean = channel_data.mean()
            
            # Plot mean as a horizontal line
            ax2.plot([i-0.3, i+0.3], [mean, mean], 'k-', linewidth=2)
            
            # Add text with stats only if show_stats is True
            if show_stats:
                std = channel_data.std()
                count = len(channel_data)
                ax2.text(i, max(channel_data) + 0.01, 
                        f"n={count}\nμ={mean:.3f}\nσ={std:.3f}", 
                        ha='center', va='bottom', fontsize=9)
    
    # Add reference line at y=0
    ax2.axhline(y=0, color='r', linestyle='--', alpha=0.5, label='Zero Difference')
    
    # Set labels and title for scatter plot
    ax2.set_xlabel("Channel", fontsize=12)
    ax2.set_ylabel("Depth1 Delta (mm)", fontsize=12)
    ax2.set_title(f"Depth1 Delta by Channel {title_suffix}", fontsize=14)
    
    # Set x-tick labels to channel names with rotation to prevent crowding
    ax2.set_xticks(range(len(channels)))
    ax2.set_xticklabels(channels, rotation=45, ha='right')
    
    # Adjust figure bottom to make room for rotated labels
    plt.subplots_adjust(bottom=0.15)
    
    # Add grid
    ax2.grid(True, linestyle='--', alpha=0.7)
    
    # Add legend if using categories
    if use_categories and 'category' in df.columns:
        ax2.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Adjust layout
    plt.tight_layout()
    
    return plt

def plot_depth_delta_comparison(depth_info_list):
    """
    Create histograms comparing depth1_delta and depth2_delta values for each depth category.
    Each category gets a separate subplot with 2-sigma markings, and an overall histogram is also created.
    The overall histogram uses different colors to represent different categories.
    Also includes a plot showing what the histogram would look like if we enforce 2-sigma to be within ±0.04.
    Excludes DFP, FBBPF and FBBPF (M) categories.
    Saves plots to S:\snaps\1. graphs\Ite3_graphs with meaningful filenames.
    
    Args:
        depth_info_list (list): List of dictionaries containing depth information
    
    Returns:
        list: List of matplotlib figures - one for each category, one overall, and one for outlier removal
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import stats
    import matplotlib.colors as mcolors
    
    # Excluded categories
    excluded_categories = ["DFP", "FBBPF", "FBBPF (M)"]
    excluded_categories = ["DFP", "Circ Scrape"]
    #excluded_categories = []
    # Group depth delta values by category for both metrics
    depth1_by_category = {}
    depth2_by_category = {}
    all_depth1_values = []
    all_depth2_values = []
    
    for info in depth_info_list:
        category = info['depth_category']
        
        # Skip excluded categories
        if category in excluded_categories:
            continue
        
        # Get depth1_delta and handle None values
        depth1_delta = info.get('depth1_delta')
        if depth1_delta is not None:
            if category not in depth1_by_category:
                depth1_by_category[category] = []
            depth1_by_category[category].append(depth1_delta)
            all_depth1_values.append(depth1_delta)
        
        # Get depth2_delta and handle None values
        depth2_delta = info.get('depth2_delta')
        if depth2_delta is not None:
            if category not in depth2_by_category:
                depth2_by_category[category] = []
            depth2_by_category[category].append(depth2_delta)
            all_depth2_values.append((depth2_delta, category))  # Store category with the value
    
    # Get unique categories from both metrics, sort by number of data points
    all_categories = sorted(set(list(depth1_by_category.keys()) + list(depth2_by_category.keys())),
                           key=lambda cat: len(depth1_by_category.get(cat, [])) + len(depth2_by_category.get(cat, [])),
                           reverse=True)
    
    # Store all created figures
    figures = []
    
    # Calculate bin size of 0.02 and determine common range for all histograms
    bin_size = 0.02
    all_values = [v for v, _ in all_depth2_values] + all_depth1_values
    min_val = min(all_values) if all_values else -1.0
    max_val = max(all_values) if all_values else 1.0
    
    # Add padding to the range
    padding = 0.1 * (max_val - min_val)
    min_val -= padding
    max_val += padding
    
    # Calculate bin edges
    bins = np.arange(min_val, max_val + bin_size, bin_size)
    
    # Create histograms for each category
    for category in all_categories:
        # Get data for this category
        depth1_data = depth1_by_category.get(category, [])
        depth2_data = depth2_by_category.get(category, [])
        
        if not depth1_data and not depth2_data:
            continue
            
        # Create figure and axis
        fig, ax = plt.subplots(figsize=(12, 5))
        
        # Calculate statistics for depth1
        if depth1_data:
            depth1_mean = np.mean(depth1_data)
            depth1_std = np.std(depth1_data)
            depth1_2sigma_low = depth1_mean - 2*depth1_std
            depth1_2sigma_high = depth1_mean + 2*depth1_std
            
            # Plot histogram for depth1
            ax.hist(depth1_data, bins=bins, alpha=0.6, color='lightblue', 
                    edgecolor='black', label=f'Depth1 Delta (n={len(depth1_data)})')
            
            # Add vertical lines for mean and 2-sigma
            ax.axvline(depth1_mean, color='blue', linestyle='-', linewidth=2, 
                      label=f'Depth1 Mean: {depth1_mean:.3f}')
            ax.axvline(depth1_2sigma_low, color='blue', linestyle='--', linewidth=1.5, 
                      label=f'Depth1 2σ Lower: {depth1_2sigma_low:.3f}')
            ax.axvline(depth1_2sigma_high, color='blue', linestyle='--', linewidth=1.5, 
                      label=f'Depth1 2σ Upper: {depth1_2sigma_high:.3f}')
        
        # Calculate statistics for depth2
        if depth2_data:
            depth2_mean = np.mean(depth2_data)
            depth2_std = np.std(depth2_data)
            depth2_2sigma_low = depth2_mean - 2*depth2_std
            depth2_2sigma_high = depth2_mean + 2*depth2_std
            
            # Plot histogram for depth2
            ax.hist(depth2_data, bins=bins, alpha=0.6, color='lightgreen', 
                    edgecolor='black', label=f'Depth2 Delta (n={len(depth2_data)})')
            
            # Add vertical lines for mean and 2-sigma
            ax.axvline(depth2_mean, color='green', linestyle='-', linewidth=2, 
                      label=f'Depth2 Mean: {depth2_mean:.3f}')
            ax.axvline(depth2_2sigma_low, color='green', linestyle='--', linewidth=1.5, 
                      label=f'Depth2 2σ Lower: {depth2_2sigma_low:.3f}')
            ax.axvline(depth2_2sigma_high, color='green', linestyle='--', linewidth=1.5, 
                      label=f'Depth2 2σ Upper: {depth2_2sigma_high:.3f}')
        
        # Add reference line at zero
        ax.axvline(0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
        
        # Add statistical information as text
        text_info = []
        
        if depth1_data:
            text_info.append(f"Depth1 Delta (n={len(depth1_data)}):")
            text_info.append(f"  Mean: {depth1_mean:.3f}")
            text_info.append(f"  Std Dev: {depth1_std:.3f}")
            text_info.append(f"  2σ Range: [{depth1_2sigma_low:.3f}, {depth1_2sigma_high:.3f}]")
        
        if depth2_data:
            if text_info:  # Add spacing if depth1 info was added
                text_info.append("")
            text_info.append(f"Depth2 Delta (n={len(depth2_data)}):")
            text_info.append(f"  Mean: {depth2_mean:.3f}")
            text_info.append(f"  Std Dev: {depth2_std:.3f}")
            text_info.append(f"  2σ Range: [{depth2_2sigma_low:.3f}, {depth2_2sigma_high:.3f}]")
        
        # Add text box with statistics
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax.text(0.05, 0.95, '\n'.join(text_info), transform=ax.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        # Set labels and title
        ax.set_xlabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
        ax.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        ax.set_title(f'Distribution of Depth Delta for Category: {category}', fontsize=14, fontweight='bold')
        
        # Add legend
        ax.legend(loc='upper right')
        
        # Add grid with tighter spacing on x-axis
        ax.grid(alpha=0.3)
        ax.xaxis.grid(True, alpha=0.3, which='both', linestyle='-')
        # Add minor ticks for more precise grid on x-axis
        from matplotlib.ticker import MultipleLocator
        ax.xaxis.set_minor_locator(MultipleLocator(bin_size))
        
        # Adjust layout
        plt.tight_layout()
        
        # Add figure to list
        figures.append(fig)
    
    # Create overall histogram with all categories combined, but colored by category
    fig_overall, ax2 = plt.subplots(1, 1, figsize=(14, 7))
    
    # Get a colormap with distinct colors for each category
    cmap = plt.cm.get_cmap('tab20', len(all_categories))
    category_colors = {cat: mcolors.rgb2hex(cmap(i)) for i, cat in enumerate(all_categories)}
    
    # Extract depth2 values (without categories)
    all_depth2_values_only = [v for v, _ in all_depth2_values]
    
    if all_depth2_values:
        # Calculate overall statistics
        depth2_mean = np.mean(all_depth2_values_only)
        depth2_std = np.std(all_depth2_values_only)
        depth2_2sigma_low = depth2_mean - 2*depth2_std
        depth2_2sigma_high = depth2_mean + 2*depth2_std
        
        # Plot histograms by category
        legend_elements = []  # for custom legend
        
        # Group data by category for histogram
        for i, category in enumerate(all_categories):
            category_data = [v for v, cat in all_depth2_values if cat == category]
            if not category_data:
                continue
                
            color = category_colors[category]
            ax2.hist(category_data, bins=bins, alpha=0.7, color=color, 
                    edgecolor='black', label=category)
            
            # Add to custom legend
            legend_elements.append(Patch(facecolor=color, edgecolor='black',
                                         alpha=0.7, label=f'{category} (n={len(category_data)})'))
        
        # Add vertical lines for mean and 2-sigma
        ax2.axvline(depth2_mean, color='black', linestyle='-', linewidth=2, 
                  label=f'Mean: {depth2_mean:.3f}')
        ax2.axvline(depth2_2sigma_low, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Lower: {depth2_2sigma_low:.3f}')
        ax2.axvline(depth2_2sigma_high, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Upper: {depth2_2sigma_high:.3f}')
        
        # Add reference line at zero
        ax2.axvline(0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
        
        # Add text with statistics
        stats_text = (f"Overall Statistics (n = {len(all_depth2_values_only)})\n"
                     f"Mean = {depth2_mean:.3f}\n"
                     f"StdDev = {depth2_std:.3f}\n"
                     f"2σ Range = [{depth2_2sigma_low:.3f}, {depth2_2sigma_high:.3f}]")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax2.text(0.05, 0.95, stats_text, transform=ax2.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax2.set_title('Overall Distribution of Depth2 Delta by Category', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
        ax2.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add custom legend with category colors
        ax2.legend(handles=legend_elements + [
            plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label=f'Mean: {depth2_mean:.3f}'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label=f'2σ Range'),
            plt.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Zero Line')
        ], loc='upper right', fontsize=9)
        
        # Add grid with tighter spacing on x-axis
        ax2.grid(alpha=0.3)
        ax2.xaxis.grid(True, alpha=0.3, which='both', linestyle='-')
        # Add minor ticks for more precise grid on x-axis
        from matplotlib.ticker import MultipleLocator
        ax2.xaxis.set_minor_locator(MultipleLocator(bin_size))
    
    # Adjust layout
    plt.tight_layout()
    
    # Add overall figure to list
    figures.append(fig_overall)
    
    # NEW: Create a plot showing outlier removal to achieve 2-sigma within ±0.04
    if all_depth2_values:
        fig_outlier, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
        
        # Target 2-sigma range
        target_2sigma = 0.025
        
        # Calculate the required standard deviation to achieve 2-sigma = ±0.04
        required_std = target_2sigma / 2
        
        # Find how many outliers need to be removed
        all_depth2_sorted = np.sort(all_depth2_values_only)
        n_original = len(all_depth2_sorted)
        
        # Calculate z-scores for each point
        z_scores = np.abs((all_depth2_sorted - depth2_mean) / depth2_std)
        
        # Find the z-score threshold needed to achieve the target standard deviation
        # Start with all data and iteratively remove outliers
        kept_data = all_depth2_sorted.copy()
        removed_pct = 0
        z_threshold = 0
        
        # Iteratively remove outliers until we reach the desired standard deviation
        for percentile in np.arange(99, 0, -0.1):
            z_threshold = np.percentile(z_scores, percentile)
            kept_indices = z_scores <= z_threshold
            kept_data = all_depth2_sorted[kept_indices]
            
            if len(kept_data) < 10:  # Prevent removing too many points
                break
                
            kept_std = np.std(kept_data)
            removed_pct = 100 * (1 - len(kept_data) / n_original)
            
            if kept_std * 2 <= target_2sigma:
                break
        
        # For the left plot (original data)
        # Plot histograms by category
        legend_elements_orig = []
        
        for i, category in enumerate(all_categories):
            category_data = [v for v, cat in all_depth2_values if cat == category]
            if not category_data:
                continue
                
            color = category_colors[category]
            ax3.hist(category_data, bins=bins, alpha=0.7, color=color, 
                    edgecolor='black', label=category)
            
            # Add to custom legend
            legend_elements_orig.append(Patch(facecolor=color, edgecolor='black',
                                        alpha=0.7, label=f'{category} (n={len(category_data)})'))
        
        # Add vertical lines for mean and 2-sigma
        ax3.axvline(depth2_mean, color='black', linestyle='-', linewidth=2, 
                  label=f'Mean: {depth2_mean:.3f}')
        ax3.axvline(depth2_2sigma_low, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Lower: {depth2_2sigma_low:.3f}')
        ax3.axvline(depth2_2sigma_high, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Upper: {depth2_2sigma_high:.3f}')
        
        # Add target 2-sigma lines for comparison
        target_low = depth2_mean - target_2sigma/2
        target_high = depth2_mean + target_2sigma/2
        ax3.axvline(target_low, color='purple', linestyle='-.', linewidth=1.5, 
                  label=f'Target ±{target_2sigma} (Low): {target_low:.3f}')
        ax3.axvline(target_high, color='purple', linestyle='-.', linewidth=1.5, 
                  label=f'Target ±{target_2sigma} (High): {target_high:.3f}')
        
        # Add reference line at zero
        ax3.axvline(0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
        
        # Add text with statistics
        stats_text = (f"Original Statistics (n = {len(all_depth2_values_only)})\n"
                    f"Mean = {depth2_mean:.3f}\n"
                    f"StdDev = {depth2_std:.3f}\n"
                    f"2σ Range = [{depth2_2sigma_low:.3f}, {depth2_2sigma_high:.3f}]")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax3.set_title('Original Distribution', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add legend to left plot
        ax3.legend(handles=legend_elements_orig + [
            plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label=f'Mean: {depth2_mean:.3f}'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label=f'Original 2σ Range'),
            plt.Line2D([0], [0], color='purple', linestyle='-.', linewidth=1.5, label=f'Target ±{target_2sigma} Range'),
            plt.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Zero Line')
        ], loc='upper right', fontsize=8)
        
        # For the right plot (after removing outliers)
        # Calculate new statistics
        kept_mean = np.mean(kept_data)
        kept_std = np.std(kept_data)
        kept_2sigma_low = kept_mean - 2*kept_std
        kept_2sigma_high = kept_mean + 2*kept_std
        
        # Plot the histogram with kept data
        ax4.hist(kept_data, bins=bins, alpha=0.7, color='skyblue', 
                edgecolor='black', label=f'Filtered Data (n={len(kept_data)})')
        
        # Add vertical lines for mean and 2-sigma
        ax4.axvline(kept_mean, color='black', linestyle='-', linewidth=2, 
                  label=f'Mean: {kept_mean:.3f}')
        ax4.axvline(kept_2sigma_low, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Lower: {kept_2sigma_low:.3f}')
        ax4.axvline(kept_2sigma_high, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Upper: {kept_2sigma_high:.3f}')
        
        # Add reference line at zero
        ax4.axvline(0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
        
        # Add text with statistics
        filtered_stats_text = (
            f"Filtered Statistics (n = {len(kept_data)})\n"
            f"Mean = {kept_mean:.3f}\n"
            f"StdDev = {kept_std:.3f}\n"
            f"2σ Range = [{kept_2sigma_low:.3f}, {kept_2sigma_high:.3f}]\n\n"
            f"Removed {removed_pct:.1f}% of data points\n"
            f"to achieve 2σ within ±{target_2sigma}\n"
            f"Z-score threshold: {z_threshold:.2f}"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax4.text(0.05, 0.95, filtered_stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax4.set_title('Distribution after Outlier Removal for 2σ ≤ ±{target_2sigma}', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add grid with tighter spacing on x-axis for both plots
        for ax in [ax3, ax4]:
            ax.grid(alpha=0.3)
            ax.xaxis.grid(True, alpha=0.3, which='both', linestyle='-')
            ax.xaxis.set_minor_locator(MultipleLocator(bin_size))
        
        # Add legend to right plot
        ax4.legend(loc='upper right', fontsize=9)
        
        # Set the same x and y scale for both plots for fair comparison
        xlim = (min(ax3.get_xlim()[0], ax4.get_xlim()[0]), 
                max(ax3.get_xlim()[1], ax4.get_xlim()[1]))
        ylim = (0, max(ax3.get_ylim()[1], ax4.get_ylim()[1]))
        
        ax3.set_xlim(xlim)
        ax4.set_xlim(xlim)
        ax3.set_ylim(ylim)
        ax4.set_ylim(ylim)
        
        # Adjust layout
        plt.tight_layout()
        
        # Add the outlier removal figure to the list
        figures.append(fig_outlier)
    
    # Save figures to the specified directory
    import os
    save_dir = r"S:\snaps\1. graphs\Ite3_graphs"
    
    # Ensure directory exists
    os.makedirs(save_dir, exist_ok=True)
    
    # Save individual category figures
    for i, fig in enumerate(figures[:-2]):  # All except the last two figures (overall and outlier)
        category = all_categories[i]
        # Clean category name for filename
        clean_category = ''.join(c if c.isalnum() else '_' for c in category)
        filename = os.path.join(save_dir, f"Depth_Delta_{clean_category}.png")
        fig.savefig(filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    # Save overall figure
    overall_filename = os.path.join(save_dir, "Depth_Delta_Overall.png")
    figures[-2].savefig(overall_filename, dpi=300, bbox_inches='tight')
    print(f"Saved: {overall_filename}")
    
    # Save outlier removal figure
    if len(figures) > len(all_categories) + 1:  # If we created the outlier figure
        outlier_filename = os.path.join(save_dir, "Depth_Delta_Outlier_Removal.png")
        figures[-1].savefig(outlier_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {outlier_filename}")
    
    return figures

def plot_depth_delta_comparison_stacked(depth_info_list):
    """
    Modified version with stacked bars for both the left plot (original data)
    and the right plot (filtered data) to show category contributions consistently.
    """
    import numpy as np
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from scipy import stats
    import matplotlib.colors as mcolors
    from matplotlib.ticker import MultipleLocator
    import os
    
    # Excluded categories - using the most recent setting from the code
    excluded_categories = ["DFP", "Circ Scrape"]
    
    # Group depth delta values by category for both metrics
    depth1_by_category = {}
    depth2_by_category = {}
    all_depth1_values = []
    all_depth2_values = []
    
    for info in depth_info_list:
        category = info['depth_category']
        
        # Skip excluded categories
        if category in excluded_categories:
            continue
        
        # Get depth1_delta and handle None values
        depth1_delta = info.get('depth1_delta')
        if depth1_delta is not None and not np.isnan(depth1_delta):
            if category not in depth1_by_category:
                depth1_by_category[category] = []
            depth1_by_category[category].append(depth1_delta)
            all_depth1_values.append(depth1_delta)
        
        # Get depth2_delta and handle None values
        depth2_delta = info.get('depth2_delta')
        if depth2_delta is not None and not np.isnan(depth2_delta):
            if category not in depth2_by_category:
                depth2_by_category[category] = []
            depth2_by_category[category].append(depth2_delta)
            all_depth2_values.append((depth2_delta, category))  # Store category with the value
    
    # Check if we have enough data to proceed
    if not all_depth2_values:
        print("Error: No valid depth2_delta values found after filtering.")
        return None
    
    # Get unique categories from both metrics
    all_categories = sorted(set(list(depth1_by_category.keys()) + list(depth2_by_category.keys())),
                           key=lambda cat: len(depth1_by_category.get(cat, [])) + len(depth2_by_category.get(cat, [])),
                           reverse=True)
    
    # Extract depth2 values (without categories)
    all_depth2_values_only = [v for v, _ in all_depth2_values]
    
    # Get a colormap with distinct colors for each category
    cmap = plt.cm.get_cmap('tab20', len(all_categories))
    category_colors = {cat: mcolors.rgb2hex(cmap(i)) for i, cat in enumerate(all_categories)}
    
    # Create outlier removal plot with different ranges
    # Create figure with two subplots
    fig_outlier, (ax3, ax4) = plt.subplots(1, 2, figsize=(20, 8))
    
    # Target 2-sigma range
    target_2sigma = 0.025
    
    # Calculate overall statistics
    depth2_mean = np.mean(all_depth2_values_only)
    depth2_std = np.std(all_depth2_values_only)
    depth2_2sigma_low = depth2_mean - 2*depth2_std
    depth2_2sigma_high = depth2_mean + 2*depth2_std
    
    # Find how many outliers need to be removed
    all_depth2_sorted = np.sort(all_depth2_values_only)
    n_original = len(all_depth2_sorted)
    
    # Calculate z-scores for each point
    z_scores = np.abs((all_depth2_sorted - depth2_mean) / depth2_std)
    
    try:
        # Iteratively remove outliers until we reach the desired standard deviation
        kept_data = all_depth2_sorted.copy()
        kept_data_with_category = [(v, cat) for v, cat in all_depth2_values]  # Keep category info for filtered data
        kept_categories = [cat for _, cat in all_depth2_values]  # Category for each data point
        removed_pct = 0
        z_threshold = 0
        
        for percentile in np.arange(99, 0, -0.1):
            z_threshold = np.percentile(z_scores, percentile)
            kept_indices = z_scores <= z_threshold
            
            # Update both the value-only array and the value-with-category array
            kept_data = all_depth2_sorted[kept_indices]
            
            # Find indices in original all_depth2_values that correspond to kept_indices
            # This is tricky because all_depth2_sorted is sorted, so indices don't match directly
            # We need to map back to the original unsorted data
            
            # Create a mapping from sorted values to original indices
            # This is a simplification and might not work perfectly for duplicate values
            sorted_to_orig_mapping = {i: all_depth2_values_only.index(val) for i, val in enumerate(all_depth2_sorted)}
            
            # Find original indices that correspond to kept indices
            kept_orig_indices = [sorted_to_orig_mapping[i] for i, keep in enumerate(kept_indices) if keep]
            
            # Filter the value-with-category array using these indices
            kept_data_with_category = [all_depth2_values[i] for i in kept_orig_indices]
            
            if len(kept_data) < 10:  # Prevent removing too many points
                break
                
            kept_std = np.std(kept_data)
            removed_pct = 100 * (1 - len(kept_data) / n_original)
            
            if kept_std * 2 <= target_2sigma:
                break
        
        # Create regular bins for the left plot
        regular_bin_size = 0.02
        padding = 0.1 * (max(all_depth2_values_only) - min(all_depth2_values_only))
        min_val = min(all_depth2_values_only) - padding
        max_val = max(all_depth2_values_only) + padding
        regular_bins = np.arange(min_val, max_val + regular_bin_size, regular_bin_size)
        
        # Create focused bins for right plot only
        focused_bin_size = 0.002
        focused_bins = np.arange(-0.05, 0.05 + focused_bin_size, focused_bin_size)
        
        # Collect category data for stacked histogram (left plot)
        left_category_data_list = []
        for category in all_categories:
            category_data = [v for v, cat in all_depth2_values if cat == category]
            if category_data:  # Only include non-empty categories
                left_category_data_list.append((category, category_data))
        
        # Sort by size (largest category first)
        left_category_data_list.sort(key=lambda x: len(x[1]), reverse=True)
        
        # For the left plot - STACKED histogram
        legend_elements_orig = []
        
        # Create stacked histogram for left plot
        if left_category_data_list:
            n, bins, patches = ax3.hist(
                [data for _, data in left_category_data_list],
                bins=regular_bins,
                stacked=True,
                alpha=0.7,
                label=[cat for cat, _ in left_category_data_list],
                color=[category_colors[cat] for cat, _ in left_category_data_list],
                edgecolor='black'
            )
            
            # Create legend elements for left plot
            for i, (category, data) in enumerate(left_category_data_list):
                legend_elements_orig.append(Patch(
                    facecolor=category_colors[category],
                    edgecolor='black',
                    alpha=0.7,
                    label=f'{category} (n={len(data)})'
                ))
        
        # Add vertical lines for mean and 2-sigma to left plot
        ax3.axvline(depth2_mean, color='black', linestyle='-', linewidth=2, 
                  label=f'Mean: {depth2_mean:.3f}')
        ax3.axvline(depth2_2sigma_low, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Lower: {depth2_2sigma_low:.3f}')
        ax3.axvline(depth2_2sigma_high, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Upper: {depth2_2sigma_high:.3f}')
        
        # Add target 2-sigma lines for comparison
        target_low = depth2_mean - target_2sigma
        target_high = depth2_mean + target_2sigma
        ax3.axvline(target_low, color='purple', linestyle='-.', linewidth=1.5, 
                  label=f'Target ±{target_2sigma} (Low): {target_low:.3f}')
        ax3.axvline(target_high, color='purple', linestyle='-.', linewidth=1.5, 
                  label=f'Target ±{target_2sigma} (High): {target_high:.3f}')
        
        # Add reference line at zero
        ax3.axvline(0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
        
        # Add text with statistics to left plot
        stats_text = (f"Original Statistics (n = {len(all_depth2_values_only)})\n"
                    f"Mean = {depth2_mean:.3f}\n"
                    f"StdDev = {depth2_std:.3f}\n"
                    f"2σ Range = [{depth2_2sigma_low:.3f}, {depth2_2sigma_high:.3f}]")
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax3.text(0.05, 0.95, stats_text, transform=ax3.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax3.set_title('Original Distribution - Stacked by Category', fontsize=14, fontweight='bold')
        ax3.set_xlabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
        ax3.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Add legend to left plot
        ax3.legend(handles=legend_elements_orig + [
            plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label=f'Mean: {depth2_mean:.3f}'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label=f'Original 2σ Range'),
            plt.Line2D([0], [0], color='purple', linestyle='-.', linewidth=1.5, label=f'Target ±{target_2sigma} Range'),
            plt.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Zero Line')
        ], loc='upper right', fontsize=8)
        
        # For the right plot (after removing outliers) - ALSO STACKED
        if np.isnan(np.mean(kept_data)) or np.isnan(np.std(kept_data)):
            print("Warning: NaN values detected in filtered data statistics.")
            # Filter out NaNs from both arrays
            valid_indices = ~np.isnan(kept_data)
            kept_data = kept_data[valid_indices]
            kept_data_with_category = [item for i, item in enumerate(kept_data_with_category) if valid_indices[i]] if len(valid_indices) == len(kept_data_with_category) else []
            
            if len(kept_data) == 0:
                print("Error: No valid data points after filtering.")
                return fig_outlier
        
        # Calculate new statistics for right plot
        kept_mean = np.mean(kept_data)
        kept_std = np.std(kept_data)
        kept_2sigma_low = kept_mean - 2*kept_std
        kept_2sigma_high = kept_mean + 2*kept_std
        
        # Collect category data for stacked histogram (right plot)
        right_category_data_list = []
        for category in all_categories:
            category_data = [v for v, cat in kept_data_with_category if cat == category]
            if category_data:  # Only include non-empty categories
                right_category_data_list.append((category, category_data))
        
        # Sort by size (largest category first) for right plot
        right_category_data_list.sort(key=lambda x: len(x[1]), reverse=True)
        
        # Create legend elements for right plot
        legend_elements_right = []
        
        # Create stacked histogram for right plot
        if right_category_data_list:
            n, bins, patches = ax4.hist(
                [data for _, data in right_category_data_list],
                bins=focused_bins,
                stacked=True,
                alpha=0.7,
                label=[cat for cat, _ in right_category_data_list],
                color=[category_colors[cat] for cat, _ in right_category_data_list],
                edgecolor='black'
            )
            
            # Create legend elements for right plot
            for i, (category, data) in enumerate(right_category_data_list):
                legend_elements_right.append(Patch(
                    facecolor=category_colors[category],
                    edgecolor='black',
                    alpha=0.7,
                    label=f'{category} (n={len(data)})'
                ))
        else:
            # If categorization failed, just plot all the data
            ax4.hist(kept_data, bins=focused_bins, alpha=0.7, color='skyblue', 
                    edgecolor='black', label=f'Filtered Data (n={len(kept_data)})')
            
            legend_elements_right = [Patch(
                facecolor='skyblue',
                edgecolor='black',
                alpha=0.7,
                label=f'Filtered Data (n={len(kept_data)})'
            )]
        
        # Add vertical lines for mean and 2-sigma to right plot
        ax4.axvline(kept_mean, color='black', linestyle='-', linewidth=2, 
                  label=f'Mean: {kept_mean:.3f}')
        ax4.axvline(kept_2sigma_low, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Lower: {kept_2sigma_low:.3f}')
        ax4.axvline(kept_2sigma_high, color='black', linestyle='--', linewidth=1.5, 
                  label=f'2σ Upper: {kept_2sigma_high:.3f}')
        
        # Add reference line at zero to right plot
        ax4.axvline(0, color='red', linestyle='-', alpha=0.5, label='Zero Line')
        
        # Add text with statistics to right plot
        filtered_stats_text = (
            f"Filtered Statistics (n = {len(kept_data)})\n"
            f"Mean = {kept_mean:.3f}\n"
            f"StdDev = {kept_std:.3f}\n"
            f"2σ Range = [{kept_2sigma_low:.3f}, {kept_2sigma_high:.3f}]\n\n"
            f"Removed {removed_pct:.1f}% of data points\n"
            f"to achieve 2σ within ±{target_2sigma}\n"
            f"Z-score threshold: {z_threshold:.2f}"
        )
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.7)
        ax4.text(0.05, 0.95, filtered_stats_text, transform=ax4.transAxes, fontsize=10,
                verticalalignment='top', bbox=props)
        
        ax4.set_title('Distribution after Outlier Removal - Stacked by Category', fontsize=14, fontweight='bold')
        ax4.set_xlabel('Depth Delta (predicted - reported)', fontsize=12, fontweight='bold')
        ax4.set_ylabel('Frequency', fontsize=12, fontweight='bold')
        
        # Set x-axis limits to the focused range for right plot
        ax4.set_xlim(-0.05, 0.05)
        
        # Add minor ticks for more precise grid on x-axis for both plots
        # Use different grid spacing for each plot
        ax3.grid(alpha=0.3)
        ax3.xaxis.grid(True, alpha=0.3, which='both', linestyle='-')
        ax3.xaxis.set_minor_locator(MultipleLocator(regular_bin_size/2))  # For left plot
        
        ax4.grid(alpha=0.3)
        ax4.xaxis.grid(True, alpha=0.3, which='both', linestyle='-')
        ax4.xaxis.set_minor_locator(MultipleLocator(focused_bin_size))  # For right plot
        
        # Add legend to right plot
        ax4.legend(handles=legend_elements_right + [
            plt.Line2D([0], [0], color='black', linestyle='-', linewidth=2, label=f'Mean: {kept_mean:.3f}'),
            plt.Line2D([0], [0], color='black', linestyle='--', linewidth=1.5, label=f'2σ Range'),
            plt.Line2D([0], [0], color='red', linestyle='-', alpha=0.5, label='Zero Line')
        ], loc='upper right', fontsize=8)
        
        # Adjust layout
        plt.tight_layout()
        
        # Save the outlier removal figure
        save_dir = r"S:\snaps\1. graphs\Ite3_graphs"
        os.makedirs(save_dir, exist_ok=True)
        outlier_filename = os.path.join(save_dir, "Depth_Delta_Outlier_Removal_Both_Stacked.png")
        fig_outlier.savefig(outlier_filename, dpi=300, bbox_inches='tight')
        print(f"Saved: {outlier_filename}")
        
    except Exception as e:
        print(f"Error during plotting: {str(e)}")
        import traceback
        traceback.print_exc()
    
    return fig_outlier

def plot_debris_fretting_depth(overcalled_flaws, output_path=None):
    """
    Creates and saves a plot showing the distribution of depth values
    for Debris fretting flaws only. Values over 0.5 are grouped as ">0.5".
    The histogram bins are 0.01 wide.
    
    Args:
        overcalled_flaws (list): List of flaw dictionaries containing 'flaw_type' and 'depth'
        output_path (str, optional): Path to save the plot. If None, will only display.
                                    Default is None.
    
    Returns:
        tuple: (fig, ax) - matplotlib figure and axis objects
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    import numpy as np
    
    # Convert to DataFrame if not already
    if not isinstance(overcalled_flaws, pd.DataFrame):
        df = pd.DataFrame([(flaw['scan_id'], flaw['ind_num'], flaw['depth'], flaw['flaw_type']) 
                          for flaw in overcalled_flaws], 
                         columns=['scan_id', 'ind_num', 'depth', 'flaw_type'])
    else:
        df = overcalled_flaws.copy()
    
    # Filter for only "Debris" flaws
    debris_fretting_df = df[df['flaw_type'] == 'Debris'].copy()
    
    if len(debris_fretting_df) == 0:
        print("No 'Debris' flaws found in the data.")
        return None, None
    
    # Create a column for plotting - values over 0.5 will be capped
    debris_fretting_df['plot_depth'] = debris_fretting_df['depth'].apply(
        lambda x: 0.5 if x > 0.5 else x
    )
    
    # Count how many values are over 0.5
    over_05_count = sum(debris_fretting_df['depth'] > 0.5)
    
    # Create bins with 0.01 width, starting from 0.1
    bins = np.arange(0.1, 0.51, 0.01)
    
    # Create the plot - narrower width
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # Plot histogram with 0.01 width bins
    counts, edges, bars = ax.hist(debris_fretting_df['plot_depth'], bins=bins, 
                                alpha=0.7, label='Depth Distribution')
    
    # If there are values over 0.5, add a special bar at the end (using same color)
    if over_05_count > 0:
        # Add a special bar for >0.5
        bar_width = edges[1] - edges[0]
        ax.bar(0.5 + bar_width/2, over_05_count, width=bar_width, 
               color=bars[0].get_facecolor(), alpha=0.7, label='>0.5')
        
        # Add text annotation for >0.5 with larger font
        ax.text(0.5 + bar_width/2, over_05_count + 0.5, f">0.5: {over_05_count}", 
                ha='center', va='bottom', fontweight='bold', fontsize=18)
    
    ax.set_title('Distribution of Depth for Debris Flaws', fontsize=24)
    ax.set_xlabel('Depth', fontsize=18)
    ax.set_ylabel('Frequency', fontsize=18)
    ax.grid(True, alpha=0.3)
    
    # Format x-axis to show the 0.01 increments
    ax.set_xticks(np.arange(0.1, 0.51, 0.05))
    ax.set_xticklabels([f'{x:.2f}' for x in np.arange(0.1, 0.51, 0.05)], fontsize=15)
    
    # Set y-axis to show integers from 0 to 10
    ax.set_yticks(range(11))
    ax.set_yticklabels([str(i) for i in range(11)], fontsize=15)
    ax.set_ylim(0, 10)
    
    # No mean line as requested
    
    # Make legend text larger
    ax.legend(fontsize=15)
    plt.tight_layout()
    
    # Save the plot if a path is provided
    if output_path:
        plt.savefig(output_path, dpi=300)
        print(f"Plot saved to {output_path}")
    
    # Print some statistics
    print(f"Number of Debris flaws: {len(debris_fretting_df)}")
    print(f"Number of flaws with depth >0.5: {over_05_count}")
    print(f"Mean depth: {debris_fretting_df['depth'].mean():.3f}")
    print(f"Min depth: {debris_fretting_df['depth'].min():.3f}")
    print(f"Max depth: {debris_fretting_df['depth'].max():.3f}")
    
    return fig, ax

if __name__ == "__main__":
   
    axial_range = None
    from db.db import ScanDatabase
    my_db = ScanDatabase(r'db\UT_db.json')
    scans_db = my_db.scans
   
    # Debris__________________________________________________________________________________
    query_results_debris = my_db.flexible_query(
    query_type='reported_flaws',
    flaw_filters={'flaw_type': ['Debris', 'FBBPF', 'BM_FBBPF']}, 
    scan_filters={'train_val': ["train","val"]}
)
    query_results_debris = my_db.flexible_query(
    query_type='reported_flaws',
    flaw_filters={'flaw_type': 'FBBPF'}, 
    scan_filters={'train_val': ["train","val"]}
)
    depth_info_list = []
    depth1_delta = []
    depth2_delta = []
    for ind, query_result in enumerate(query_results_debris):
    # Extract depth information
        try:
            depth_info_list = extract_depth_info(query_result, depth_info_list, depth2_buffer=0)
        except:
            continue
    depth_info_list = [item for item in depth_info_list if item.get('classified') != False]
    
    debris_null_depth_diff = [
    {
        'flaw': reported_flaw,
        'scan_id': result['scan']['scan_id']
    }
    for result in query_results_debris
    for reported_flaw in result['scan']['reported_flaws']
    if reported_flaw.get('flaw_type') == 'Debris' and 
       reported_flaw.get('metrics', {}).get('depth_diff') is None
]
    
    #pd.DataFrame(depth_info_list).to_excel(r"S:\snaps\1. graphs\Ite3_graphs\Depth_Analysis_Results_Debris.xlsx", index=False)
    category_counts = count_depth_categories(depth_info_list)
    plot_category_distribution(category_counts)

    fig = plot_depth_delta_comparison(depth_info_list)
    fig = plot_depth_delta_comparison_stacked(depth_info_list)
    # fig = plot_depth_delta_by_year_category(depth_info_list)
    # plt.show()
    
    # fig = plot_depth_delta_by_channel(depth_info_list)
    # plt.show()

    # __________________________________________________________________________________
    outfolder = r'S:\snaps\3.outlier'
    root_folder = r'C:\Users\LIUJA\Documents\GitHub\PTFAST-v1.2.0\auto-analysis-results\val3\depth'  # Update this to your actual root folder path
    
    # Copy outlier images for V category
    fnames, V_inds = copy_outlier_images(
        depth_info_list, 
        categories='BM_FBBPF', 
        delta_threshold=0.05,
        root_folder=root_folder,
        outfolder=outfolder,
        delta_field='depth2_delta'
    )
    
    print(f"Identified {len(fnames)} category outliers")
    
    # overcalling analysis__________________________________________________________________________________
    # overcalling analysis
    
    query_results_val = my_db.flexible_query(
    query_type='reported_flaws',
    scan_filters={'train_val': "val"}
)
    
    query_results_val = my_db.flexible_query(
    query_type='flaws',
    scan_filters={"outage_number": "D2421"}
)
    overcalled = []
    for ind, query_result in enumerate(query_results_val):
        # Pass the index to the function
        scan_flaws = extract_and_clean_extra_flaws(query_result, ind)
        overcalled.extend(scan_flaws)
   
    pd.DataFrame([(flaw['scan_id'], flaw['ind_num'], flaw['depth'], flaw['flaw_type']) for flaw in overcalled], columns=['scan_id', 'ind_num', 'depth', 'flaw_type']).to_excel(r'auto-analysis-results\val3\overcalled_flaws.xlsx', index=False)
    
    plot_debris_fretting_depth(overcalled, r'auto-analysis-results\D2421\debris_fretting_depth_distribution.png')
    
    raw_depths_val = "0.1342 0.1195 0.1325 0.1152 0.1134 0.1292 0.1065 0.1213 0.1104 0.107 0.1403 (2) 0.1289 (2) 0.1041 (2) 0.1 0.1194 (2) 0.1253 0.1134 0.9391 0.123 (2) 0.1423 (2) 0.1232 0.124 0.1399 0.1069 0.1057 0.124 0.1223 (2) 0.1334 (2) 0.1046 (2) 0.1167 0.1092"

    # Parse the data, eliminating (2) notations
    depths = []
    tokens = raw_depths_val.split()
    i = 0

    while i < len(tokens):
        # Check if the current token is a float
        try:
            depth = float(tokens[i])
            depths.append(depth)
        except ValueError:
            # Not a valid float, skip it
            pass
        
        # Move to next token
        i += 1

    # Create a list of dictionaries in the required format
    overcalled_flaws_val = []
    for i, depth in enumerate(depths):
        flaw = {
            'scan_id': f'Scan_{i//3}',  # Assign a dummy scan ID (3 flaws per scan)
            'ind_num': i,
            'depth': depth,
            'flaw_type': 'Debris'  # All are debris fretting
        }
        overcalled_flaws_val.append(flaw)
        
    plot_debris_fretting_depth(overcalled_flaws_val, r'auto-analysis-results\val3\V1_overcalled_depth_distribution.png')
    
    raw_depths_d = "0.124 0.1371 0.1094 0.1062 0.1574 0.1257 0.1413 0.1085 0.1275 0.1324 0.1171 0.1156 0.1294 0.1074 0.1136 0.1237 0.1153 0.1216 0.1386 0.1337 0.1514 0.1325 0.127 0.2543 0.1528 0.1119 0.1594 0.2042 0.2042 0.1995 (2)"
    
      # Parse the data, eliminating (2) notations
    depths = []
    tokens = raw_depths_d.split()
    i = 0

    while i < len(tokens):
        # Check if the current token is a float
        try:
            depth = float(tokens[i])
            depths.append(depth)
        except ValueError:
            # Not a valid float, skip it
            pass
        
        # Move to next token
        i += 1

    # Create a list of dictionaries in the required format
    overcalled_flaws_d = []
    for i, depth in enumerate(depths):
        flaw = {
            'scan_id': f'Scan_{i//3}',  # Assign a dummy scan ID (3 flaws per scan)
            'ind_num': i,
            'depth': depth,
            'flaw_type': 'Debris'  # All are debris fretting
        }
        overcalled_flaws_d.append(flaw)
        
    plot_debris_fretting_depth(overcalled_flaws_d, r'auto-analysis-results\D2421\V1_overcalled_depth_distribution.png')