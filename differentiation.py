import pickle
import pandas as pd

def check_overlap(flaws, possible_cats, overlap_threshold=0.5):
    """
    Checks for overlap in flaws by calculating Intersection over Union (IoU).

    Parameters:
    ----------
    flaws (list): A list of Flaw objects.
    possible_cats (list): A list of possible categories for each flaw.
    overlap_threshold (float, optional): The threshold for IoU. Defaults to 0.5.

    Returns:
    -------
    possible_cats (list): An updated list of possible categories for each flaw.
    flaws_to_keep (list): The list of flaws after removing overlapped ones.
    """
    flaws_to_delete = set()
    
    for i, flaw1 in enumerate(flaws):
        # Get bounding box coordinates for flaw1
        x1_min, y1_min = flaw1.x_start, flaw1.y_start
        x1_max, y1_max = flaw1.x_end, flaw1.y_end
        area1 = (x1_max - x1_min) * (y1_max - y1_min)

        for j in range(i + 1, len(flaws)):
            flaw2 = flaws[j]
            
            # Skip if either flaw is already marked for deletion
            if i in flaws_to_delete or j in flaws_to_delete:
                continue
                
            # Get bounding box coordinates for flaw2
            x2_min, y2_min = flaw2.x_start, flaw2.y_start
            x2_max, y2_max = flaw2.x_end, flaw2.y_end
            area2 = (x2_max - x2_min) * (y2_max - y2_min)

            # Calculate intersection
            x_left = max(x1_min, x2_min)
            y_top = max(y1_min, y2_min)
            x_right = min(x1_max, x2_max)
            y_bottom = min(y1_max, y2_max)

            intersection_area = max(0, x_right - x_left) * max(0, y_bottom - y_top)
            iou = intersection_area / float(area1 + area2 - intersection_area)

            if iou >= overlap_threshold:
                # Add flaw types to possible categories
                if flaw1.flaw_type != 'Note 1':
                    possible_cats[i].append(flaw1.flaw_type)
                    possible_cats[j].append(flaw1.flaw_type)
                if flaw2.flaw_type != 'Note 1':
                    possible_cats[i].append(flaw2.flaw_type)
                    possible_cats[j].append(flaw2.flaw_type)

                # Ensure both flaws have the same possible categories
                if len(possible_cats[i]) > len(possible_cats[j]):
                    possible_cats[j] = possible_cats[i].copy()
                elif len(possible_cats[j]) > len(possible_cats[i]):
                    possible_cats[i] = possible_cats[j].copy()

                # Keep the flaw with higher confidence, mark the other for deletion
                if flaw1.confidence > flaw2.confidence:
                    flaw1.flaw_type = 'Note 1 - overlapped'
                    flaw2.flaw_type = 'Note 1 - overlapped_del'
                    flaws_to_delete.add(j)
                else:
                    flaw1.flaw_type = 'Note 1 - overlapped_del'
                    flaw2.flaw_type = 'Note 1 - overlapped'
                    flaws_to_delete.add(i)
                    break  # Break inner loop if flaw1 is marked for deletion

    # Create list of indices to keep
    indices_to_keep = [i for i in range(len(flaws)) if i not in flaws_to_delete]
    
    # Filter possible_cats list
    filtered_possible_cats = [possible_cats[i] for i in indices_to_keep]
    
    # Filter flaws list
    flaws_to_keep = [flaws[i] for i in indices_to_keep]
    
    return filtered_possible_cats, flaws_to_keep

def remove_duplicates_ordered(category_list):
    """
    Removes duplicate strings from category_list while preserving the order of the original elements.

    Parameters:
    ----------
        category_list (list of str): A list containing strings that may have duplicates.

    Returns:
    -------
        list of str: A new list containing the unique strings from the input list, preserving the order of their first occurrence.
        This is important because the first occurrence of a string is the classification returned by the object detection model.
    """
    seen = set()
    return [x for x in category_list if not (x in seen or seen.add(x))]


def generate_note(possible_cats):
    '''
    This function takes in the possible categories and generates the note.
    Parameters:
    ----------
    possible_cats : list
        list of possible categories
    Returns:
    -------
    note : str
    '''
    cats = remove_duplicates_ordered(possible_cats)
    if len(cats) > 1:
        msg = f"Warning: Differentiation between {' and '.join(cats)} is inconclusive."
    else:
        msg = 'N/A'
    return msg


def del_overlapped(pred_coords, comments, possible_cats):
    """
    Delete elements that have 'Note 1 - overlapped_del' as their classification.

    Parameters:
    ----------
    pred_coords (list): List of prediction coordinates.
    comments (list): List of comments.
    possible_cats (list): List of possible categories.

    Returns:
    -------
    tuple: A tuple containing updated lists and the indices of keep elements.
    """
    indices_to_keep = [i for i, pred in enumerate(pred_coords) if pred[4] != 'Note 1 - overlapped_del']

    pred_coords = [pred_coords[i] for i in indices_to_keep]
    comments = [comments[i] for i in indices_to_keep]
    possible_cats = [possible_cats[i] for i in indices_to_keep]

    return pred_coords, comments, possible_cats, indices_to_keep

def update_classifications_and_cats(flaw, debris_prob, CC_prob, possible_cats, diff_config):
    """
    Updates the classifications and possible categories based on Debris and CC probabilities.

    Parameters:
        pred (list): A list of prediction parameters.
        i (int): Index of the current prediction.
        debris_prob (float): The probability of Debris.
        CC_prob (float): The probability of CC.
        pred_coords (list): A list of prediction coordinates from the auto_analysis module.
        possible_cats (list): A list of possible categories for each prediction.
        diff_config (Config): A configuration object containing debris_threshold, CC_threshold, and overlap_threshold.
    """

    classification = flaw.flaw_type
    comment = generate_note(possible_cats)
    if (classification == "CC" and debris_prob >= diff_config.debris_threshold) or \
       (classification == "Debris" and CC_prob >= diff_config.CC_threshold):

        # Change classification to Note 1
        flaw.note1 = comment
        flaw.possible_cat = ['CC', 'Debris'] if classification == "CC" else ['Debris', 'CC']

def run_diff_module(scan, config):
    """
    Runs the differentiation model on given flaws and configuration.

    Parameters:
    ----------
    scan (Scan): A Scan object containing flaws.
    config (Config): A configuration object containing differentiation settings.

    Returns:
    -------
    None: The function updates the scan object's flaws in-place.

    This function applies a differentiation model to the flaws in the scan object,
    updating their classifications and possible categories based on the model's predictions.
    """
    diff_config = config.characterization.diff
    diff_model_columns = ['Axial Start', 'Rotary Start', 'Length (mm)', 'Width (deg)', 'lw ratio']
    
    # Load the differentiation model
    with open(diff_config.rf_model_path, 'rb') as model_file:
        loaded_model = pickle.load(model_file)

    # Prepare input data for all flaws at once
    data = [[
        flaw.axial_start,
        flaw.rotary_start,
        flaw.length - diff_config.length_buffer,
        flaw.width,
        (flaw.length - diff_config.length_buffer) / flaw.width
    ] for flaw in scan.flaws]

    # Run differentiation model for all flaws at once
    test_df = pd.DataFrame(data, columns=diff_model_columns)
    probabilities = loaded_model.predict_proba(test_df)

    # Update classifications and categories for all flaws
    for i, (flaw, (CC_prob, debris_prob)) in enumerate(zip(scan.flaws, probabilities)):
        update_classifications_and_cats(flaw, debris_prob, CC_prob, diff_config)
        scan.flaws[i] = flaw

     # Check for overlapping flaws if there are multiple flaws
    if len(scan.flaws) > 1:
        # Initialize possible categories list
        possible_cats = [flaw.possible_category.copy() if hasattr(flaw, 'possible_category') and flaw.possible_category else [] for flaw in scan.flaws]
        
        # Check for overlaps
        overlap_threshold = getattr(diff_config, 'overlap_threshold', 0.5)
        possible_cats, flaws_to_keep = check_overlap(scan.flaws, possible_cats, overlap_threshold)
        
        # Generate comments based on possible categories
        comments = [generate_note(cats) for cats in possible_cats]
        
        # Update the flaws in the scan
        for i, flaw in enumerate(flaws_to_keep):
            flaw.possible_category = possible_cats[i]
            flaw.note1 = comments[i]
        
        # Replace the flaws list with the updated one
        scan.flaws = flaws_to_keep
        
def update_classifications_and_cats(flaw, debris_prob, CC_prob, diff_config):
    """
    Updates the classifications and possible categories based on Debris and CC probabilities.

    Parameters:
    ----------
    flaw (Flaw): A Flaw object to be updated.
    debris_prob (float): The probability of Debris.
    CC_prob (float): The probability of CC.
    diff_config (Config): A configuration object containing debris_threshold and CC_threshold.
    """
    if (flaw.flaw_type == "CC" and debris_prob >= diff_config.debris_threshold) or \
       (flaw.flaw_type == "Debris" and CC_prob >= diff_config.CC_threshold):
        new_categories = ['CC', 'Debris'] if flaw.flaw_type == "CC" else ['Debris', 'CC']
        flaw.update_possible_category(new_categories)
        flaw.note1 = generate_note(['CC', 'Debris'])
    else: 
        flaw.update_possible_category([flaw.flaw_type])
        