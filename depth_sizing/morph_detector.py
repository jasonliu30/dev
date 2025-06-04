"""
Ultrasonic Scan Analysis Module
-------------------------------
This module provides tools for analyzing ultrasonic scan data,
detecting flaws using B-scan data (views), and calculating depth measurements.
"""

import numpy as np
import cv2
import os
import matplotlib.pyplot as plt
from collections import Counter
import tensorflow as tf
import copy
from typing import Dict, List, Tuple, Optional, Any, Union


class FlawMorphologyAnalyzer:
    """Main class for processing ultrasonic scan data and detecting flaws in B-scans."""
    
    def __init__(self, output_dir: str = "depth_vision_snap"):
        """
        Initialize the FlawMorphologyAnalyzer.
        
        Args:
            output_dir: Directory for saving output visualizations and results
        """
        self.output_dir = output_dir
        os.makedirs(output_dir, exist_ok=True)
        
    def get_flaw_ranges(self, scan, probes_data) -> Dict:
        """
        Extract circumferential, axial, and time ranges for debris flaws.
        
        Args:
            scan: Scan object containing flaws
            probes_data: Dictionary containing probe data
            
        Returns:
            Dictionary with ranges for each flaw
        """
        ranges = {}
        
        # Process both debris and CC flaws
        selected_flaws = [flaw for flaw in scan.flaws if 
                        flaw.flaw_type == 'Debris' or "Debris" in flaw.possible_category or
                        flaw.flaw_type == 'CC' or "CC" in flaw.possible_category]
        scan_circ_start = 0
        scan_circ_end = probes_data['APC'].data.shape[1]
        for flaw in selected_flaws:
            try:
                # Calculate circumferential range
                circ_start = flaw.x_start
                circ_end = flaw.x_end
                circ_length = circ_end - circ_start
                circ_range = [max(scan_circ_start, circ_start - circ_length), min(scan_circ_end, circ_end + circ_length)]
                
                # Axial range is directly from flaw
                axial_range = [flaw.y_start, flaw.y_end]
                
                # Find time range based on max FWG using multiple slices
                time_range_nb1 = self._calculate_time_range(probes_data, 'NB1', circ_range, axial_range)
                time_range_nb2 = self._calculate_time_range(probes_data, 'NB2', circ_range, axial_range)
                
                ranges[flaw.ind_num] = {
                    'circ_range': circ_range,
                    'axial_range': axial_range,
                    'time_range_nb1': time_range_nb1,
                    'time_range_nb2': time_range_nb2
                }
                
            except Exception as e:
                print(f'Error processing flaw {flaw}: {str(e)}')
                continue
                
        return ranges
    
    def _calculate_time_range(self, probes_data, probe, circ_range, axial_range):
        """Calculate time range based on FWG analysis."""
        # Using full NB1 data at flaw location with padding
        NB1_bscan_flaw = probes_data[probe].data[:, circ_range[0]:circ_range[1], :]

        # Get indices for first, middle and last slices
        first_idx = axial_range[0]
        middle_idx = axial_range[0] + len(range(axial_range[0], axial_range[-1])) // 2
        last_idx = axial_range[-1]

        # Find max FWG positions for each slice
        max_fwg_first = np.argmax(NB1_bscan_flaw[first_idx][0])
        max_fwg_middle = np.argmax(NB1_bscan_flaw[middle_idx][0])
        max_fwg_last = np.argmax(NB1_bscan_flaw[last_idx][0])

        # Find median to exclude potential outliers
        max_fwg = int(np.median([max_fwg_first, max_fwg_middle, max_fwg_last]))

        # Set time range around the determined max_fwg
        return [max_fwg - 50, max_fwg + 250]
    
    def calc_vision_depth(self, probes_data, model, scan_name, scan, reported_depth=None, 
                          outlier_diff_range=0.4) -> Dict:
        """
        Process depth profiles using probe data with provided ranges.
        
        Args:
            probes_data: Dictionary containing NB1 and NB2 probe data
            model: Detection model
            scan_name: Name of the scan for output files
            reported_depth: List of reported depth values
            scan: Scan object containing flaws
            outlier_diff_range: Threshold for outlier detection
            
        Returns:
            Dictionary with processed depth profile data and maximum depths
        """
        depth_profile = {}
        ranges = self.get_flaw_ranges(scan, probes_data)
        # Process each indicator
        for ind in ranges:
            depth_profile[ind] = {}
            range_info = ranges[ind]
            
            # Extract range parameters
            left1, right1 = range_info['time_range_nb1']
            left2, right2 = range_info['time_range_nb2']
            
            pred_axial_range = range_info['axial_range']
            circ_start, circ_end = range_info['circ_range']
            
            # Process frames for this indicator
            results, depth_profile = self.process_frame_depth(
                probes_data, model,
                frame_range=(pred_axial_range[0], pred_axial_range[1]),
                circ_range=(circ_start, circ_end),
                depth_range=(left1, right1, left2, right2),
                depth_profile=depth_profile,
                ind=ind
            )
        # Post-process depth profiles
        self._postprocess_depth_profiles(depth_profile, scan_name, reported_depth, 
                                        scan, outlier_diff_range)
        
        return depth_profile
    
    def _postprocess_depth_profiles(self, depth_profile, scan_name, reported_depth, 
                       scan, outlier_diff_range):
        """Post-process depth profiles to find maximum depths and update flaw info."""
        for ind in depth_profile:
            depths = []
            for key, value in depth_profile[ind].items():
                if isinstance(value, dict) and value['depth'] is not None:
                    depths.append({
                        'depth': value['depth'],
                        'frame': key.split('-')[0],
                        'probe': key.split('-')[1],
                        'max_amp': value['max_amp'],
                        'y': value['y'],
                        'cls': value['cls'],
                        'viz': value['viz'],
                        'raw': value['raw'],
                        'stats': value['stats']
                    })
            
            # Get reported depth value if available
            reported_depth_value = None
            if reported_depth is not None:
                reported_depth_value = str(next((item[1] for item in reported_depth 
                                            if item[0] == ind), None))
            
            if depths:
                # Filter outliers
                filtered_depths = self._filter_depth_outliers(depths, outlier_diff_range, depth_profile, ind)
                # Find maximum depth
                if filtered_depths:
                    max_depth_info = max(filtered_depths, key=lambda x: x['depth'])
                    max_index = filtered_depths.index(max_depth_info)
                    amplitude_based_selection_occurred = False
                    amplitude_ratio_value = None
                    frames_apart_value = None
                    avg_max_amp_value = None

                    if len(filtered_depths) > 1:
                        # Calculate average max_amp excluding the selected max depth frame
                        other_depths = [d for d in filtered_depths if d != max_depth_info]
                        max_amp = max(d['max_amp'] for d in other_depths) 
                        
                        # If the max depth frame has significantly lower amplitude (e.g., less than 60% of max)
                        if max_depth_info['max_amp'] < 0.6 * max_amp:
                            original_max_frame_num = int(max_depth_info['frame'])
                            original_max_depth_info = max_depth_info  # Store original for potential revert
                            
                            # Record the ignored max depth in stats
                            ignored_frame_key = f"{max_depth_info['frame']}-{max_depth_info['probe']}"
                            if ignored_frame_key in depth_profile[ind] and isinstance(depth_profile[ind][ignored_frame_key], dict):
                                if depth_profile[ind][ignored_frame_key]['stats'] is None:
                                    depth_profile[ind][ignored_frame_key]['stats'] = []
                                elif not isinstance(depth_profile[ind][ignored_frame_key]['stats'], list):
                                    depth_profile[ind][ignored_frame_key]['stats'] = []
                                
                                # Add information about why this max depth was ignored
                                depth_profile[ind][ignored_frame_key]['stats'].append({
                                    'reason': 'low_amplitude',
                                    'max_amp_ignore': max_depth_info['max_amp'],
                                    'max_amp_others': max_amp,
                                    'amplitude_ratio': max_depth_info['max_amp'] / max_amp,
                                    'frames_apart': 0  # This is the original max depth frame
                                })
                            
                            # Select the frame with highest depth that has reasonable amplitude
                            # Sort by depth in descending order
                            sorted_depths = sorted(filtered_depths, key=lambda x: x['depth'], reverse=True)
                            alternative_found = False
                            
                            for d in sorted_depths:
                                if d['max_amp'] >= 0.6 * max_amp:
                                    # Calculate frame distance before accepting the alternative
                                    selected_frame_num = int(d['frame'])
                                    frame_distance = abs(selected_frame_num - original_max_frame_num)
                                    
                                    # Only accept if frames apart is 2 or less
                                    if frame_distance <= 2:
                                        max_depth_info = d
                                        print(f"New max depth selected: frame {max_depth_info['frame']}-{max_depth_info['probe']}")
                                        
                                        # Record the selection of new max depth in stats
                                        amplitude_based_selection_occurred = True
                                        amplitude_ratio_value = max_depth_info['max_amp'] / max_amp
                                        frames_apart_value = frame_distance
                                        max_amp_value = max_amp
                                        selected_frame_key = f"{max_depth_info['frame']}-{max_depth_info['probe']}"
                                        
                                        if selected_frame_key in depth_profile[ind] and isinstance(depth_profile[ind][selected_frame_key], dict):
                                            if depth_profile[ind][selected_frame_key]['stats'] is None:
                                                depth_profile[ind][selected_frame_key]['stats'] = []
                                            elif not isinstance(depth_profile[ind][selected_frame_key]['stats'], list):
                                                depth_profile[ind][selected_frame_key]['stats'] = []
                                            
                                            depth_profile[ind][selected_frame_key]['stats'].append({
                                                'reason': 'amplitude_based_selection',
                                                'max_amp_ignore': max_depth_info['max_amp'],
                                                'avg_max_amp_others': max_amp,
                                                'amplitude_ratio': max_depth_info['max_amp'] / max_amp,
                                                'frames_apart': frame_distance
                                            })
                                        
                                        alternative_found = True
                                        break
                                    
                    if not alternative_found:
                        max_depth_info = original_max_depth_info
                        print(f"Reverting to original max depth frame {max_depth_info['frame']}-{max_depth_info['probe']} - no suitable alternative within 2 frames")
                        
                        # Update the ignored frame stats to indicate reversion
                        if ignored_frame_key in depth_profile[ind] and isinstance(depth_profile[ind][ignored_frame_key], dict):
                            if depth_profile[ind][ignored_frame_key]['stats']:
                                # Update the last stats entry to indicate reversion
                                depth_profile[ind][ignored_frame_key]['stats'][-1]['reverted'] = True
                                depth_profile[ind][ignored_frame_key]['stats'][-1]['revert_reason'] = 'alternatives_too_far'
                    
                    if amplitude_based_selection_occurred:
                        depth_profile[ind]['_amplitude_based_selection'] = True
                        depth_profile[ind]['_amplitude_ratio'] = amplitude_ratio_value
                        depth_profile[ind]['_frames_apart'] = frames_apart_value
                        depth_profile[ind]['_avg_max_amp_others'] = avg_max_amp_value
                    else:
                        depth_profile[ind]['_amplitude_based_selection'] = False
                        depth_profile[ind]['_amplitude_ratio'] = None
                        depth_profile[ind]['_frames_apart'] = None
                        depth_profile[ind]['_avg_max_amp_others'] = None                
                    # Check if max depth class is '7', if so, use previous max V depth
                    if max_depth_info['cls'] == Counter({'7': 1}):
                        # Sort depths by frame number to find previous frames
                        sorted_depths = sorted(filtered_depths, key=lambda x: int(x['frame']))
                        
                        # Find the index of max_depth_info in filtered_depths - FIX: changed from max_depth to max_depth_info
                        try:
                            current_depth_index = filtered_depths.index(max_depth_info)
                        
                            # Check if we have at least 2 previous depths
                            if current_depth_index >= 2:
                                # Get the previous two depths
                                prev_depth1 = filtered_depths[current_depth_index - 1]
                                prev_depth2 = filtered_depths[current_depth_index - 2]
                                
                                # Check if both previous depths are of class 'V'
                                if prev_depth1['cls'] in ['V', 'MV'] and prev_depth2['cls'] in ['V', 'MV']:
                                    # Use the max depth from the previous V depths
                                    v_depths = [d for d in filtered_depths if d['cls'] in ['V', 'MV']]
                                    if v_depths:
                                        original_max_depth_info = max_depth_info  # Store original for stats
                                        v_max_depth = max(v_depths, key=lambda x: x['depth'])
                                        max_depth_info = v_max_depth  # FIX: changed from max_depth to max_depth_info
                                        
                                        # Record the class-based selection change in stats
                                        original_frame_key = f"{original_max_depth_info['frame']}-{original_max_depth_info['probe']}"
                                        if original_frame_key in depth_profile[ind] and isinstance(depth_profile[ind][original_frame_key], dict):
                                            if depth_profile[ind][original_frame_key]['stats'] is None:
                                                depth_profile[ind][original_frame_key]['stats'] = {}
                                            elif not isinstance(depth_profile[ind][original_frame_key]['stats'], dict):
                                                depth_profile[ind][original_frame_key]['stats'] = {}
                                            
                                            depth_profile[ind][original_frame_key]['stats']['ignored_max_depth'] = {
                                                'reason': 'class_7_replaced_by_V',
                                                'original_class': '7',
                                                'replacement_class': max_depth_info['cls'],
                                                'original_depth': original_max_depth_info['depth'],
                                                'replacement_depth': max_depth_info['depth']
                                            }
                                        
                                        # Record selection in the new max depth frame
                                        selected_frame_key = f"{max_depth_info['frame']}-{max_depth_info['probe']}"
                                        if selected_frame_key in depth_profile[ind] and isinstance(depth_profile[ind][selected_frame_key], dict):
                                            if depth_profile[ind][selected_frame_key]['stats'] is None:
                                                depth_profile[ind][selected_frame_key]['stats'] = {}
                                            elif not isinstance(depth_profile[ind][selected_frame_key]['stats'], dict):
                                                depth_profile[ind][selected_frame_key]['stats'] = {}
                                            
                                            depth_profile[ind][selected_frame_key]['stats']['selected_as_max_depth'] = {
                                                'reason': 'V_class_preference_over_7',
                                                'replaced_class': '7',
                                                'selected_class': max_depth_info['cls'],
                                                'depth': max_depth_info['depth']
                                            }
                        except ValueError:
                            # Handle case where max_depth_info might not be in filtered_depths
                            print(f"Warning: Could not find max depth in filtered depths for indicator {ind}")
                            
                    # Get frames before and after max depth frame
                    if max_depth_info['viz'] is not None:
                        # Extract the frame number
                        max_frame = int(max_depth_info['frame'])
                        max_probe = max_depth_info['probe']
                        
                        # Create a combined visualization with 2 frames before and 2 frames after
                        combined_viz = self._create_combined_viz(
                            depth_profile[ind], 
                            max_frame, 
                            max_probe, 
                            num_frames_before=2, 
                            num_frames_after=2,
                            scan_name=scan_name,
                            ind=ind
                        )
                        
                        # Save the combined visualization
                        if reported_depth_value not in (None, 'None'):
                            filename = f"{scan_name}_{ind}_{reported_depth_value}.png"
                        else:
                            filename = f"{scan_name}_{ind}_depth.png"
                        
                        filepath = os.path.join(self.output_dir, filename)
                        cv2.imwrite(filepath, combined_viz)
                  
                    # Update depth profile with maximum values
                    depth_profile[ind].update({
                        'max_depth': max_depth_info['depth'],
                        'max_amp': max_depth_info['max_amp'],
                        'max_depth_frame': max_depth_info['frame'],
                        'max_depth_probe': max_depth_info['probe'],
                        'max_depth_circ': max_depth_info['y'],
                        'max_depth_cls': categorize_depth_category(dict(max_depth_info['cls'])),
                        'max_depth_viz': max_depth_info['viz'],
                        'max_stats': max_depth_info['stats'],
                    })
                else:
                    # Set default values if no valid depths
                    depth_profile[ind].update({
                        'max_depth': None,
                        'max_amp': None,
                        'max_depth_frame': None,
                        'max_depth_probe': None,
                        'max_depth_circ': None,
                        'max_depth_cls': None,
                        'max_depth_viz': None,
                        'max_stats': None,
                    })
            else:
                # Set default values if no valid depths
                depth_profile[ind].update({
                    'max_depth': None,
                    'max_amp': None,
                    'max_depth_frame': None,
                    'max_depth_probe': None,
                    'max_depth_circ': None,
                    'max_depth_cls': None,
                    'max_depth_viz': None,
                    'max_stats': None,
                })
            
        # Generate plots and update flaw depths
        self.plot_depth_profiles(depth_profile, scan_name, reported_depth, scan)
        self.update_flaw_depths_from_profile(scan, depth_profile, reported_depth)
    
    def _create_combined_viz(self, profile_data, max_frame, max_probe, num_frames_before=2, num_frames_after=2, scan_name=None, ind=None, label="frames"):
        """
        Create a combined visualization with frames before and after the maximum depth frame.
        Also saves each individual frame-probe image separately.
        
        Args:
            profile_data: Dictionary with depth profile data
            max_frame: Frame number of the maximum depth
            max_probe: Probe identifier of the maximum depth frame
            num_frames_before: Number of frames to include before the max frame
            num_frames_after: Number of frames to include after the max frame
            scan_name: Name of the scan for file naming
            ind: Index or identifier for file naming
            label: Subfolder name to save images to
            
        Returns:
            Combined visualization image
        """
        # Create a list of frame-probe pairs to include
        frames_to_show = []
        current_frame = max_frame
        current_probe = max_probe
        
        # Add max frame first
        key = f"{current_frame}-{current_probe}"
        if key in profile_data and isinstance(profile_data[key], dict) and profile_data[key]['viz'] is not None:
            frames_to_show.append({
                'frame_num': current_frame,
                'probe': current_probe,
                'viz': profile_data[key]['viz'],
                'is_max': True,
                'order': 0,  # Center position
                'reported_depth': profile_data[key].get('reported_depth', None)  # Add depth value if available
            })
        
        # Add frames before max frame
        for i in range(1, num_frames_before + 1):
            # Determine previous frame and probe based on alternating pattern
            if current_probe == "NB2":
                prev_probe = "NB1"
                prev_frame = current_frame
            else:  # current_probe == "NB1"
                prev_probe = "NB2"
                prev_frame = current_frame - 1
            
            # Update current for next iteration
            current_frame = prev_frame
            current_probe = prev_probe
            
            # Add the frame if it exists
            key = f"{current_frame}-{current_probe}"
            if key in profile_data and isinstance(profile_data[key], dict) and profile_data[key]['viz'] is not None:
                frames_to_show.append({
                    'frame_num': current_frame,
                    'probe': current_probe,
                    'viz': profile_data[key]['viz'],
                    'is_max': False,
                    'order': -i,  # Negative to indicate before max
                    'reported_depth': profile_data[key].get('reported_depth', None)  # Add depth value if available
                })
        
        # Reset to max frame and continue with frames after
        current_frame = max_frame
        current_probe = max_probe
        
        # Add frames after max frame
        for i in range(1, num_frames_after + 1):
            # Determine next frame and probe based on alternating pattern
            if current_probe == "NB1":
                next_probe = "NB2"
                next_frame = current_frame
            else:  # current_probe == "NB2"
                next_probe = "NB1"
                next_frame = current_frame + 1
            
            # Update current for next iteration
            current_frame = next_frame
            current_probe = next_probe
            
            # Add the frame if it exists
            key = f"{current_frame}-{current_probe}"
            if key in profile_data and isinstance(profile_data[key], dict) and profile_data[key]['viz'] is not None:
                frames_to_show.append({
                    'frame_num': current_frame,
                    'probe': current_probe,
                    'viz': profile_data[key]['viz'],
                    'is_max': False,
                    'order': i,  # Positive to indicate after max
                    'reported_depth': profile_data[key].get('reported_depth', None)  # Add depth value if available
                })
        
        # # Save each individual frame separately before any modifications
        # if scan_name is not None and ind is not None:
        #     # Ensure output directory exists
        #     save_dir = os.path.join(self.output_dir, label)
        #     os.makedirs(save_dir, exist_ok=True)
            
        #     # Save the raw images directly from profile_data
        #     # This ensures we're getting the original images before any modifications
        #     for frame_data in frames_to_show:
        #         # Get the key for this frame in the original profile_data
        #         key = f"{frame_data['frame_num']}-{frame_data['probe']}"
                
        #         # Retrieve the original raw visualization image directly from profile_data
        #         if key in profile_data and isinstance(profile_data[key], dict) and profile_data[key]['viz'] is not None:
        #             # Get the original visualization without any modifications
        #             raw_viz = profile_data[key]['raw'].copy()
                    
        #             # Convert grayscale to color if needed for saving
        #             if len(raw_viz.shape) == 2:
        #                 raw_viz = cv2.cvtColor(raw_viz, cv2.COLOR_GRAY2BGR)
                    
        #             # Create filename with frame-probe info
        #             frame_probe_suffix = f"frame{frame_data['frame_num']}-{frame_data['probe']}"
                    
        #             # Add reported depth to filename if available
        #             reported_depth_value = profile_data[key].get('reported_depth')
        #             if reported_depth_value not in (None, 'None'):
        #                 filename = f"{scan_name}_{ind}_{frame_probe_suffix}_{reported_depth_value}.png"
        #             else:
        #                 filename = f"{scan_name}_{ind}_{frame_probe_suffix}_depth.png"
                    
        #             # Save the raw image
        #             filepath = os.path.join(save_dir, filename)
        #             cv2.imwrite(filepath, raw_viz)
        
        # Sort frames by order to get correct sequence
        frames_to_show.sort(key=lambda x: x['order'])
        
        if not frames_to_show:
            # Return the max frame visualization if no other frames are available
            return profile_data[f"{max_frame}-{max_probe}"]['viz']
        
        # Get the dimensions of the visualizations
        sample_viz = frames_to_show[0]['viz']
        height, width = sample_viz.shape[:2]
        
        # Create a combined image
        # Add space between frames and for frame labels
        spacing = 10
        label_width = 100
        combined_height = (height + spacing) * len(frames_to_show) - spacing
        combined_width = width + label_width
        
        # Create a blank image (white background)
        combined_viz = np.ones((combined_height, combined_width, 3), dtype=np.uint8) * 255
        
        # Place each frame in the combined image
        y_offset = 0
        for frame_data in frames_to_show:
            # Convert grayscale to color if needed
            viz = frame_data['viz']
            if len(viz.shape) == 2:
                viz = cv2.cvtColor(viz, cv2.COLOR_GRAY2BGR)
            
            # Add frame to the combined image
            combined_viz[y_offset:y_offset+height, label_width:label_width+width] = viz
            
            # Add frame number and probe as text
            frame_text = f"Frame {frame_data['frame_num']}-{frame_data['probe']}"
            if frame_data['is_max']:
                frame_text += " (MAX)"
                # Add a highlight border around the max frame
                cv2.rectangle(
                    combined_viz,
                    (label_width-2, y_offset-2),
                    (label_width+width+2, y_offset+height+2),
                    (0, 0, 255),  # Red border
                    2
                )
            
            # Add the text
            cv2.putText(
                combined_viz,
                frame_text,
                (5, y_offset + height//2),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (0, 0, 0),  # Black text
                1,
                cv2.LINE_AA
            )
            
            y_offset += height + spacing
        
        return combined_viz
    
    
    def _filter_depth_outliers(self, depths, outlier_diff_range, depth_profile=None, ind=None):
        """Filter outliers by comparing each depth to median of top 5 depths."""
        if not depths:
            return []
        
        # Get top 5 depths (or all if less than 5)
        sorted_by_depth = sorted(depths, key=lambda x: x['depth'], reverse=True)
        top_depths = sorted_by_depth[:5]  # Get top 5 (or all if less than 5)
        top_depth_values = [d['depth'] for d in top_depths]
        
        # Calculate median of top depths
        reference_depth = np.median(top_depth_values)
        
        # Identify outliers
        outliers = []
        filtered_depths = []
        
        for depth in depths:
            diff_from_reference = abs(depth['depth'] - reference_depth)
            
            # Check if current depth exceeds threshold compared to reference
            if diff_from_reference > outlier_diff_range:
                outliers.append({
                    'data': depth,
                    'diff_from_reference': diff_from_reference,
                    'reference_depth': reference_depth,
                    'top_depths_used': len(top_depths)
                })
            else:
                filtered_depths.append(depth)
        
        # If we filtered out all depths, use the original list
        if not filtered_depths:
            filtered_depths = depths
            outliers = []  # Don't mark any as outliers if we keep all
        
        # If depth_profile and ind are provided, update outlier depths
        if depth_profile is not None and ind is not None and outliers:
            for outlier_entry in outliers:
                outlier_depth = outlier_entry['data']
                
                # Construct the key from frame and probe
                key = f"{outlier_depth['frame']}-{outlier_depth['probe']}"
                
                if key in depth_profile[ind] and isinstance(depth_profile[ind][key], dict):
                    # Store original depth value before modification
                    original_depth = depth_profile[ind][key]['depth']
                    
                    # Set outlier depth to the reference depth (median of top 5)
                    depth_profile[ind][key]['depth'] = reference_depth
                    
                    # Add outlier information to stats
                    if depth_profile[ind][key]['stats'] is None:
                        depth_profile[ind][key]['stats'] = {}
                    elif not isinstance(depth_profile[ind][key]['stats'], dict):
                        depth_profile[ind][key]['stats'] = {}
                    
                    # Add outlier modification info to stats
                    depth_profile[ind][key]['stats']['outlier_modified'] = {
                        'original_depth': original_depth,
                        'reference_depth': reference_depth,
                        'diff_from_reference': outlier_entry['diff_from_reference'],
                        'top_depths_count': outlier_entry['top_depths_used'],
                        'top_depth_values': top_depth_values,
                        'outlier_threshold': outlier_diff_range,
                        'modified_to': reference_depth
                    }
                    
                    print(f"Outlier detected: {key}, original: {original_depth:.3f}, "
                        f"reference (median of top {len(top_depths)}): {reference_depth:.3f}, "
                        f"diff: {outlier_entry['diff_from_reference']:.3f}")
        
        return filtered_depths

    def process_frame_depth(self, probes_data, model, frame_range, circ_range, 
                           depth_range, depth_profile=None, ind=0) -> Tuple[Dict, Dict]:
        """
        Process a range of frames to analyze depth using detection and peak analysis.
        
        Args:
            probes_data: Dictionary with probe data ('NB1', 'NB2')
            model: Detection model
            frame_range: (start, end) range for axial frames
            circ_range: (start, end) range for circumferential dimension
            depth_range: (left, right) range for depth dimension
            depth_profile: Dictionary to store results (created if None)
            ind: Index for depth_profile dictionary
            
        Returns:
            Tuple containing:
            - Dictionary of processing results
            - Updated depth_profile with stored results
        """
        results = {'NB1': {}, 'NB2': {}}
        
        # Initialize depth_profile if None
        if depth_profile is None:
            depth_profile = {ind: {}}
        
        # Unpack ranges
        frame_start, frame_end = frame_range
        circ_start, circ_end = circ_range
        left1, right1, left2, right2, = depth_range
        
        # Process each frame
        for frame in range(frame_start, frame_end):
            try:
                # Get frame data for both probes
                NB1_bscan_flaw = probes_data['NB1'].data[frame, circ_start:circ_end, left1:right1]
                NB2_bscan_flaw = probes_data['NB2'].data[frame, circ_start:circ_end, left2:right2]
                
                # Process both probes
                self._process_probe_frame(
                    'NB1', frame, NB1_bscan_flaw, model, results, depth_profile, ind, circ_start
                )
                self._process_probe_frame(
                    'NB2', frame, NB2_bscan_flaw, model, results, depth_profile, ind, circ_start
                )
            except Exception as e:
                # Log error and continue with next frame
                self._handle_frame_error(e, frame, results, depth_profile, ind, 
                                        NB1_bscan_flaw if 'NB1_bscan_flaw' in locals() else None,
                                       NB2_bscan_flaw if 'NB2_bscan_flaw' in locals() else None)
        
        return results, depth_profile
    
    def _process_probe_frame(self, probe, frame, bscan_flaw, model, results, depth_profile, ind, circ_start):
        """Process a single probe frame."""
        window_min = 40
        window_max = 55
        window = False
        # Enhance image
        if window:
            enhanced = MorphologyDetector.enhance_image(bscan_flaw, enhancement_method='clahe')
        else: 
            enhanced = MorphologyDetector.apply_amplitude_window(bscan_flaw, window_min, window_max)
        # Process the frame
        raw = enhanced.copy()
        enhanced_for_processing = enhanced.copy()
        frame_results = MorphologyDetector.process_single_frame(enhanced_for_processing, model, bscan_flaw)
        # Store results
        results[probe][frame] = frame_results
        
        # Store in depth_profile with detection status check
        key = f"{frame}-{probe}"
        if frame_results['detection_status']:
            depth_profile[ind][key] = {
                'depth': frame_results['depth'],
                'cls': frame_results.get('class_counts', {}),
                'max_amp': frame_results.get('max_amp', {}),
                'viz': frame_results.get('visualization', None),
                'raw': raw,
                'y': float(frame_results.get('y', None)) + circ_start,
                'stats': frame_results.get('debug_info', None),
            }
        else:
            depth_profile[ind][key] = {
                'depth': None,
                'cls': None,
                'viz': enhanced,
                'raw': raw,
                'y': None,
                'stats': None
            }
    
    def _handle_frame_error(self, error, frame, results, depth_profile, ind, NB1_bscan_flaw, NB2_bscan_flaw):
        """Handle errors during frame processing."""
        error_info = {'error': f"Frame processing error: {str(error)}"}
        
        # Create default error results
        for probe, data in [('NB1', NB1_bscan_flaw), ('NB2', NB2_bscan_flaw)]:
            error_result = {
                'depth': None,
                'class_counts': {},
                'visualization': data,
                'detection_status': False,
                'debug_info': error_info
            }
            
            # Store error results
            results[probe][frame] = error_result
            depth_profile[ind][f"{frame}-{probe}"] = {
                'depth': None,
                'cls': None,
                'viz': None,
                'y': None,
            }
            
    def categorize_depth_category(self, depth_category):
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
        # Handle None or empty dictionaries
        if not depth_category:
            return 'dnd'
        
        # Handle string input case for "spaghetti"
        if isinstance(depth_category, str) and depth_category.lower() == "spaghetti":
            return "s"
        
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
    
    def plot_depth_profiles(self, depth_profiles, scan_name, reported_depth, scan):
        """
        Create line plots of depth values for each indication and save them.
        
        Args:
            depth_profiles: Dictionary containing depth profiles
            scan_name: Base name for saving plots
            reported_depth: List of reported depth values
            scan: Scan object containing flaws
        """
        
        for ind, depth_profile in depth_profiles.items():
            # Find the reported depth value for this indicator
            reported_depth_value = None
            for flaw in scan.flaws:
                if flaw.ind_num == ind:
                    try:
                        reported_depth_value = str(next(item[1] for item in reported_depth 
                                                    if item[0] == flaw.ind_num))
                    except:
                        pass
            
            # Extract and organize data for plotting (now includes None values)
            sequence, depths, labels, cls_values = self._prepare_plot_data(
                depth_profile, categorize_depth_category
            )
            
            if not any(d is not None for d in depths):  # Skip if no valid data
                continue
            
            # Find max depth and its index for delta calculations (only among valid depths)
            valid_depths = [d for d in depths if d is not None]
            if not valid_depths:
                continue
                
            max_depth = max(valid_depths)
            max_depth_index = next(i for i, d in enumerate(depths) if d == max_depth)
            max_depth_label = labels[max_depth_index]
            
            # Calculate depth spread statistics (only for valid depths)
            depth_std = np.std(valid_depths) if len(valid_depths) > 1 else 0
            depth_mean = np.mean(valid_depths)
            depth_cv = depth_std / depth_mean if depth_mean > 0 else 0  # Coefficient of variation
            
            # Update flaw stats with neighboring classes and depth deltas
            for flaw in scan.flaws:
                if flaw.ind_num == ind and hasattr(flaw, 'stats'):
                    # Check if stats exists and is a dictionary, if not create one
                    if not isinstance(flaw.stats, dict):
                        # If stats is None, tuple, or other type, create new dict
                        if flaw.stats is None:
                            flaw.stats = {}
                        else:
                            flaw.stats = {}
                    
                    # Add current frame class
                    flaw.stats['current_frame_class'] = cls_values[max_depth_index] if max_depth_index < len(cls_values) else None
                    
                    # Add neighboring frame classes (relative to max depth frame)
                    flaw.stats['prev_1_frame_class'] = cls_values[max_depth_index - 1] if max_depth_index >= 1 else None
                    flaw.stats['prev_2_frame_class'] = cls_values[max_depth_index - 2] if max_depth_index >= 2 else None
                    flaw.stats['next_1_frame_class'] = cls_values[max_depth_index + 1] if max_depth_index < len(cls_values) - 1 else None
                    flaw.stats['next_2_frame_class'] = cls_values[max_depth_index + 2] if max_depth_index < len(cls_values) - 2 else None
                    flaw.stats['max_depth_frame_probe'] = max_depth_label
                    flaw.stats['total_frames_analyzed'] = len([d for d in depths if d is not None])
                    
                    # Add depth spread statistics
                    flaw.stats['depth_std'] = round(depth_std, 4)
                    flaw.stats['depth_cv'] = round(depth_cv, 4)  # Coefficient of variation (std/mean)
                    flaw.stats['amplitude_based_selection'] = depth_profiles[ind].get('_amplitude_based_selection', False)
                    flaw.stats['amplitude_ratio'] = depth_profiles[ind].get('_amplitude_ratio')
                    flaw.stats['frames_apart'] = depth_profiles[ind].get('_frames_apart')
                    flaw.stats['avg_max_amp_others'] = depth_profiles[ind].get('_avg_max_amp_others')
                    
                    # Add depth deltas only for 2 frames before and after max depth
                    if max_depth_index >= 1 and depths[max_depth_index - 1] is not None:
                        frame_probe_key = labels[max_depth_index - 1].replace('-', '_')
                        flaw.stats[f'delta_prev_1_{frame_probe_key}'] = depths[max_depth_index - 1] - max_depth
                    
                    if max_depth_index >= 2 and depths[max_depth_index - 2] is not None:
                        frame_probe_key = labels[max_depth_index - 2].replace('-', '_')
                        flaw.stats[f'delta_prev_2_{frame_probe_key}'] = depths[max_depth_index - 2] - max_depth
                    
                    if max_depth_index < len(depths) - 1 and depths[max_depth_index + 1] is not None:
                        frame_probe_key = labels[max_depth_index + 1].replace('-', '_')
                        flaw.stats[f'delta_next_1_{frame_probe_key}'] = depths[max_depth_index + 1] - max_depth
                    
                    if max_depth_index < len(depths) - 2 and depths[max_depth_index + 2] is not None:
                        frame_probe_key = labels[max_depth_index + 2].replace('-', '_')
                        flaw.stats[f'delta_next_2_{frame_probe_key}'] = depths[max_depth_index + 2] - max_depth
                    
                    break
                
            # Create the plot
            fig, ax = plt.subplots(figsize=(12, 6))
            
            # Plot line connecting only valid points
            valid_indices = [i for i, d in enumerate(depths) if d is not None]
            valid_sequence = [sequence[i] for i in valid_indices]
            valid_depths_list = [depths[i] for i in valid_indices]
            
            if valid_indices:
                ax.plot(valid_sequence, valid_depths_list, 'b-', linewidth=2, label='Depth Profile')
                
                # Plot dots only for valid points
                ax.plot(valid_sequence, valid_depths_list, 'bo', markersize=8)
                
                # Highlight the max depth point
                max_seq_pos = sequence[max_depth_index]
                ax.plot(max_seq_pos, max_depth, 'ro', markersize=12, label=f'Max Depth: {max_depth:.3f} mm')
            
            ax.legend(loc='upper right')
            
            # Set labels and grid
            ax.set_title(f"Depth Profile - {ind}", fontsize=12)
            ax.set_xlabel('frame/probe', fontsize=10)
            ax.set_ylabel('Depth (mm)', fontsize=10)
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # Add point labels for all positions (including missing data)
            ax.set_xticks(sequence)
            ax.set_xticklabels(labels, rotation=45)
            
            # Add value labels with depth and class (only for valid depths)
            for i, depth in enumerate(depths):
                if depth is not None:
                    cls_text = " - " + cls_values[i] if cls_values[i] is not None else ""
                    delta_text = " "
                    ax.text(sequence[i], depth + 0.005, f'{depth:.3f}{cls_text}{delta_text}', 
                            ha='center', va='bottom', fontsize=8)
                else:
                    # Optional: Add "No Detection" label for missing points
                    ax.text(sequence[i], -0.01, 'No Detection', 
                            ha='center', va='top', fontsize=6, alpha=0.6)
            
            # Set y-axis limits with padding
            if valid_depths_list:
                y_max = max(valid_depths_list) * 1.2
                y_min = min(0, -0.02)  # Small negative space for "No Detection" labels
                ax.set_ylim(y_min, y_max)
            
            # Save and close
            plt.tight_layout()
            save_path = os.path.join(self.output_dir, f"{scan_name}_{ind}_profile.png")
            plt.savefig(save_path)
            plt.close()

    def _prepare_plot_data(self, depth_profile, categorize_func):
        """Prepare data for depth profile plotting, including missing data points."""
        sequence = []
        depths = []
        labels = []
        cls_values = []
        
        # Get all frame numbers
        frame_numbers = sorted(list(set([int(key.split('-')[0]) for key in depth_profile.keys() 
                                    if isinstance(key, str) and '-' in key])))
        
        # Process frames in order, including missing probes
        x = 0
        for frame in frame_numbers:
            for probe in ['NB1', 'NB2']:
                key = f"{frame+1}-{probe}"
                sequence.append(x)
                labels.append(key)
                
                if key in depth_profile:
                    depth_value = depth_profile[key].get('depth', None)
                    cls_value = depth_profile[key].get('cls', None)
                    cls_value = categorize_func(cls_value) if cls_value else None
                    
                    depths.append(depth_value)
                    cls_values.append(cls_value)
                else:
                    # No detection for this frame/probe combination
                    depths.append(None)
                    cls_values.append(None)
                
                x += 1
        
        return sequence, depths, labels, cls_values
        
    def update_flaw_depths_from_profile(self, scan, depth_profiles, reported_depth):
        """
        Update flaws in the scan with max depths and depth categories.
        
        Args:
            scan: Scan object containing flaws
            depth_profiles: Dictionary with depth profile data
            reported_depth: List of reported depth values
        """
        for flaw in scan.flaws:
            # Get the indicator number from flaw's ind_num
            if flaw.ind_num in depth_profiles:
                try:
                    reported_depth_value = str(next(item[1] for item in reported_depth 
                                              if item[0] == flaw.ind_num))
                except:
                    reported_depth_value = None
                
                profile = depth_profiles[flaw.ind_num]
                
                # Update flaw with depth profile data
                self._update_flaw_with_profile(flaw, profile, reported_depth_value)
    
    def _update_flaw_with_profile(self, flaw, profile, reported_depth_value):
        """Update a flaw object with profile data."""
        # Update depth with max_depth, rounded to 4 decimal places
        if flaw.flaw_type == 'Debris':
            if 'max_depth' in profile and profile['max_depth'] is not None:
                flaw.depth = round(profile['max_depth'], 4)
                flaw.depth_nb = round(profile['max_depth'], 4)
                # Add depth values for NB1 and NB2 probes
                if profile['max_depth_probe'] == 'NB1':
                    flaw.depth_nb1 = round(profile['max_depth'], 4)
                elif profile['max_depth_probe'] == 'NB2':
                    flaw.depth_nb2 = round(profile['max_depth'], 4)
                max_stats = profile['max_stats']
                max_location = max_stats['max_location']
                
                # Preserve existing stats if they exist
                existing_stats = {}
                if hasattr(flaw, 'stats') and isinstance(flaw.stats, dict):
                    existing_stats = flaw.stats.copy()
                
                flaw.stats = {# From max_location (most specific/detailed info)
                    'class': max_location['bbox_class'],
                    'max_value': max_location['max_value'],
                    'area': max_location['area'],
                    'width': max_location['width'],
                    'height': max_location['height'],
                    'length': max_location['length'],
                    'avg_value': max_location['avg_value'],
                    # From main level (unique fields)
                    'avg_position': max_stats['avg_position'],
                    'exclusion_ratio': max_stats['exclusion_ratio'],
                    # Box statistics
                    'total_bboxes': max_location.get('total_bboxes', 0),
                    'total_v_mr_7_boxes': max_location.get('total_v_mr_7_boxes', 0),
                    'ignored_v_mr_7_boxes': max_location.get('ignored_v_mr_7_boxes', 0),
                    'v_mr_7_usage_ratio': max_location.get('v_mr_7_usage_ratio', 1.0),
                    # Neighboring class information
                    'prev_1_class': max_location.get('prev_1_class'),
                    'prev_2_class': max_location.get('prev_2_class'),
                    'next_1_class': max_location.get('next_1_class'),
                    'next_2_class': max_location.get('next_2_class')
                }
                flaw.stats.update({k: v for k, v in existing_stats.items() 
                  if k.startswith(('prev_', 'next_', 'delta_', 'max_depth_frame_probe', 
                                 'total_frames_analyzed', 'current_frame_class', 
                                 'depth_std', 'depth_cv', 'amplitude_based_selection',
                                 'amplitude_ratio', 'frames_apart', 'avg_max_amp_others'))})
            # else:
            #     flaw.depth = None
            #     flaw.depth_nb = None
            #     flaw.depth_nb1 = None
            #     flaw.depth_nb2 = None
                
        if flaw.flaw_type == 'CC':
            if 'max_depth' in profile and profile['max_depth'] is not None:
                flaw.depth_nb = round(profile['max_depth'], 4)
                # Add depth values for NB1 and NB2 probes
                if profile['max_depth_probe'] == 'NB1':
                    flaw.depth_nb1 = round(profile['max_depth'], 4)
                elif profile['max_depth_probe'] == 'NB2':
                    flaw.depth_nb2 = round(profile['max_depth'], 4)
            # else:
            #     flaw.depth_nb = None
            #     flaw.depth_nb1 = None
            #     flaw.depth_nb2 = None
        # Update circumferential position
        if 'max_depth_circ' in profile and profile['max_depth_circ'] is not None:
            flaw.depth_circ = round(profile['max_depth_circ'], 2)
        else:
            flaw.depth_circ = None
            
        # Update frame number
        if 'max_depth_frame' in profile and profile['max_depth_frame'] is not None:
            flaw.depth_frame = int(profile['max_depth_frame'])+1
        else:
            flaw.depth_frame = None
            
        # Update depth category with all classes from max_depth_cls
        if 'max_depth_cls' in profile and profile['max_depth_cls'] is not None:
            flaw.depth_category = profile['max_depth_cls']
        else:
            flaw.depth_category = None
        
        # Update with reported depth if available
        if reported_depth_value is not None:
            flaw.depth_reported = reported_depth_value
    
    def add_depth_fields(self, query_result, scan):
        """
        Add depth-related fields to query results.
        
        Args:
            query_result: Dictionary containing scan query results
            scan: Scan object with depth information
            
        Returns:
            Updated query_result with depth fields
        """
        # Ensure 'extra' field exists
        if 'extra' not in query_result['scan']:
            query_result['scan']['extra'] = []
        
        # Update matched flaws
        self._update_matched_flaws(query_result, scan)
        
        # Add unmatched flaws to 'extra'
        self._add_unmatched_flaws(query_result, scan)
        
        return query_result
    
    def _update_matched_flaws(self, query_result, scan):
        """Update matched flaws with depth information."""
        for flaw in query_result['scan']['flaws']:
            for scanflaw in scan.flaws:
                if flaw['ind_num'] == scanflaw.ind_num:
                    # Update matched flaw with additional attributes
                    if "FBBPF" in scanflaw.flaw_type:
                        flaw['pred_depth'] = scanflaw.depth
                    else:
                        flaw['pred_depth'] = getattr(scanflaw, 'depth', None)
                
                    # Add additional depth fields if available
                    for field in ['depth_circ', 'depth_frame', 'depth_category', 'depth_reported']:
                        if hasattr(scanflaw, field):
                            flaw[field] = getattr(scanflaw, field)
                    
                    break
        query_result['scan']['extra'] = []        
        for rep_flaws in query_result['scan']['reported_flaws']:
            rx1, ry1, rx2, ry2 = rep_flaws['rotary_start'], rep_flaws['axial_start'], rep_flaws['rotary_start'] + rep_flaws['width'], rep_flaws['axial_start'] + rep_flaws['length']  
            
            detected = False
            best_match = None
            best_iou = 0
            
            for scanflaw in scan.flaws:
                px1, py1, px2, py2 = scanflaw.rotary_start, scanflaw.axial_start, scanflaw.rotary_start + scanflaw.width, scanflaw.axial_start + scanflaw.length
                
                # Calculate intersection coordinates
                intersection_x1 = max(rx1, px1)
                intersection_y1 = max(ry1, py1)
                intersection_x2 = min(rx2, px2)
                intersection_y2 = min(ry2, py2)
                
                # Check if there is an intersection
                if intersection_x2 > intersection_x1 and intersection_y2 > intersection_y1:
                    # Calculate areas
                    intersection_area = (intersection_x2 - intersection_x1) * (intersection_y2 - intersection_y1)
                    rep_area = (rx2 - rx1) * (ry2 - ry1)
                    scan_area = (px2 - px1) * (py2 - py1)
                    union_area = rep_area + scan_area - intersection_area
                    
                    # Calculate IOU
                    iou = intersection_area / union_area
                    
                    # Keep track of the best match
                    if iou > best_iou:
                        best_iou = iou
                        best_match = scanflaw
                        detected = True
            
            # After checking all scan flaws against the current reported flaw
            if detected and best_match:
                # Check if flaw types match
                def normalize_flaw_type(flaw_type_str):
                    # Convert to uppercase for case-insensitive comparison
                    flaw_type = flaw_type_str.upper()
                    
                    # Remove common prefixes
                    prefixes = ["BM_", "EM_"]
                    for prefix in prefixes:
                        if flaw_type.startswith(prefix):
                            flaw_type = flaw_type[len(prefix):]
                    
                    # Remove suffixes like " (M)" or similar
                    if "(" in flaw_type:
                        flaw_type = flaw_type.split("(")[0].strip()
                    
                    return flaw_type

                # Now use this for comparison
                classified_true = normalize_flaw_type(rep_flaws['flaw_type']) == normalize_flaw_type(best_match.flaw_type)
                
                # Calculate metric differences for matched pairs
                width_diff = abs(rep_flaws['width'] - best_match.width)
                length_diff = abs(rep_flaws['length'] - best_match.length)
                position_x_diff = abs(rep_flaws['rotary_start'] - best_match.rotary_start)
                position_y_diff = abs(rep_flaws['axial_start'] - best_match.axial_start)
                
                # Calculate depth difference if available
                depth_diff = None
                if 'depth' in rep_flaws and hasattr(best_match, 'depth'):
                    if isinstance(best_match.depth, str) and '(' in best_match.depth:
                        depth_value = float(best_match.depth.split('(')[0].strip())
                    else:
                        if best_match.depth is not None:
                            depth_value = float(best_match.depth)
                        else:
                            depth_value = None  # Or use a default value like 0.0
                    
                    # Check if rep_flaws['depth'] is a string with special notation like "<0.10"
                    if isinstance(rep_flaws['depth'], str) and any(char in rep_flaws['depth'] for char in ['<', '>']):
                        # Handle case where depth is given as '<0.10' or similar
                        try:
                            # Extract the numeric value (remove < or > symbol)
                            threshold_value = 0.1
                            
                            # For '<' comparison
                            if '<' in rep_flaws['depth']:
                                # If depth_value is below threshold, diff is 0
                                if depth_value < threshold_value:
                                    depth_diff = 0
                                # If depth_value exceeds threshold, diff is depth_value - threshold
                                else:
                                    depth_diff = depth_value - threshold_value
                            
                         
                        except (ValueError, TypeError):
                            # Handle any conversion errors for the threshold value
                            depth_diff = None
                    else:
                        # Normal case: convert to float and calculate difference
                        try:
                            depth_diff = float(rep_flaws['depth']) - depth_value
                        except (ValueError, TypeError):
                            # Handle any other conversion errors
                            depth_diff = None
                            
                rep_flaws['pred_depth'] = best_match.depth
                rep_flaws['depth_nb1'] = best_match.depth_nb1
                rep_flaws['depth_nb2'] = best_match.depth_nb2
                rep_flaws['depth_apc'] = best_match.depth_apc
                rep_flaws['depth_cpc'] = best_match.depth_cpc
                rep_flaws['pred_ind_num'] = best_match.ind_num
                rep_flaws['VERSION'] = 'V2.0'
                rep_flaws['is_predicted'] = detected
                rep_flaws['is_classified_correct'] = classified_true
                rep_flaws['iou'] = best_iou
                rep_flaws['depth_category'] = best_match.depth_category if hasattr(best_match, "depth_category") else rep_flaws['flaw_type']
                rep_flaws['metrics'] = {
                        'width_diff': width_diff,
                        'length_diff': length_diff,
                        'position_x_diff': position_x_diff,
                        'position_y_diff': position_y_diff,
                        'depth_diff': depth_diff
                    }
                if (best_match.flaw_type == "Debris" and 
                    hasattr(best_match, 'stats') and 
                    isinstance(best_match.stats, dict)):
                    
                    # Helper function to safely get stats values
                    def safe_get_stats(stats_dict, key, default=None):
                        """Safely get a value from stats dictionary with fallback to default"""
                        return stats_dict.get(key, default)
                    
                    rep_flaws['bbox_stats'] = {
                            'class': safe_get_stats(best_match.stats, 'class'),
                            'max_value': int(safe_get_stats(best_match.stats, 'max_value', 0)),
                            'area': int(safe_get_stats(best_match.stats, 'area', 0)),
                            'width': int(safe_get_stats(best_match.stats, 'width', 0)),
                            'height': int(safe_get_stats(best_match.stats, 'height', 0)),
                            'length': int(safe_get_stats(best_match.stats, 'length', 0)),
                            'avg_value': float(safe_get_stats(best_match.stats, 'avg_value', 0.0)),
                            'avg_position': float(safe_get_stats(best_match.stats, 'avg_position', 0.0)),
                            'exclusion_ratio': float(safe_get_stats(best_match.stats, 'exclusion_ratio', 0.0)),
                            # Current frame class
                            'total_bboxes': safe_get_stats(best_match.stats, 'total_bboxes', 0),
                            'total_v_mr_7_boxes': safe_get_stats(best_match.stats, 'total_v_mr_7_boxes', 0),
                            'ignored_v_mr_7_boxes': safe_get_stats(best_match.stats, 'ignored_v_mr_7_boxes', 0),
                            'v_mr_7_usage_ratio': safe_get_stats(best_match.stats, 'v_mr_7_usage_ratio', 1.0),
                            'current_frame_class': safe_get_stats(best_match.stats, 'current_frame_class'),
                            # Frame-based neighboring classes
                            'prev_1_frame_class': safe_get_stats(best_match.stats, 'prev_1_frame_class'),
                            'prev_2_frame_class': safe_get_stats(best_match.stats, 'prev_2_frame_class'),
                            'next_1_frame_class': safe_get_stats(best_match.stats, 'next_1_frame_class'),
                            'next_2_frame_class': safe_get_stats(best_match.stats, 'next_2_frame_class'),
                            # Max depth frame info
                            'max_depth_frame_probe': safe_get_stats(best_match.stats, 'max_depth_frame_probe'),
                            'total_frames_analyzed': safe_get_stats(best_match.stats, 'total_frames_analyzed'),
                            # Depth spread statistics
                            'depth_std': safe_get_stats(best_match.stats, 'depth_std'),
                            'depth_cv': safe_get_stats(best_match.stats, 'depth_cv'),
                            'amplitude_based_selection': safe_get_stats(best_match.stats, 'amplitude_based_selection', False),
                            'amplitude_ratio': safe_get_stats(best_match.stats, 'amplitude_ratio'),
                            'frames_apart': safe_get_stats(best_match.stats, 'frames_apart'),
                            'avg_max_amp_others': safe_get_stats(best_match.stats, 'avg_max_amp_others'),
                            # Depth deltas (only 2 before and after)
                            'delta_prev_1': safe_get_stats(best_match.stats, [k for k in best_match.stats.keys() if k.startswith('delta_prev_1_')][0]) if any(k.startswith('delta_prev_1_') for k in best_match.stats.keys()) else None,
                            'delta_prev_2': safe_get_stats(best_match.stats, [k for k in best_match.stats.keys() if k.startswith('delta_prev_2_')][0]) if any(k.startswith('delta_prev_2_') for k in best_match.stats.keys()) else None,
                            'delta_next_1': safe_get_stats(best_match.stats, [k for k in best_match.stats.keys() if k.startswith('delta_next_1_')][0]) if any(k.startswith('delta_next_1_') for k in best_match.stats.keys()) else None,
                            'delta_next_2': safe_get_stats(best_match.stats, [k for k in best_match.stats.keys() if k.startswith('delta_next_2_')][0]) if any(k.startswith('delta_next_2_') for k in best_match.stats.keys()) else None,
                        }
                            
            else:
                # Report undetected flaws
                rep_flaws['is_predicted'] = detected
                rep_flaws['pred_ind_num'] = "NA"
                
        # Check for false positives (scan flaws that don't match any reported flaws)
        for scanflaw in scan.flaws:
            px1, py1, px2, py2 = scanflaw.rotary_start, scanflaw.axial_start, scanflaw.rotary_start + scanflaw.width, scanflaw.axial_start + scanflaw.length
            
            matched = False
            for result in query_result['scan']['reported_flaws']:
                if result['pred_ind_num'] == scanflaw.ind_num:
                    matched = True
                    break
            
            if not matched:
                query_result['scan']['extra'].append({
                    'ind_num': scanflaw.ind_num,
                    'pred_flaw_type': scanflaw.flaw_type,
                    'pred_depth': scanflaw.depth,
                    'axial_start': scanflaw.axial_start,
                    'rotary_start': scanflaw.rotary_start,
                    'length': scanflaw.length,
                    'width' : scanflaw.width
                })
    def _add_unmatched_flaws(self, query_result, scan):
        """Add unmatched flaws to the 'extra' list."""
        for scanflaw in scan.flaws:
            # Check if this flaw exists in the primary flaws list
            if not any(flaw['ind_num'] == scanflaw.ind_num 
                      for flaw in query_result['scan']['flaws']):
                # Create a dictionary for the unmatched flaw
                extra_flaw = {
                    'ind_num': scanflaw.ind_num,
                    'depth': getattr(scanflaw, 'depth', getattr(scanflaw, 'depth', None)),
                    'flaw_type': getattr(scanflaw, 'flaw_type', None),
                }
                
                # Add additional depth fields if available
                for field in ['depth_circ', 'depth_frame', 'depth_category', 'depth_reported']:
                    if hasattr(scanflaw, field):
                        extra_flaw[field] = getattr(scanflaw, field)
                
                # Add to the 'extra' list
                query_result['scan']['extra'].append(extra_flaw)


class MorphologyDetector:
    """Class for B-scan processing and analysis functions."""
    @staticmethod
    def apply_amplitude_window(image, min_threshold=40, max_threshold=55, total_range=100):
        """
        Apply amplitude windowing to an ultrasonic image.
        
        Args:
            image: Input ultrasonic image (grayscale)
            min_threshold: Lower threshold value (0-100)
            max_threshold: Upper threshold value (0-100)
            total_range: The total range of the amplitude scale
            
        Returns:
            Processed image with applied amplitude window
        """
        # Make a copy to avoid modifying the original
        #processed_image = image.copy()
        scale_factor=3
         
        height, width = image.shape
        new_size = (width * scale_factor, height * scale_factor)
        
        # Resize the image
        processed_image = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Normalize to 0-100 scale if needed
        if processed_image.max() > 1.0:
            normalized_image = processed_image / 255.0 * 100
        else:
            normalized_image = processed_image * 100
        
        # Convert thresholds to the actual image range
        min_val = min_threshold
        max_val = max_threshold
        
        # Create the windowed image
        windowed_image = np.copy(normalized_image)
        
        # Apply windowing
        # Set values below min_threshold to black
        windowed_image[normalized_image < min_val] = 0
        
        # Set values above max_threshold to white
        windowed_image[normalized_image > max_val] = 100
        
        # Scale the values in between to utilize full contrast
        mask = (normalized_image >= min_val) & (normalized_image <= max_val)
        if np.any(mask):
            windowed_image[mask] = (normalized_image[mask] - min_val) / (max_val - min_val) * 100
        
        # Normalize back to 0-255 range for display
        return np.uint8(windowed_image * 255 / 100)
    
    @staticmethod
    def enhance_image(image, scale_factor=3, brightness_alpha=1.5, brightness_beta=10, 
                    enhancement_method='brightness', clip_limit=3.0, tile_grid_size=(8,8)):
        """
        Enhance an image by resizing and adjusting brightness or applying CLAHE.
        
        Args:
            image: Input image array
            scale_factor: Factor by which to scale the image
            brightness_alpha: Contrast control (used only for brightness method)
            brightness_beta: Brightness control (used only for brightness method)
            enhancement_method: Method to use for enhancement ('brightness' or 'clahe')
            clip_limit: Threshold for contrast limiting in CLAHE (used only for clahe method)
            tile_grid_size: Size of grid for histogram equalization in CLAHE (used only for clahe method)
            
        Returns:
            Enhanced image
        """
        # Get dimensions and calculate new size
        height, width = image.shape
        new_size = (width * scale_factor, height * scale_factor)
        
        # Resize the image
        enlarged = cv2.resize(image, new_size, interpolation=cv2.INTER_LINEAR)
        
        # Apply chosen enhancement method
        if enhancement_method.lower() == 'clahe':
            # Convert to uint8 if not already
            if enlarged.dtype != np.uint8:
                # Scale to 0-255 range
                enlarged_scaled = cv2.normalize(enlarged, None, 0, 255, cv2.NORM_MINMAX)
                enlarged_uint8 = np.uint8(enlarged_scaled)
            else:
                enlarged_uint8 = enlarged
                
            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=tile_grid_size)
            clahe_result = clahe.apply(enlarged_uint8)
        
            # After CLAHE, apply brightness adjustment to make it brighter
            enhanced = cv2.convertScaleAbs(clahe_result, alpha=brightness_alpha, beta=brightness_beta)
        else:  # Default to brightness adjustment
            enhanced = cv2.convertScaleAbs(enlarged, alpha=brightness_alpha, beta=brightness_beta)
        
        return enhanced
    
    @staticmethod
    def process_single_frame(frame, model, b_frame):
        """
        Process a single frame for detection and depth analysis.
        
        Args:
            frame: Enhanced input frame
            model: Detection model
            b_frame: Original frame for visualization
            
        Returns:
            Dictionary with processing results
        """
        # Analyze frame and find peaks
        result_image, peak_data = MorphologyDetector.analyze_image_absolute_extremes(frame, model, b_frame)
        
        if len(result_image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
            
        # Count detected classes
        class_counts = Counter(bbox['class'] for bbox in peak_data)
        
        # If no detections, return early
        if not peak_data:
            return {
                'depth': None,
                'class_counts': {},
                'visualization': b_frame,
                'detection_status': False,
                'debug_info': {'error': 'No detections found'}
            }
        
        # Process peaks and calculate positions
        filtered_array, excluded_y = MorphologyDetector.exclude_peak_regions(b_frame, peak_data)
        max_location = MorphologyDetector.find_max_of_max_x(peak_data)
        # Calculate average position
        avg_position = round(MorphologyDetector.calculate_average_max_position(filtered_array))
        
        # Check if max location was found
        if max_location is None:
            return {
                'depth': None,
                'class_counts': class_counts,
                'visualization': result_image,
                'detection_status': False,
                'debug_info': {'error': 'No maximum location found'}
            }
        
        # Calculate depth
        depth_time = max_location['x'] - avg_position
        depth = MorphologyDetector.convert_units(depth_time)
        
        # Draw visualization
        visualization = MorphologyDetector.draw_vertical_lines_and_plot(
            result_image, avg_position, max_location['x'], depth
        )
        
        # Return results
        return {
            'depth': depth,
            'class_counts': class_counts,
            'visualization': visualization,
            'detection_status': True,
            'y': max_location['y'],
            'max_amp': max_location['max_value'],
            'debug_info': {
                'max_location': max_location,
                'avg_position': avg_position,
                'depth_time': depth_time,
                'excluded_y': excluded_y,
                'exclusion_ratio': filtered_array.shape[0]/b_frame.shape[0],
                'bbox_class': max_location['bbox_class'],
                'bbox_area': max_location['area'],
                'bbox_width': max_location['width'],
                'bbox_height': max_location['height'],
                'bbox_avg_amp': max_location['avg_value'],
            }
        }
    
    @staticmethod
    def analyze_image_absolute_extremes(image, model, b_frame, conf_threshold=0.10):
        """
        Detect objects and find absolute maximum and minimum points.
        
        Args:
            image: Input image array
            model: Detection model
            b_frame: Original frame for reference
            conf_threshold: Confidence threshold for detections
            
        Returns:
            Tuple containing:
            - Annotated output image
            - List of dictionaries with bbox info and extreme points
        """
        # Prepare image for model
        model_input, scale_coords_matrix = MorphologyDetector._prepare_model_input(image)
        
        # Run inference
        detections = model(model_input)
        
        # Process detections
        return MorphologyDetector._process_detections(
            image, detections, b_frame, scale_coords_matrix, conf_threshold
        )
    
    @staticmethod
    def _prepare_model_input(image):
        """Prepare image for model inference."""
        # Convert grayscale to RGB
        if len(image.shape) == 2:
            image = cv2.cvtColor(image.astype(np.uint8), cv2.COLOR_GRAY2RGB)
        
        # Get dimensions
        original_height, original_width = image.shape[:2]
        
        # Set target size to model's expected input shape
        target_size = (640, 640)
        
        # Create canvas and calculate scaling
        canvas = np.ones((target_size[1], target_size[0], 3), dtype=np.uint8) * 114
        scale = min(target_size[0] / original_width, target_size[1] / original_height)
        new_width = int(original_width * scale)
        new_height = int(original_height * scale)
        
        # Resize image and place on canvas
        resized_img = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)
        pad_x = (target_size[0] - new_width) // 2
        pad_y = (target_size[1] - new_height) // 2
        canvas[pad_y:pad_y + new_height, pad_x:pad_x + new_width] = resized_img
        
        # Track transformation parameters
        scale_coords_matrix = {
            'scale': scale,
            'padding': (pad_x, pad_y),
            'target_size': target_size,
            'orig_size': (original_width, original_height),
            'new_size': (new_width, new_height)
        }
        
        # Prepare for inference
        input_tensor = tf.convert_to_tensor(canvas, dtype=tf.float32)
        input_tensor = input_tensor[tf.newaxis, ...]
        input_tensor = input_tensor / 255.0
        
        return input_tensor, scale_coords_matrix
    
    @staticmethod
    def _apply_nms(boxes, scores, classes, iou_threshold=0.45):
        """
        Apply Non-Maximum Suppression to bounding boxes.
        
        Args:
            boxes: numpy array of bounding boxes in format [x1, y1, x2, y2]
            scores: numpy array of confidence scores
            classes: numpy array of class IDs
            iou_threshold: IoU threshold for filtering overlapping boxes
            
        Returns:
            Tuple of filtered (boxes, scores, classes) arrays
        """
        # If no boxes, return empty arrays
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert boxes to format expected by NMS (ensure they are floats)
        boxes = boxes.astype(np.float32)
        
        # Initialize list of picked indices
        picked_indices = []
        
        # Get coordinates of bounding boxes
        x1 = boxes[:, 0]
        y1 = boxes[:, 1]
        x2 = boxes[:, 2]
        y2 = boxes[:, 3]
        
        # Compute the area of each box
        areas = (x2 - x1 + 1) * (y2 - y1 + 1)
        
        # Sort the boxes by score (descending)
        order = np.argsort(scores)[::-1]
        
        # Process boxes by score
        while len(order) > 0:
            # Pick the box with the highest score
            i = order[0]
            picked_indices.append(i)
            
            # Get remaining indices
            if len(order) == 1:
                break
            order = order[1:]
            
            # Get coordinates for remaining boxes
            xx1 = np.maximum(x1[i], x1[order])
            yy1 = np.maximum(y1[i], y1[order])
            xx2 = np.minimum(x2[i], x2[order])
            yy2 = np.minimum(y2[i], y2[order])
            
            # Compute width and height of the intersection
            w = np.maximum(0, xx2 - xx1 + 1)
            h = np.maximum(0, yy2 - yy1 + 1)
            
            # Compute the intersection area
            intersection = w * h
            
            # Compute the IoU
            union = areas[i] + areas[order] - intersection
            iou = intersection / (union + 1e-6)  # Add small epsilon to avoid division by zero
            
            # Keep only the boxes with IoU less than the threshold
            keep = np.where(iou < iou_threshold)[0]
            order = order[keep]
        
        # Return the filtered arrays
        return boxes[picked_indices], scores[picked_indices], classes[picked_indices]

    @staticmethod
    def combine_v_mr_boxes(boxes, scores, classes, iou_threshold=0.1):
        """
        Combine 'V' and 'MR' class bounding boxes from NMS results.
        
        Args:
            boxes: numpy array of bounding boxes in format [x1, y1, x2, y2]
            scores: numpy array of confidence scores
            classes: numpy array of class labels
            iou_threshold: IoU threshold for combining overlapping boxes
            
        Returns:
            Tuple of (boxes, scores, classes) arrays with combined V/MR boxes
        """
        if len(boxes) == 0:
            return np.array([]), np.array([]), np.array([])
        
        # Convert arrays to lists for easier manipulation
        boxes_list = boxes.tolist() if isinstance(boxes, np.ndarray) else list(boxes)
        scores_list = scores.tolist() if isinstance(scores, np.ndarray) else list(scores)
        classes_list = classes.tolist() if isinstance(classes, np.ndarray) else list(classes)
        
        # Identify indices of V and MR classes
        v_mr_indices = [i for i, cls in enumerate(classes_list) if cls == 'V' or cls == 'MR']
        
        # If no V or MR boxes, return original arrays
        if not v_mr_indices:
            return np.array(boxes_list), np.array(scores_list), np.array(classes_list)
        
        # Track which indices have been processed
        processed = set()
        
        # Process each V/MR box
        i = 0
        while i < len(v_mr_indices):
            if i in processed:
                i += 1
                continue
            
            idx_i = v_mr_indices[i]
            box_i = boxes_list[idx_i]
            
            # Find all overlapping V/MR boxes
            merged_indices = [idx_i]
            merged_box = box_i.copy()
            max_score = scores_list[idx_i]
            has_mr = classes_list[idx_i] == 'MR'
            
            j = i + 1
            while j < len(v_mr_indices):
                idx_j = v_mr_indices[j]
                
                if idx_j in processed:
                    j += 1
                    continue
                
                box_j = boxes_list[idx_j]
                
                # Calculate IoU
                x1 = max(box_i[0], box_j[0])
                y1 = max(box_i[1], box_j[1])
                x2 = min(box_i[2], box_j[2])
                y2 = min(box_i[3], box_j[3])
                
                w = max(0, x2 - x1 + 1)
                h = max(0, y2 - y1 + 1)
                
                intersection = w * h
                area_i = (box_i[2] - box_i[0] + 1) * (box_i[3] - box_i[1] + 1)
                area_j = (box_j[2] - box_j[0] + 1) * (box_j[3] - box_j[1] + 1)
                
                iou = intersection / (area_i + area_j - intersection + 1e-6)
                
                if iou >= iou_threshold:
                    # Update merged box to be the union
                    merged_box[0] = min(merged_box[0], box_j[0])
                    merged_box[1] = min(merged_box[1], box_j[1])
                    merged_box[2] = max(merged_box[2], box_j[2])
                    merged_box[3] = max(merged_box[3], box_j[3])
                    
                    # Update max score
                    if scores_list[idx_j] > max_score:
                        max_score = scores_list[idx_j]
                    
                    # Check if this box is MR class
                    if classes_list[idx_j] == 'MR':
                        has_mr = True
                    
                    # Mark this index as processed
                    merged_indices.append(idx_j)
                    processed.add(j)
                
                j += 1
            
            # Mark original box as processed
            processed.add(i)
            
            # If we merged boxes, replace the first box with the merged one and remove others
            if len(merged_indices) > 1:
                # Sort merged indices in descending order to safely remove from list
                merged_indices.sort(reverse=True)
                
                # Replace first box with merged box
                boxes_list[merged_indices[-1]] = merged_box
                scores_list[merged_indices[-1]] = max_score
                classes_list[merged_indices[-1]] = 'MR' if has_mr else 'V'
                
                # Remove other boxes (except the first one we just updated)
                for idx in merged_indices[:-1]:
                    boxes_list.pop(idx)
                    scores_list.pop(idx)
                    classes_list.pop(idx)
                    
                    # Update remaining v_mr_indices to account for removed elements
                    v_mr_indices = [i if i < idx else i - 1 for i in v_mr_indices if i != idx]
            
            i += 1
        
        return np.array(boxes_list), np.array(scores_list), np.array(classes_list)

    @staticmethod
    def _process_detections(image, detections, b_frame, scale_coords_matrix, conf_threshold):
        """Process model detections and extract peak information."""
        # Ensure output_image is in color (BGR) format
        if len(image.shape) == 2:  # If grayscale
            output_image = cv2.cvtColor(image.copy(), cv2.COLOR_GRAY2BGR)
        else:
            output_image = image.copy()
            
        extremes_data = []
        
        # Helper functions for finding extremes
        def find_absolute_max(signal):
            max_idx = np.argmax(signal)
            return max_idx if max_idx is not None else None
        
        def find_absolute_min(signal):
            min_idx = np.argmin(signal)
            return min_idx if min_idx is not None else None
        
        # Class name mapping
        class_names = {
            0: 'V', 1: '7', 2: 'default', 3: 'MR', 4: 'Spaghetti'
        }
        
        # Check for valid detections
        if detections is None or not isinstance(detections, tf.Tensor):
            return output_image, extremes_data
            
        # Extract boxes, scores, and classes
        boxes, scores, classes = MorphologyDetector._extract_detection_data(
            detections, conf_threshold
        )
        
        if len(boxes) > 0:
            # Apply Non-Maximum Suppression to filter overlapping boxes
            boxes, scores, classes = MorphologyDetector._apply_nms(boxes, scores, classes)
            boxes, scores, classes = MorphologyDetector.combine_v_mr_boxes(boxes, scores, classes)
            # Transform coordinates back to original image space
            detection_boxes = MorphologyDetector._transform_coordinates(
                boxes, scale_coords_matrix
            )
            
            # Process each detected box
            for box_id, (box, conf, cls) in enumerate(zip(detection_boxes, scores, classes)):
                if conf < conf_threshold:
                    continue
                    
                x1, y1, x2, y2 = box
                
                # Get class name
                cls_id = int(cls)
                class_name = class_names.get(cls_id, f'class_{cls_id}')
                
                # Skip low confidence spaghetti
                if class_name.lower() == 'spaghetti' and conf < 0.5:
                    continue
                
                # Add buffer for certain classes
                if cls_id == 2:  # Default class
                    x1 = x1 + 10
                
                # Store bbox data
                bbox_data = {
                    'bbox_id': box_id,
                    'bbox': (x1, y1, x2, y2),
                    'class': class_name,
                    'confidence': float(conf),
                    'rows': []
                }
                
                # Handle default bounding boxes
                if bbox_data['class'].lower() in ('default', 'spaghetti'):
                    first_row = b_frame[0:1, :][0]
                    first_max_idx = find_absolute_max(first_row)
                    if first_max_idx is not None:
                        x1 = first_max_idx
                
                # Draw rectangle on the output image
                cv2.rectangle(output_image, (x1*3, y1*3), (x2*3, y2*3), (0, 255, 0), 2)
                
                # Process each row in the bounding box
                MorphologyDetector._process_bbox_rows(bbox_data, b_frame, x1, x2, y1, y2, 
                                                output_image, find_absolute_max, find_absolute_min)
                
                # Add label to the box
                label = f"{class_name} {conf:.2f}"
                cv2.putText(output_image, label, (x1*3, y1*3-10), 
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
                
                extremes_data.append(bbox_data)
        
        return output_image, extremes_data
    
    @staticmethod
    def _extract_detection_data(detections, conf_threshold):
        """Extract boxes, scores, and classes from detection results."""
        boxes = []
        scores = []
        classes = []
        
        if isinstance(detections, tf.Tensor) and len(detections.shape) == 3:
            if detections.shape[0] == 1 and detections.shape[1] == 9:
                detections_np = detections.numpy()[0].transpose()
                
                # Process detections
                for detection in detections_np:
                    cx, cy, w, h = detection[0:4]
                    x1 = cx - w/2
                    y1 = cy - h/2
                    x2 = cx + w/2
                    y2 = cy + h/2
                    
                    # Get class scores
                    class_scores = detection[4:]
                    class_id = np.argmax(class_scores)
                    score = class_scores[class_id]
                    
                    # If score is above threshold, add to results
                    if score >= conf_threshold:
                        boxes.append([x1, y1, x2, y2])
                        scores.append(float(score))
                        classes.append(int(class_id))
        
        return np.array(boxes) if boxes else np.array([]), np.array(scores), np.array(classes)
    
    @staticmethod
    def _transform_coordinates(boxes, scale_coords_matrix):
        """Transform coordinates from model space to original image space."""
        original_boxes = []
        
        # Extract transformation parameters
        pad_x, pad_y = scale_coords_matrix['padding']
        scale = scale_coords_matrix['scale']
        original_width, original_height = scale_coords_matrix['orig_size']
        
        for box in boxes:
            x1, y1, x2, y2 = box
            
            # Remove padding
            x1 = x1 - pad_x
            y1 = y1 - pad_y
            x2 = x2 - pad_x
            y2 = y2 - pad_y
            
            # Reverse scaling
            x1 = x1 / scale
            y1 = y1 / scale
            x2 = x2 / scale
            y2 = y2 / scale
            
            # Clip to image boundaries and scale by factor of 3
            x1 = max(0, min(original_width, x1)) / 3
            y1 = max(0, min(original_height, y1)) / 3
            x2 = max(0, min(original_width, x2)) / 3
            y2 = max(0, min(original_height, y2)) / 3
            
            original_boxes.append([x1, y1, x2, y2])
        
        return np.array(original_boxes).astype(int)
    
    @staticmethod
    def _process_bbox_rows(bbox_data, b_frame, x1, x2, y1, y2, output_image, 
                          find_absolute_max, find_absolute_min):
        """Process rows within a bounding box to find extremes."""
        for y in range(y1, y2):
            if y < b_frame.shape[0] and x1 < b_frame.shape[1] and x2 <= b_frame.shape[1]:
                row = b_frame[y:y+1, x1:x2][0]
                
                max_idx = find_absolute_max(row)
                min_idx = find_absolute_min(row)
                
                # Store row data
                row_data = {
                    'y': y,
                    'max_x': x1 + max_idx if max_idx is not None else None,
                    'min_x': x1 + min_idx if min_idx is not None else None,
                    'max_value': int(row[max_idx]) if max_idx is not None else None,
                    'min_value': int(row[min_idx]) if min_idx is not None else None
                }
                bbox_data['rows'].append(row_data)
                
                # Draw points on image
                if max_idx is not None:
                    cv2.circle(output_image, (x1*3 + max_idx*3, y*3), 2, (0, 0, 255), -1)
    
    @staticmethod
    def find_max_of_max_x(peak_data, y_threshold=5):
        """
        Find the maximum x-coordinate among all max_x values.
        
        Args:
            peak_data: List of dictionaries with bbox and row information
            y_threshold: Maximum difference in y-coordinates to consider boxes at the same level
            
        Returns:
            Dictionary with information about the location of the maximum x-coordinate
        """
        max_x = float('-inf')
        max_location = None
        final_bbox_index = None
        
        # Track statistics about boxes
        total_v_mr_7_boxes = 0
        ignored_v_mr_7_boxes = 0
        ignored_box_ids = []
        
        def calculate_bbox_metrics(bbox):
            """Calculate area, length, width, and average value for a bbox"""
            bbox_coords = bbox.get('bbox', [])
            rows = bbox.get('rows', [])
            
            # Calculate dimensions from bbox coordinates [x1, y1, x2, y2]
            if len(bbox_coords) >= 4:
                x1, y1, x2, y2 = bbox_coords[:4]
                width = abs(x2 - x1)
                height = abs(y2 - y1)
                area = width * height
            else:
                width = height = area = 0
            
            # Calculate average value from all rows with valid max_value
            valid_values = [row.get('max_value') for row in rows if row.get('max_value') is not None]
            avg_value = sum(valid_values) / len(valid_values) if valid_values else None
            
            return {
                'area': area,
                'width': width,
                'height': height,
                'length': max(width, height),  # Length as the longer dimension
                'avg_value': avg_value
            }
        
        # First, identify all V bboxes and their middle y-coordinates
        v_boxes = []
        for bbox in peak_data:
            if bbox['class'] in ['V', 'MR', '7'] and bbox['rows']:
                total_v_mr_7_boxes += 1  # Count total V/MR/7 boxes
                middle_idx = len(bbox['rows']) // 2
                middle_row = bbox['rows'][middle_idx]
                first_row  = bbox['rows'][0]
                last_row = bbox['rows'][-1]
                v_boxes.append({
                    'bbox_id': bbox['bbox_id'],
                    'class': bbox['class'],
                    'bbox': bbox,
                    'middle_y': middle_row['y'],
                    'first_y': first_row['y'],
                    'last_y': last_row['y'],
                    'middle_row': middle_row,
                    'left_x': bbox.get('bbox')[0]
                })

        v_groups = MorphologyDetector._group_boxes_by_y_level(v_boxes, y_threshold)
        
        # Select representative boxes for each group
        v_boxes_to_use = MorphologyDetector._select_representative_boxes(v_groups)
        
        # Calculate ignored boxes
        used_bbox_ids = {vb['bbox_id'] for vb in v_boxes_to_use}
        for vb in v_boxes:
            if vb['bbox_id'] not in used_bbox_ids:
                ignored_v_mr_7_boxes += 1
                ignored_box_ids.append(vb['bbox_id'])
        
        # Process boxes with the filtered V boxes
        for index, bbox in enumerate(peak_data):
            rows = bbox['rows']
            if not rows:
                continue
            
            if bbox['class'] in ['V', 'MR', '7']:
                # Check if this box is in our selected representatives
                is_selected = any(vb['bbox_id'] == bbox['bbox_id'] for vb in v_boxes_to_use)
                if not is_selected:
                    # Skip this box if it's not a selected representative
                    continue
                
            bbox_metrics = calculate_bbox_metrics(bbox)
            # Process based on class type
            if bbox['class'] == 'V':
                new_location = MorphologyDetector._process_v_box(
                    bbox, rows, v_boxes_to_use, max_x, max_location
                )
                if new_location and new_location['x'] > max_x:
                    max_x = new_location['x']
                    max_location = new_location
                    max_location.update(bbox_metrics)
                    final_bbox_index = index
            elif bbox['class'] == 'MR':
                # For MR, take the middle 50% of rows and average their positions
                num_rows = len(rows)
                start_idx = num_rows // 4  # Start at 25% from the beginning
                end_idx = num_rows - start_idx  # End at 25% from the end
                
                # Take the middle 50% rows
                middle_rows = rows[start_idx:end_idx]
                
                # Calculate average location from middle rows
                valid_rows = [row for row in middle_rows if row['max_x'] is not None]
                
                if valid_rows:
                    avg_x = sum(row['max_x'] for row in valid_rows) / len(valid_rows)
                    # Find the row with max_x closest to average for max_value and y
                    closest_row = min(valid_rows, key=lambda row: abs(row['max_x'] - avg_x))
                    
                    if avg_x > max_x:
                        max_x = avg_x
                        max_location = {
                            'bbox_id': bbox['bbox_id'],
                            'bbox_class': bbox['class'],
                            'y': closest_row['y'],  # Take y from the closest row
                            'x': avg_x,
                            'max_value': closest_row.get('max_value'),
                            'row_position': 'middle_50_percent_average'
                        }
                        max_location.update(bbox_metrics)
                        final_bbox_index = index
            elif bbox['class'] == 'Spaghetti':
                # For Spaghetti, use IQR to filter outliers
                new_loc = MorphologyDetector._process_spaghetti_box(bbox, rows, max_x)
                if new_loc and new_loc['x'] > max_x:
                    max_x = new_loc['x']
                    max_location = new_loc
                    max_location.update(bbox_metrics)
                    final_bbox_index = index
            else:
                # For other classes, check all rows
                for row in rows:
                    if row['max_x'] is not None and row['max_x'] > max_x:
                        max_x = row['max_x']
                        max_location = {
                            'bbox_id': bbox['bbox_id'],
                            'bbox_class': bbox['class'],
                            'y': row['y'],
                            'x': row['max_x'],
                            'max_value': row.get('max_value'),
                            'row_position': 'any'
                        }
                        max_location.update(bbox_metrics)
                        final_bbox_index = index
        
        # Add box statistics to max_location
        if max_location:
            max_location.update({
                'total_bboxes': len(peak_data),
                'total_v_mr_7_boxes': total_v_mr_7_boxes,
                'ignored_v_mr_7_boxes': ignored_v_mr_7_boxes,
                'ignored_box_ids': ignored_box_ids,
                'v_mr_7_usage_ratio': (total_v_mr_7_boxes - ignored_v_mr_7_boxes) / total_v_mr_7_boxes if total_v_mr_7_boxes > 0 else 1.0
            })
        
        return max_location
        
    def _group_boxes_by_y_level(boxes, threshold):
        """Group boxes that are at similar y-levels.
        
        This implementation sorts boxes by y-level first to ensure
        consistent grouping regardless of input order.
        Additionally, boxes with first_y or last_y differences greater than 3
        will not be grouped together.
        """
        # Make a copy and sort by y-level
        sorted_boxes = sorted(boxes, key=lambda box: box['middle_y'])
        
        groups = []
        for box in sorted_boxes:
            added = False
            # Try to add to an existing group
            for group in groups:
                # Check if this box can fit in the group by comparing with the group's range
                group_min_y = min(b['middle_y'] for b in group)
                group_max_y = max(b['middle_y'] for b in group)
                
                # If box is within threshold of the entire group range
                if (box['middle_y'] - group_min_y <= threshold and 
                    group_max_y - box['middle_y'] <= threshold):
                    
                    # Check if both first_y and last_y differences are not more than 3 with any box in the group
                    compatible = True
                    for group_box in group:
                        if (abs(box['first_y'] - group_box['first_y']) > 3 or 
                            abs(box['last_y'] - group_box['last_y']) > 3):
                            compatible = False
                            break
                    
                    if compatible:
                        group.append(box)
                        added = True
                        break
            
            # If couldn't add to any existing group, create a new one
            if not added:
                groups.append([box])
        
        return groups
        
    @staticmethod
    def _select_representative_boxes(groups):
        """Select representative boxes from each group."""
        selected_boxes = []
        for group in groups:
            if len(group) > 1:
                # Multiple boxes at similar y-level, pick leftmost
                leftmost = min(group, key=lambda box: box['left_x'])
                selected_boxes.append(leftmost)
            else:
                # Single box at this y-level
                selected_boxes.append(group[0])
        return selected_boxes
    
    @staticmethod
    def _process_v_box(bbox, rows, v_boxes_to_use, max_x, max_location):
        """Process a V-class bounding box."""
        # Skip if not in our filtered list
        matching_boxes = [vb for vb in v_boxes_to_use if vb['bbox_id'] == bbox['bbox_id']]
        if not matching_boxes:
            return max_location
            
        # Use the middle row for V boxes
        middle_idx = len(rows) // 2
        middle_row = rows[middle_idx]
        
        if middle_row['max_x'] is None or middle_row['max_x'] <= max_x:
            return max_location
            
        # Get bbox coordinates and calculate width
        bbox_coords = bbox.get('bbox')
        bbox_x_min, _, bbox_x_max, _ = bbox_coords
        bbox_width = bbox_x_max - bbox_x_min
        
        # Adjust x position if needed
        if middle_row['max_x'] > (bbox_x_min + (bbox_width / 2)):  # Changed to check if in right half
            # If in right half, adjust to left fourth
            adjusted_x = bbox_x_min + (bbox_width / 4)  # Changed to left fourth
            return {
                'bbox_id': bbox['bbox_id'],
                'bbox_class': bbox['class'],
                'y': middle_row['y'],
                'x': adjusted_x,
                'max_value': middle_row.get('max_value'),
                'row_position': 'middle',
                'selection_reason': 'leftmost V box (adjusted)'
            }
        else:
            # Otherwise, use original max_x
            return {
                'bbox_id': bbox['bbox_id'],
                'bbox_class': bbox['class'],
                'y': middle_row['y'],
                'x': middle_row['max_x'],
                'max_value': middle_row.get('max_value'),
                'row_position': 'middle',
                'selection_reason': 'leftmost V box'
            }
    
    @staticmethod
    def _process_spaghetti_box(bbox, rows, current_max_x):
        """Process a Spaghetti-class bounding box."""
        # Get all max_x values
        values = [row['max_x'] for row in rows if row['max_x'] is not None]
        
        if not values:
            return None
            
        # Calculate statistics for outlier detection
        sorted_values = sorted(values)
        Q1 = sorted_values[len(sorted_values)//4]
        Q3 = sorted_values[3*len(sorted_values)//4]
        IQR = Q3 - Q1
        upper_bound = Q3 + 1 * IQR  # Stricter threshold
        
        # Filter outliers
        valid_values = [x for x in values if x <= upper_bound]
        
        if valid_values:
            local_max_x = max(valid_values)
            if local_max_x > current_max_x:
                # Find corresponding row
                for row in rows:
                    if row['max_x'] == local_max_x:
                        return {
                            'bbox_id': bbox['bbox_id'],
                            'bbox_class': bbox['class'],
                            'y': row['y'],
                            'x': row['max_x'],
                            'max_value': row.get('max_value'),
                            'row_position': 'any'
                        }
        
        return None
    
    @staticmethod
    def calculate_average_max_position(array_2d):
        """
        Calculate the average position of maximum values across rows.
        
        Args:
            array_2d: 2D array to analyze
            
        Returns:
            Average position of maximums across rows
        """
        if len(array_2d) == 0:
            return None
            
        max_positions = []
        
        for row in array_2d:
            # Normalize relative to 128 and take absolute values
            normalized_row = np.abs(row - 128)
            
            # Find position of maximum value
            max_pos = np.argmax(normalized_row)
            
            # Check if negative peak
            if row[max_pos] < 128:
                # Look for the next positive peak to the right
                for i in range(max_pos + 1, len(row)):
                    if row[i] > 128 and normalized_row[i] > 0:
                        max_pos = i
                        break
            
            max_positions.append(max_pos)
        
        return np.mean(max_positions)
    
    @staticmethod
    def exclude_peak_regions(image, peak_data):
        """
        Create an array from image excluding rows that contain peaks.
        
        Args:
            image: Input image array
            peak_data: List with bbox and row information
            
        Returns:
            Tuple containing:
            - 2D array with peak-containing rows removed
            - List of Y coordinates that were excluded
        """
        # Convert to grayscale if needed
        if len(image.shape) == 3:
            gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        else:
            gray_image = image.copy()
        
        # Collect Y coordinates with peaks
        excluded_y_coords = set()
        for bbox in peak_data:
            for row in bbox['rows']:
                if row['max_x'] is not None or row['min_x'] is not None:
                    excluded_y_coords.add(row['y'])
        
        # Convert to sorted list
        excluded_y_coords = sorted(list(excluded_y_coords))
        
        # Create mask for rows to keep
        keep_mask = np.ones(gray_image.shape[0], dtype=bool)
        keep_mask[excluded_y_coords] = False
        
        # Return filtered array and excluded coordinates
        return gray_image[keep_mask], excluded_y_coords
    
    @staticmethod
    def convert_units(index):
        """
        Convert from index to time units (depth).
        
        Args:
            index: Index value to convert
            
        Returns:
            Converted time/depth value
        """
        micro_dec_per_index = 0.710
        unit_micro_sec = 0.008
        
        # Perform the calculation
        time = index * (micro_dec_per_index * unit_micro_sec)
        
        # Round to match original precision
        return np.round(time, 4)
    
    @staticmethod
    def draw_vertical_lines_and_plot(result_image, avg_position, max_x, depth):
        """
        Draw vertical lines and labels on the image.
        
        Args:
            result_image: Input image to draw on
            avg_position: Position for first vertical line
            max_x: Position for second vertical line
            depth: Calculated depth value
            
        Returns:
            Image with lines and labels
        """
        if len(result_image.shape) == 2:
            result_image = cv2.cvtColor(result_image, cv2.COLOR_GRAY2BGR)
        # Make a copy of the image
        image_with_lines = result_image.copy()
        
        # Get dimensions
        height, width = image_with_lines.shape[:2]
        
        # Scale positions
        avg_position = avg_position * 3
        max_x = max_x * 3
        
        # Draw line for avg_position (green)
        cv2.line(image_with_lines, 
                (int(avg_position), 0), 
                (int(avg_position), height-1), 
                (0, 255, 0), 2)
        
        # Draw line for max_x (red)
        cv2.line(image_with_lines, 
                (int(max_x), 0), 
                (int(max_x), height-1), 
                (0, 0, 255), 2)
        
        # Add depth label
        font = cv2.FONT_HERSHEY_SIMPLEX
        cv2.putText(image_with_lines, 
                   f'depth: {depth:.4f}', 
                   (int(max_x)-40, 20), 
                   font, 0.5, (0, 0, 255), 2)
        import time
        timestamp = int(time.time() * 1000)  # milliseconds
        filename = f"{timestamp}.png"
        cv2.imwrite(os.path.join('output', filename), image_with_lines)
        return image_with_lines

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
