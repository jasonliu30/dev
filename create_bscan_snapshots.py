import numpy as np
import cv2
import matplotlib.pyplot as plt
import os

def reverse_convert_units(time):
   micro_dec_per_index = 0.710
   unit_micro_sec = 0.008
   # Extract numeric value if in format like "0.55(1)"
   if isinstance(time, str) and '(' in time:
       time = float(time.split('(')[0])
   return max(np.round(time / (micro_dec_per_index * unit_micro_sec), 4), 0)

def find_lowest_point(array_2d):
    """
    Find the coordinates of the lowest value in a 2D NumPy array.
    
    Args:
        array_2d (numpy.ndarray): A 2D array of numbers
        
    Returns:
        tuple: (x, y) coordinates of the lowest value
        
    Raises:
        ValueError: If the input array is empty or not a valid 2D array
    """
    if not isinstance(array_2d, np.ndarray):
        array_2d = np.array(array_2d)
    
    if array_2d.size == 0 or array_2d.ndim != 2:
        raise ValueError("Input must be a non-empty 2D array")
    
    # Find the index of the minimum value
    min_idx = np.argmin(array_2d)
    
    # Convert flat index to 2D coordinates
    y, x = np.unravel_index(min_idx, array_2d.shape)
    
    return x, y

def create_snapshot(scan, query_result, probes_data, stats, depth_profile):
    #stats = stats.reset_index()
    snapshots = {}
    folder_name_reported = 'reported_flaws_snap'
    folder_name_FP = 'FP_flaws_snap'
    fig_reported = 'fig_reported_flaws_snap'
    fig_FP = 'fig_FP_flaws_snap'
    
    os.makedirs(folder_name_reported, exist_ok=True)  # reported_flaws_snap
    os.makedirs(folder_name_FP, exist_ok=True)        # FP_flaws_snap
    os.makedirs(fig_reported, exist_ok=True)          # fig_reported_flaws_snap
    os.makedirs(fig_FP, exist_ok=True)                # fig_FP_flaws_snap
    # Process matched flaws upfront
    matched_flaws = [
        {**pred_flaw, 'matched_flaw': next(
            flaw for flaw in query_result['scan']['reported_flaws'] 
            if flaw['ind_num'] == pred_flaw['reported_id']
        )}
        for pred_flaw in query_result['scan']['flaws']
    ]
    
    def get_frame_and_circ(stats_row, depth_profile, total_frames=None):
        """Helper function to determine frame and circumferential position"""
        if not np.isnan(stats_row['flaw_ax']):
            return int(stats_row['flaw_ax'])-1, int(stats_row['flaw_circ'])
        elif not np.isnan(stats_row['flaw_fwg_rmv_ax']):
            return int(stats_row['flaw_fwg_rmv_ax'])-1, int(stats_row['flaw_fwg_rmv_circ'])
        elif not np.isnan(stats_row['flaw_ax_invert']):
            return int(stats_row['flaw_ax_invert'])-1, int(stats_row['flaw_circ_invert'])
        else:
            # Safer access to depth_profile with fallback to middle frame
            try:
                if depth_profile and isinstance(depth_profile[1], dict) and depth_profile[1].get('max_depth_frame') is not None:
                    frame = int(depth_profile[1]['max_depth_frame'])
                else:
                    # If total_frames is provided, use middle frame, otherwise default to frame 0
                    frame = total_frames // 2 if total_frames else 0
            except (IndexError, TypeError, KeyError):
                # Handle any errors by using middle frame or frame 0
                frame = total_frames // 2 if total_frames else 0
                
            # Safe calculation of circumferential position
            try:
                circ = int(flaw.x_start + ((flaw.x_end - flaw.x_start) / 2))
            except (AttributeError, TypeError):
                # If flaw isn't accessible, default to 0
                circ = 0
                
            return frame, circ
    def get_depth_values(stats_row, flaw):
        """Helper function to extract depth values"""
        pred_depth = stats_row['pred_depth']
        if np.isnan(pred_depth):
            pred_depth = flaw.depth
        elif np.isnan(flaw.depth):
            for q in query_result['scan']['flaws']:
                if flaw.ind_num == q['ind_num']:
                    pred_depth = q['depth']
                    
                    if any(text in str(pred_depth) for text in ["(2)", "See note_2"]):
                        if "<" in str(pred_depth):
                            pred_depth = float(pred_depth.split()[1]) - 0.01
                        else:
                            pred_depth = float(pred_depth.split()[0])
                    else:
                        if "<" in str(pred_depth):
                            pred_depth = float(pred_depth.split()[1]) - 0.01
                        else:
                            pred_depth = float(pred_depth) 
        else:
            pred_depth = None
            
        return {
            'pred_depth': pred_depth,
            'fwg_rmv': stats_row['pred_depth_fwg_rmv'],
            'invert': stats_row['pred_depth_invert'],
            'fwg_rmv_invert': stats_row['pred_depth_fwg_rmv_invert'],
            'invert_reflections': stats_row['pred_depth_invert_reflections'],
            'fwg_rwv_inc': stats_row['pred_depth_fwg_rwv_inc'],
            'reflections_fwg': stats_row['reflections_fwg'],
            'n_multi_reflections': stats_row['n_multi_reflections'],
            'surface_circ': stats_row['surface_circ']
        }
    
    def create_figure(ascan, img, enhanced, horizontal_line, depth_info, flaw, reported_depth=None, circ_start_loc=None):
        """Helper function to create and configure the figure with aligned plots"""
        # Resize enhanced image to desired aspect ratio (height:width = 1:2)
        target_height = enhanced.shape[1] // 2
        from scipy.ndimage import zoom
        zoom_factor = target_height / enhanced.shape[0]
        enhanced_resized = zoom(enhanced, (zoom_factor, 1), order=0)
        
        # Update horizontal line position
        horizontal_line = int(horizontal_line * zoom_factor)
        
        fig = plt.figure(figsize=(10, 10))
        gs = fig.add_gridspec(3, 1, height_ratios=[1, 1, 4])
        ax1, ax2, ax3 = fig.add_subplot(gs[0]), fig.add_subplot(gs[1]), fig.add_subplot(gs[2])
        
        
        if np.isnan(depth_info['surface_circ']):
            return None
            # Calculate surface_estimated and max_fwg_updated
        surface_estimated = int(depth_info['surface_circ']-circ_start_loc)
        max_fwg_updated = np.argmax(img[surface_estimated])
        
        # Get the maximum width among all data
        signal_length = len(img[surface_estimated])
        ascan_length = len(ascan)
        max_width = max(signal_length, ascan_length)
        
        # Plot A-scan and enhanced signal with consistent x-range
        ax1.plot(np.arange(signal_length), img[surface_estimated], 'g-', label='signal')
        ax1.axvline(x=max_fwg_updated, color='blue', linestyle='--', label='Max FWG')

        ax2.plot(np.arange(ascan_length), ascan, 'b-', label='A-scan')
        
        if reported_depth is not None:
            reported_depth_time = reverse_convert_units(reported_depth)
            red_line = max_fwg_updated + round(reported_depth_time)
            ax2.axvline(x=red_line, color='red', linestyle='--', label=f'Reported: {reported_depth}')
            ax3.axvline(x=red_line, color='red', linestyle='--')
        
        pred_depth_time = reverse_convert_units(depth_info['pred_depth'])
        green_line = max_fwg_updated + round(pred_depth_time)
        
        # Add lines
        ax2.axvline(x=green_line, color='green', linestyle='--', label=f'Predicted: {depth_info["pred_depth"]}')
        ax3.axvline(x=green_line, color='green', linestyle='--')
        ax3.axvline(x=max_fwg_updated, color='blue', linestyle='--')
        ax3.axhline(y=surface_estimated, color='blue', linestyle='--')
        
        # Configure bottom plot with matching width
        ax3.imshow(enhanced_resized, cmap='gray', extent=[0, max_width, enhanced_resized.shape[0], 0], aspect='auto')
        ax3.axhline(y=horizontal_line, color='white', linestyle='--')
        
        # Set consistent x-axis limits for all plots
        ax1.set_xlim(0, max_width)
        ax2.set_xlim(0, max_width)
        ax3.set_xlim(0, max_width)
        
        # Set y-axis limits for cleaner appearance
        ax1.set_ylim(min(img[surface_estimated])*0.9, max(img[surface_estimated])*1.1)
        ax2.set_ylim(min(ascan)*0.9, max(ascan)*1.1)
        
        # Set x-axis visibility
        ax1.xaxis.set_visible(False)
        ax2.xaxis.set_visible(False)
        
        plt.subplots_adjust(bottom=0.15, hspace=0.1)
        
        # Add legend
        handles1, labels1 = ax1.get_legend_handles_labels()
        handles2, labels2 = ax2.get_legend_handles_labels()
        ax3.legend(handles1 + handles2, labels1 + labels2, 
                bbox_to_anchor=(0.5, -0.15), 
                loc='upper center', 
                ncol=4, 
                bbox_transform=ax3.transAxes)
        
        ax2.set_title('A-scan at Flaw Position')
        ax3.set_xlabel('Time')
        ax3.set_ylabel('Circumferential Position')
        
        return fig
        
    def create_title(scan, frame, circ_og, flaw, depth_info):
        """Helper function to create the title"""
        return (
            f"Scan: {scan.scan_name}\n"
            f"Frame/Circ: {frame+1} / {circ_og}\n"
            f"Flaw {flaw.ind_num} - Depth Analysis\n\n"
            f"Inverted: {depth_info['invert']:.3f}\n"
            f"Inverted Reflections: {depth_info['invert_reflections']:.3f}\n"
            f"FWG Removed: {depth_info['fwg_rmv']:.3f}\n"
            f"FWG RWV Inc: {depth_info['fwg_rwv_inc']:.3f}\n"
            f"FWG Removed & Inverted: {depth_info['fwg_rmv_invert']:.3f}\n\n"
            f"Reflections: {int(depth_info['n_multi_reflections'])}"
        )
    
    # Process each flaw
    debris_flaws = [flaw for flaw in scan.flaws if flaw.flaw_type == 'Debris']
    for i, flaw in enumerate(debris_flaws):
        #try:
            extra_pred = True
            # Get frame and circumferential position
            frame, circ_og = get_frame_and_circ(stats.iloc[i], list(depth_profile.items())[i])
            
            # Calculate flaw region
            circ_start = flaw.x_start
            circ_end = flaw.x_end
            circ_length = circ_end - circ_start
            
            # Extract B-scan data and process image
            NB1_bscan_flaw = probes_data['NB1'].data[frame, circ_start-circ_length:circ_end+circ_length, :]
            NB1_bscan_flaw_1 = probes_data['NB1'].data[frame+1, circ_start-circ_length:circ_end+circ_length, :]
            NB1_bscan_flaw_n1 = probes_data['NB1'].data[frame-1, circ_start-circ_length:circ_end+circ_length, :]
            NB2_bscan_flaw = probes_data['NB2'].data[frame, circ_start-circ_length:circ_end+circ_length, :]
            NB2_bscan_flaw_1 = probes_data['NB2'].data[frame+1, circ_start-circ_length:circ_end+circ_length, :]
            NB2_bscan_flaw_n1 = probes_data['NB2'].data[frame-1, circ_start-circ_length:circ_end+circ_length, :]
            
            frames_list = [NB2_bscan_flaw_n1, NB2_bscan_flaw_1,NB2_bscan_flaw, NB1_bscan_flaw_n1, NB1_bscan_flaw_1, NB1_bscan_flaw]
            horizontal_line = circ_og - (circ_start - circ_length)
            max_fwg = np.argmax(NB1_bscan_flaw[0])
            left, right = max_fwg - 50, max_fwg + 250
            
            img = NB1_bscan_flaw[:,left:right]
            img2 = NB2_bscan_flaw[:,left:right]
            ascan = img[horizontal_line,:]
            enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=0)
            
            # Get depth information
            depth_info = get_depth_values(stats.iloc[i], flaw)
            
            # Create figure
            reported_depth = None
            if flaw.ind_num in [mf['ind_num'] for mf in matched_flaws]:
                reported_depth = next(mf['matched_flaw']['depth'] for mf in matched_flaws if mf['ind_num'] == flaw.ind_num)
                extra_pred = False
            
            name = f'{scan.scan_name} - {flaw.ind_num} - frame{frame}'
            frames_name = ['-1-NB2','+1-NB2','-NB2','-1','+1','']
            # save raw
            for i,frames_enhance in enumerate(frames_list):
                img = frames_enhance[:,left:right]
                #ascan = img[horizontal_line,:]
                enhanced = cv2.convertScaleAbs(img, alpha=1.5, beta=0)

                height, width = enhanced.shape
                new_size = (width * 3, height * 3)
                enlarged = cv2.resize(enhanced, new_size, interpolation=cv2.INTER_LINEAR)
                save_name = name + str(frames_name[i]) + '.png'
                # Save based on condition
                if extra_pred:
                    cv2.imwrite(os.path.join(folder_name_FP, save_name), enlarged)
                else:
                    cv2.imwrite(os.path.join(folder_name_reported, save_name), enlarged)
                    
          
            # Set title
            try:
                fig = create_figure(ascan, img, enhanced, horizontal_line, depth_info, flaw, reported_depth, circ_start-circ_length)
                title = create_title(scan, frame, circ_og, flaw, depth_info)
                fig.suptitle(title)
                plt.tight_layout()
            except:
                continue
            
            
            # Store snapshot
            snapshots[flaw.ind_num] = {
                'figure': fig,
                'predicted_depth': depth_info['pred_depth'],
                'reported_depth': reported_depth,
                'frame': frame,
                'circ_position': circ_og,
                'enhanced_image': enhanced,
                'circ_range': [circ_start-circ_length,circ_end+circ_length],
                'axial_range': [flaw.y_start, flaw.y_end],
                'time_range': [left,right]
            }
            
            plt.close()

              # save figures
            if extra_pred:
                fig.savefig(os.path.join(fig_FP, name), dpi=300, bbox_inches='tight')
            else:
                fig.savefig(os.path.join(fig_reported, name), dpi=300, bbox_inches='tight')
            
        # except:
        #     print(f'pass {flaw}')
        #     continue
    return snapshots

# if __name__ == '__main__':
#     snapshots = create_snapshot(scan, query_result, probes_data, stats)

    # for ind_num, snapshot in snapshots.items():
    #     display(snapshot['figure'])
        