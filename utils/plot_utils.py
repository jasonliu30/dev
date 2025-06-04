import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os
from matplotlib.patches import ConnectionPatch

def plot_depth_profile_single_flaw(row, depth_flaw_arr, probe, SAVE_ROOT):
    
    """
    Generates and save the depth profile

    Args:
        row(pd.dataseries): contains the info about the flaw instance such as, Outage number, Channel, etc.
        depth_flaw_arr(np.array): Depth for every ax and circ inside the flaw extent

    Returns:
        None
    """
    # print('Plotting depth profile...')

    OUTAGE_NUMBER = row['Outage Number']
    CHANNEL = row['Channel']
    FILENAME = row['Filename'].split('.')[0]
    frame_start = row['Ax Start'] + 1
    frame_end = frame_start + depth_flaw_arr.shape[0]

    try:
        total_frames = row['Ax End'] - row['Ax Start']
        if total_frames >= 50:
            figsize=(int(total_frames*0.5), 6)
            rotation=90
        else:
            figsize=(18, 6)
            rotation=0

        sns.set_style("whitegrid")
        fig = plt.figure(figsize=figsize)
        plt.plot(np.arange(frame_start, frame_end), np.nanmax(depth_flaw_arr, axis = 1), marker='o')
        plt.xticks(np.arange(frame_start, frame_end), fontsize=20, rotation=rotation)
        plt.yticks(fontsize=20)
        plt.ylim(0)
        plt.xlim(frame_start, frame_end)
        plt.xlabel('Frame Number', fontsize=24)
        plt.ylabel('Depth (mm)', fontsize=24)
        ind = row['Indication']

        plt.title(f'Depth Sizing Profile\nOutage : {OUTAGE_NUMBER}, Channel : {CHANNEL}, Indication : {ind}', fontsize=24)

        # save the plot
        save_path = os.path.join(SAVE_ROOT, 'Depth Sizing Profile', FILENAME)
        os.makedirs(save_path, exist_ok=True)
        plt.savefig(os.path.join(save_path, f"{OUTAGE_NUMBER}-{CHANNEL}-{ind}-{probe}.png"), bbox_inches = 'tight')
        plt.clf()
        
    except Exception as e:
        ind = row['Indication']
        save_path = os.path.join(SAVE_ROOT, 'Depth Sizing Profile', FILENAME)
        os.makedirs(save_path, exist_ok=True)
        with open(os.path.join(save_path, f"{OUTAGE_NUMBER}-{CHANNEL}-{ind}-{probe}.txt"), 'w') as file:
            file.write('Not able to plot depth profile!!!\nError: ' + str(e))