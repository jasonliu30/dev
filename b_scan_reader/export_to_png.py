from BScan_reader import load_bscan
from BScan import BScan
import cv2
import os
from tqdm import trange
from pathlib import Path


def export_to_png(bscan_path: str, probe: str, out_dir: str):
        """_summary_

        Args:
            probe (_type_): _description_
            out_dir (_type_): _description_
        """
        
        # Create the output directory
        probe_dir = Path(out_dir) / probe
        os.makedirs(probe_dir, exist_ok=True)
        
        # Load the BScan File and extract the probe data
        bscan_file = load_bscan(bscan_path)
        probe_data = bscan_file.read_channel_data(probe)
        data = probe_data.data
        
        # Write images to file
        for i in trange(data.shape[0], desc=f"Exporting Probe {probe} to png.", ascii=True):
            img0 = cv2.merge((data[i,:,:],data[i,:,:],data[i,:,:]))
            fn = probe_dir / f'Frame_{i:03d}.png'
            cv2.imwrite(str(fn), img0)
    

if __name__ == "__main__":  # pragma: no cover
    bscan_file = r'\\azu-fsai01\\RAW DATA\\Scan Data - Pickering\\P1551\\L18\\BSCAN Type A  L-18 Pickering B Unit-5 east 20-Mar-2015 181513 [A3720-4020][R1500-2010].anf'
    export_dir = os.path.join(os.getcwd(),
                              '../C_scan_results/L18-P1551-[A3720-4020][R1500-2010] (contact)/group_det/png_exports')

    export_to_png(bscan_file, 'CPC', export_dir)
    export_to_png(bscan_file, 'APC', export_dir)
