
from inferencing_script import inference
from postprocessor import postprocessor
from b_scan_reader import bscan_structure_definitions as bssd
from pathlib import Path
from utils.folder_utils import mkdir


def ml_pipeline(probe: bssd.BScanData, tf_model, autoanalysis_config):
    """
    This is the top-level ml function, which calls the inferencing and postprocessing functions for the current probe.

    This function is a wrapper that calls all the necessary functions for the ML_pipeline in order.

    It takes in probe_data, a 3D np arrays that holds image data of the frames in the scan.

    After inferencing, the postprocessor is called.

    Args:
        save_path: Path() to the root output directory
        probe_data: 3D numpy array of the probe data - BScanData.data
        probe_type: enum which determines the current probe.

    Returns:
        frames_ranges - A list of range objects containing the range of G4 locations after Imputations.
    """
    # Run inference
    infer_bbox_data = inference(probe.data, tf_model, autoanalysis_config)

    # Post-process the inference data
    frames_ranges = postprocessor(probe.data, tf_model, infer_bbox_data, autoanalysis_config)[0]
    
    return frames_ranges
