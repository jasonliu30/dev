# from numba import config
# config.DISABLE_JIT = True
from typing import Dict, List, Optional, Any
import matplotlib.pyplot as plt
from pathlib import Path
import os
import pickle
import numpy as np
import pandas as pd
from b_scan_reader import bscan_structure_definitions as bssd
from differentiation import run_diff_module
from inferencing_script import inference_char
from ML_pipeline import ml_pipeline
from predict_flaw_depth import predict_flaw_depth
from ToF import generate_cscans
from utils import image_slicing
from utils.normalize import preprocessing, NormalizeData
from utils.logger_init import create_dual_loggers
from scan_data_structure import Scan
from flaw_data_structure import Flaw
from depth_sizing.morph_detector import FlawMorphologyAnalyzer

# create loggers
dual_logger, file_logger  = create_dual_loggers()

def auto_analysis(bscan_path, models, config, axial_range=None, save_path='auto_analysis', processor=None,
                  adjusted_plot=None, cscans_output=None, 
                  scan_reference=None, save_cscans=True):
    """
    Perform automatic analysis of B-scan data, including flaw detection and characterization.
    
    Args:
        bscan_path: Path to the B-scan file to be analyzed
        models: Dictionary containing all required models for analysis
        config: Configuration object containing analysis parameters
        axial_range: Optional range for axial data filtering
        run_name: Name identifier for this analysis run
        adjusted_plot: Optional pre-adjusted plot data
        cscans_output: Optional pre-computed C-scans to skip generation step
        scan_reference: Optional reference scan containing reported flaws
        save_cscans: Boolean indicating whether to save generated C-scans
        
    Returns:
        scan: Processed scan object containing all detected and characterized flaws
    """
    
    dual_logger.info('Loading B-scan data...')
    scan = load_and_prepare_bscan(bscan_path, axial_range)
    if scan_reference is not None:
        scan.reported_flaws = scan_reference.reported_flaws
    
    probes_data = scan.get_bscan_data()
    
    if not cscans_output or 'wavegroups' not in cscans_output[0]:
        dual_logger.info('Detect Wavegroups and generate C-scans for each probe...')
        cscans_output = analyze_and_generate_cscans(scan, probes_data, models, config)

        if save_cscans:
            save_dir = Path(r'T:\all_scans_dec20_2022') / scan.bscan_path.stem.replace('Type A', 'Type D')
            try:
                save_dir.mkdir(exist_ok=True)
                with open(save_dir / 'C-scan_rep.sav', 'wb') as f:
                    pickle.dump(cscans_output, f)
            except Exception as e:
                dual_logger.warning(f'Failed to save C-scans: {str(e)}')

    dual_logger.info('Detect Flaws on C-scans...')
    predictions, cscan = detect(cscans_output, models, scan, config)
    
    for index, prediction in enumerate(predictions):
        scan.add_flaw(Flaw.from_prediction(prediction, index, scan, config))
    
    if not scan.flaws:
        dual_logger.info('No indications found')
        return scan
        
    dual_logger.info('Running Differentiation Module...')
    run_diff_module(scan, config)
    scan.delete_flaws_by_confidence(config.characterization.result_summary.conf_threshold.debris)
    scan.generate_ind_num()
    dual_logger.info(f'  - {len(scan.flaws)} indications identified')

    file_logger.info('Initializing flaw depth sizing module...')
    
    depth_profile = processor.calc_vision_depth(
        probes_data, 
        model=models['vision_depth_model'], 
        scan_name=scan.scan_name, 
        scan=scan,
        reported_depth=None  
    )
    #print(depth_profile['Ind 2'])
    results_depth = predict_flaw_depth(scan, cscans_output, probes_data, config)
    
    fig = scan.plot_flaws()
    fig.savefig(os.path.join(save_path, scan.scan_name+'.png'), dpi=300, bbox_inches='tight')
    plt.close(fig) 
    
    scan.merge_flaws_cross_0(config)
    
    pickle_path = os.path.join(save_path,'intermediate', scan.scan_name+'.pkl')
    scan.bscan = None
    scan.bscan_d = None
    with open(pickle_path, 'wb') as file:
        pickle.dump(scan, file)
        
    return scan

def load_and_prepare_bscan(bscan_path: str, axial_range: Optional[List[int]] = None) -> Scan:
    """
    Load a B-scan from the given path and prepare it with the specified axial range.

    Parameters
    ----------
    bscan_path : str
        Path to the B-scan file.
    axial_range : Optional[List[int]], optional
        The axial range to use, by default None.

    Returns
    -------
    Scan
        A prepared Scan object.

    Raises
    ------
    FileNotFoundError
        If the bscan_path does not exist.
    ValueError
        If the axial range is invalid.
    """
    if not os.path.exists(bscan_path):
        raise FileNotFoundError(f"B-scan file not found: {bscan_path}")
    
    scan = Scan.from_bscan_path(bscan_path)
    
    scan.update_axial_range(axial_range)

    return scan

def analyze_and_generate_cscans(scan: Scan,
                                probes_data: Dict[str, bssd.BScanData],
                                models: Dict[str, object],
                                config: object
                            ) -> List[object]:
    """
    Run wavegroup detection on each probe in the bscan file and generate c_scans based on the analysis.

    Parameters
    ----------
    scan : bssd.Scan
        The Scan object containing probe names.
    probes_data : Dict[str, bssd.BScanData]
        A dictionary mapping probe names to their BScanData.
    models : Dict[str, object]
        A dictionary containing the required models for the analysis.
    config : object
        Overall configuration data for the analysis.

    Returns
    -------
    List[object]
        A list of generated c_scans after the analysis for each probe.
    """
    cscans_output = []
    for probe in scan.probe_names:
        auto_config = getattr(config.autoanalysis, probe.lower(), None)
        
        if probe in ('APC', 'CPC'):
            tf_model = models[f'{probe}_wgdm']
            focus_groups = ml_pipeline(probes_data[probe], tf_model, auto_config)
        else:
            focus_groups = []
        
        c_scans = generate_cscans(probes_data[probe], focus_groups, config)
        cscans_output.append(c_scans)
    
    return cscans_output

def detect(cscans_output, models, scan, config):
    """
    Processes predictions from the characterization and binary detection models.

    Parameters
    ----------
    cscans_output : ndarray 
        computed cscans.
    models : dict 
        models for auto analysis.
    bscan : object 
        object representing the loaded bscan data.
    bscan_path : str 
        path to the bscan to be analyzed.
    save_path : str 
        path to save the cscan plot.
    config : Config 
        configuration settings for the analysis.

    Returns
    -------
    predictions : list 
        predictions made by the models.
    cscan : ndarray 
        array representing the cscan.
    """
    min_frame = config.characterization.inference.min_frame
    min_width_pixel = config.characterization.inference.min_width_pixel

    predictions, cscan = run_inference_on_cscan(
        cscans_output, 
        scan,
        models['char_model'], 
        config, 
        model='default',
        binary_det=False
    )

    if cscan.shape[0] > min_frame and cscan.shape[1] > min_width_pixel:
        predictions_b, _= run_inference_on_cscan(
            cscans_output, 
            scan,
            models['binary_det_model'], 
            config, 
            model='binary_det',
            binary_det=True
        )
        predictions += predictions_b
    scan.cscan = cscan
    return predictions, cscan

def prep_cscan(cscans_output, scan, alignment=True):
    """
    Prepare a C-scan by pre-processing its different features.

    This function pre-processes the C-scan's average amplitudes, cors_whole,
    and ToF_flat features. The pre-processed features are then stacked to form
    the final C-scan.

    Parameters
    ----------
    cscans_output : list
        List containing the output from the cscans.
    bscan : BScan
        The BScan object containing the scan data.
    bscan_path : str
        The path to the BScan file.
    alignment : bool, optional (default=True)
        Whether or not to perform alignment during pre-processing.

    Returns
    -------
    cscan : ndarray
        The pre-processed and stacked C-scan.
    """
    axial_pitch = scan.scan_axial_pitch
    nb1_avg_amp = preprocessing(cscans_output[2].get('average_amplitudes'), axial_pitch, alignment)
    apc_cors_whole = preprocessing(cscans_output[0].get('cors_whole'), axial_pitch, alignment)
    nb1_tof_flat = 255 - preprocessing(cscans_output[2].get('ToF_flat'), axial_pitch, alignment)

    cscan = np.stack((nb1_tof_flat, nb1_avg_amp, apc_cors_whole)).transpose((1, 2, 0))

    return cscan

def run_inference_on_cscan(cscans_output, scan, tf_model, config, model='default', binary_det=False):

    """
    run inference using the given model.

    - Loading slicing configuration.
    - Preparing cscans for prediction and processing.
    - Saving the original cscan.
    - Creating image patches from the 3-channel cscan.
    - Rescaling image patches if exhaust is True.
    - Running inference on image patches.
    - If the model is 'binary_det', changing the classification to 7.
    - Calculating the global position of the flaws.
    - Merging bounding boxes.

    Parameters
    ----------
    tf_model : tensorflow.keras.Model
        The Tensorflow model to be used for inference.
    bscan : BScan object
        The bscan object to be processed.
    bscan_path : str
        The path to the bscan to be processed.
    save_path : str
        The path where the processed cscan will be saved.
    config : Config
        The configuration settings for the process.
    model : str, optional (default='default')
        The model to be used for processing.

    Returns
    -------
    merged_labels : list
        The list of merged labels.
    cscan_og : array-like
        The original cscan.
    """
    slicing_config = config.characterization.slicing
    if model == 'default':
        patchsize, img_size, conf, exhaust = slicing_config.patchsize_main, config.characterization.inference.img_size_main, 0.01 if config.characterization.inference.exhaust else config.characterization.inference.conf_main, config.characterization.inference.exhaust
    else:
        patchsize, img_size, conf, exhaust = slicing_config.patchsize_binary, config.characterization.inference.img_size_binary, config.characterization.inference.conf_binary_det, False

    cscan_3c = prep_cscan(cscans_output, scan)
    cscan_og = prep_cscan(cscans_output, scan, alignment=False)
    # cv2.imwrite(os.path.join(save_path, 'cscan.png'), cv2.cvtColor(np.uint8(cscan_og), cv2.COLOR_RGB2BGR))

    img_patches, indices = image_slicing.create_img_patches(cscan_3c, patchsize=patchsize, overlap=slicing_config.overlap)

    if exhaust:
        for i, img in enumerate(img_patches):
            img = NormalizeData(img)
            img = img * 255
            img_patches[i] = img
            # cv2.imwrite(os.path.join(save_path, f"{i}img.png"), img)

    results = inference_char(img_patches, tf_model, config.characterization.inference, img_size, conf,binary_det=binary_det)
    if model == 'binary_det':
        for result in results:
            result.classification = 7
            result.flaw_name = 'Others'

    flaw_patches = image_slicing.calc_flaw_locations(results, indices, cscan_3c.shape)
    merged_labels = image_slicing.merge_bboxes(flaw_patches, slicing_config.iou_threshold, slicing_config.ioa)

    return merged_labels, cscan_og

def update_flaws_from_results(flaws: List[Flaw], results_df: pd.DataFrame, 
                            possible_cats: Dict, flaw_type: str, probe_type: str) -> None:
    for _, row in results_df.iterrows():
        flaw = next(f for f in flaws if f.ind_num == row['Indication'])
        
        if probe_type == 'pc':
            flaw.depth_pc = row.get('flaw_depth')
        else:
            flaw.depth_nb = row.get('flaw_depth')
            
        flaw.depth = row.get('flaw_depth')
        flaw.feature_amp = row.get('flaw_feature_amp')
        flaw.max_amp = row.get('flaw_max_amp')
        flaw.chatter_amplitude = row.get('chatter_amplitude')
        flaw.probe_depth = row.get('probe_depth')
        
        if flaw_type in possible_cats.get(flaw.ind_num, {}):
            possible_cats[flaw.ind_num][flaw_type] = row['flaw_depth']

def handle_special_flaw_types(flaws: List[Flaw], possible_cats: Dict) -> None:
    for flaw in flaws:
        if flaw.flaw_type in ['Note 1', 'Note 1 - overlapped']:
            flaw_cats = possible_cats.get(flaw.ind_num, {})
            if flaw_cats:
                flaw.depth = next(iter(flaw_cats.values()))

def get_cscan(query_result, pickle_root = r"T:\all_scans_dec20_2022",pickle_root_backup= r"T:\cscans_new"):

    def get_scan_from_dir(query_result, pickle_root):

        found= False
        for folder in os.listdir(pickle_root):
            if query_result['scan']['scan_id'] in folder:
                    dir = os.path.join(pickle_root,folder)
                    found= True


        if not found:
            for folder in os.listdir(pickle_root):
                if query_result['scan']['scan_id'].replace('Type A', 'Type A') in folder:
                    dir = os.path.join(pickle_root,folder)
                    found= True
        if found:
            files_found = False
            for file in os.listdir(dir):
                if'C-scan_rep.sav' in file:
                    files_found = True
                
            if files_found:
                with open(os.path.join(dir,'C-scan_rep.sav'),'rb') as f:
                    cscans=pickle.load(f)


                for i in range(len(cscans)):
                    if 'ToF_flat' not in cscans[i].keys():
                        cscans[i]['ToF_flat'] = cscans[i]['tof_flat']
                for i in range(len(cscans)):
                    if 'tof' not in cscans[i].keys():
                        cscans[i]['tof'] = cscans[i]['ToF']
                # if cscans[1].get('wavegroups') is None:
                #     cscans = None
            
        if not found or not files_found:
            cscans = None

        return cscans
    

    cscans = get_scan_from_dir(query_result,pickle_root)

    if type(cscans)==type(None):
        cscans = get_scan_from_dir(query_result,pickle_root_backup)

    return cscans