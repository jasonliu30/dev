from pathlib import Path
import argparse
from tensorflow import saved_model
import time
from config import Config
from auto_analysis import auto_analysis
import pandas as pd
import os
from sklearn.ensemble import RandomForestClassifier
import sys
from utils.logger_init import create_dual_loggers
import shutil
from ultralytics import YOLO

"""
PTFAST V1.1 Usage Instructions:

1. Running on Multiple Scans:
   Command: python run_ptfast.py <scan_list_file> [--run_name <name_of_run>]
   - --scan_list <scan_list_file>: Text file with file paths of scans to process, one per line.
   - --run_name <name_of_run>: Optional. Specify a name for the run.

2. Running on a Single Scan:
   Command: python run_ptfast.py <scan_path> [--run_name <name_of_run>] [--axial_range <start>-<end>]
   - <scan_path>: Specify the file path of the single scan.
   - --run_name <name_of_run>: Optional. Specify a name for the run.
   - --axial_range <start>-<end>: Optional. Define an axial range (in mm) for large scans. 
     This is only available for single scans. Syntax: --axial_range <start>-<end>.
     Example: python run_ptfast.py "C:\scans\scan1.txt" --axial_range 7000-7150 --run_name validation_run
     This analyzes only the specified range of the scan.

Example:
   python run_ptfast.py --scan_list file_paths.txt --run_name validation_run
   - Runs PTFAST on scans listed in 'file_paths.txt' with run named 'validation_run'.
"""
# create loggers
dual_logger, file_logger  = create_dual_loggers()


class RandomForestNullHandlingClassifier(RandomForestClassifier):
    def fit(self, X, y):
        X_no_null = self._handle_null_values(X)
        super().fit(X_no_null, y)
    
    def predict(self, X):
        X_no_null = self._handle_null_values(X)
        return super().predict(X_no_null)
    
    def predict_proba(self, X):
        X_no_null = self._handle_null_values(X)
        return super().predict_proba(X_no_null)
    
    def _handle_null_values(self, X):
        X_no_null = X.fillna(-9999)
        return X_no_null
    
    
def load_models(config):
    """
    This function will load the models from the config file
    
    Parameters
    ----------
    config : dict
    
    Returns
    -------
    models : dict containing loaded models
    """
    CPC_wgdm_path = config.autoanalysis.cpc.inference.model_dir
    APC_wgdm_path = config.autoanalysis.apc.inference.model_dir
    characterization_model_path = config.characterization.inference.model_main_dir
    binary_det_model_path = config.characterization.inference.model_binary_dir
    vision_model_path = r'C:\Users\LIUJA\Documents\GitHub\PTFAST-v1.2.0\models\vision_model'
    
    CPC_wgdm = saved_model.load(CPC_wgdm_path)
    APC_wgdm = saved_model.load(APC_wgdm_path)
    char_model = saved_model.load(characterization_model_path)
    binary_det_model = saved_model.load(binary_det_model_path)
    vision_depth_model = saved_model.load(vision_model_path)

    models = {'CPC_wgdm': CPC_wgdm, 'APC_wgdm': APC_wgdm, 'char_model': char_model, 'binary_det_model': binary_det_model, 'vision_depth_model': vision_depth_model}

    return models

def check_scan_extension(scans):
    """
    Identifies scans with invalid extensions from a list of scans.

    Parameters:
    ----------
    scans (list): A list containing the file names of scans.

    Returns:
    -------
    invalid_indices (list): Indices of scans with invalid extensions.
    """
    
    invalid_indices = [index for index, scan in enumerate(scans) 
                       if not str(scan).lower().endswith(('.anf', '.daq'))]

    return invalid_indices


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the PTFAST software on one or more BScans")
    parser.add_argument('scan', nargs='?',  type=str, default=None)
    parser.add_argument('--scan_list', type=str, default=None)
    parser.add_argument('--axial_range', type=str, default=None)
    parser.add_argument('--run_name', type=str, default='ptfast_output')
    parser.add_argument('-o', type=str, default=None)
    return parser.parse_args()


def main():

    """
    This function will run the PTFAST software on one or more BScans
    """


    args = parse_arguments()

    save_dir = os.getcwd() + '/auto-analysis-results'
    scans = []

    if args.scan is not None: # if single scan is provided
        scans = [args.scan]
    elif args.scan_list is not None: # if list of scans is provided
        with open(args.scan_list, "r") as f:
            scans = [Path(line.rstrip()) for line in f]

    if args.o is not None:
        save_dir = args.o

    # Create save_dir if necessary
    if not os.path.exists(save_dir):
        os.mkdir(save_dir)
    
    dual_logger, file_logger  = create_dual_loggers(save_dir + '/log_file.log')

    file_logger.info("\n"+"-"*50)
    dual_logger.info('Initializing PTFAST...')

    # check scan extensions are valid
    try:
        invalid_scan_indices = check_scan_extension(scans)
        if len(invalid_scan_indices) > 0:
            invalid_scans = [str(scans[i]) for i in invalid_scan_indices]
            raise ValueError(f"Invalid file extension. Expected '.anf' or 'daq, but got '{', '.join(invalid_scans)}'.")
    except ValueError as e:
        dual_logger.error(f"Error: {e}", exc_info=True)
        sys.exit(1)
        
    run_name=args.run_name

    if not scans:
        print('No scans selected')
        exit(0)


    save_dir = Path(save_dir)



    file_logger.info(f'Run Name: {run_name}')
    if args.axial_range is not None:
        axial_range = [int(args.axial_range.split('-')[0]), int(args.axial_range.split('-')[1])]
    else:
        axial_range = None
    auto_sizing_run_summary = []

    # read all configurations
    file_logger.info('Reading all configurations...')
    config = Config.load_all_configs()

    # load models
    dual_logger.info('Loading Models...')
    models = load_models(config)
    
    start_time = time.time()
    total_scans = len(scans)
    current_scan_count = 0    
    for scan in scans:
        current_scan_count += 1
        dual_logger.info(f"\n--- Processing {current_scan_count}/{total_scans} ---")
        try:
            bscan_path = Path(str(scan).replace('Type D', 'Type A'))
            auto_sizing_scan_summary, save_path = auto_analysis(bscan_path, models, config, axial_range=axial_range,run_name=run_name,out_dir=save_dir)
            if auto_sizing_scan_summary is None:
                continue
            auto_sizing_run_summary.append(auto_sizing_scan_summary)
        except Exception as e:
            dual_logger.error(f'Error in {scan}: {e}', exc_info=True)
            continue
    dual_logger.info(f'Finished processing {len(scans)} scans in run {run_name}')
    dual_logger.info("--- %s minutes %s seconds ---" % (int((time.time() - start_time)/60), int((time.time() - start_time)%60)))
    
    # combining the results for all scans
    try:
        auto_sizing_run_summary = pd.concat(auto_sizing_run_summary)
        save_path = Path(save_path).parent.parent
        summary_path = os.path.join(save_path,"auto_sizing_run_summary.csv")

        # check to see if the file already exists - append if it does
        if Path(summary_path).exists():
            dual_logger.info('Scan has been analyzed before, using existing auto sizing summary')
            tmp = pd.read_csv(summary_path)
            auto_sizing_run_summary = pd.concat([auto_sizing_run_summary, tmp])

        dual_logger.info('Saving PTFAST Summary...')
        auto_sizing_run_summary.to_csv(summary_path, index=False)

        dual_logger.info(f'Summary saved to: {summary_path}')

    except ValueError:
        save_path = Path(save_path).parent.parent
        dual_logger.error('No indications found in this run. Ending run without saving summary.', exc_info=False)

if __name__ == "__main__":
    main()
