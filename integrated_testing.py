import pandas as pd
from pathlib import Path
from auto_analysis import auto_analysis
import pandas as pd
from utils.datamap_utils import FlawLocation
from utils import error_histogram_req as ehr
from b_scan_reader import bscan_structure_definitions as bssd
from b_scan_reader.BScan_reader import load_bscan
from run_ptfast import load_models
from sklearn.ensemble import RandomForestClassifier
from config import Config
import time
import numpy as np
import os




output_columns = [
    "scan_name",
    "Ind num",
    "flaw_type",
    "flaw_rotary_start",
    "flaw_width",
    "flaw_axial_start",
    "flaw_length",
    "flaw_depth",
    "disposition",
    "pred_flaw_type",
    "confidence",
    "pred_flaw_axial_start",
    "pred_flaw_length",
    "pred_flaw_rotary_start",
    "pred_flaw_width",
    "frame_start",
    "frame_end",
    "pred_flaw_depth",
    'probe_depth',
    'depth_nb',
    'depth_pc',
    'flaw_feature_amp',
    "pred_flaw_max_amp",
    'chatter_amplitude',
    "possible_category",
    "possible category flaw depth",
    "note_1",
    "note_2",
    "iou",
    "pred_indnum",
]


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
    

def box_iou_calc(boxes1, boxes2):
    # https://github.com/pytorch/vision/blob/master/torchvision/ops/boxes.py
    """
    Return intersection-over-union (Jaccard index) of boxes.
    Both sets of boxes are expected to be in (x1, y1, x2, y2) format.
    Parameters:
    ----------
        boxes1 (Array[N, 4])
        boxes2 (Array[M, 4])
    Returns:
    -------
        iou (Array[N, M]): the NxM matrix containing the pairwise
            IoU values for every element in boxes1 and boxes2
    This implementation is taken from the above link and changed so that it only uses numpy..
    """

    def box_area(box):
        # box = 4xn
        return (box[2] - box[0]) * (box[3] - box[1])

    area1 = box_area(boxes1.T)
    area2 = box_area(boxes2.T)

    lt = np.maximum(boxes1[:, None, :2], boxes2[:, :2])  # [N,M,2]
    rb = np.minimum(boxes1[:, None, 2:], boxes2[:, 2:])  # [N,M,2]

    inter = np.prod(np.clip(rb - lt, a_min=0, a_max=None), 2)
    return inter / (area1[:, None] + area2 - inter)  # iou = inter / (area1 + area2 - inter)


def retrieve_adjusted_flaw_location(excel_path, val_path):
    """
    Retrieve the adjusted location of the flaw from the provided excel file.

    This function reads an excel file that contains adjusted flaw locations.
    It filters rows that match the given filename, and retrieves the required
    information such as flaw type, start and end positions, depth, and disposition.

    Parameters
    ----------
    excel_path : str
        Path to the excel file containing adjusted flaw locations.
    val_path : str
        Path to the validation data.
    Returns
    -------
    adjusted_labels : pandas.DataFrame
        A DataFrame containing the flaw details.
    adjusted_plot : list
        A list containing flaw location details for plotting.
    """
    # Load excel data
    test_adjusted_labels = pd.read_csv(excel_path)

    # Get filename from bscan_path (without extension)
    filename = Path(val_path).stem

    # Filter rows that match the filename
    matching_labels = test_adjusted_labels[test_adjusted_labels['anf_fn'] == filename]

    # Prepare list for adjusted labels and plot information
    adjusted_labels, adjusted_plot = [], []

    # Load bscan data
    bscan_path = Path(val_path.replace('Type D', 'Type A'))
    bscan = load_bscan(bscan_path)
    bscan_axes = bscan.get_channel_axes(bssd.Probe['APC'])

    # Process each matching label
    for _, label_data in matching_labels.iterrows():
        # Retrieve flaw information
        flaw_info = process_flaw_label(label_data, bscan_axes)
        adjusted_labels.append(flaw_info)

        # Prepare flaw location for plotting
        flaw_location = prepare_flaw_location(label_data)
        adjusted_plot.append(flaw_location)

    # Convert adjusted labels to DataFrame
    columns = ['Ind num', 'flaw_type', 'flaw_rotary_start', 'flaw_width', 'flaw_axial_start', 'flaw_length', 'flaw_depth', 'disposition']
    adjusted_labels = pd.DataFrame(adjusted_labels, columns=columns)
    adjusted_labels = adjusted_labels[adjusted_labels['Ind num'] != 'Added']
    return adjusted_labels, adjusted_plot


def process_flaw_label(label_data, bscan_axes):
    """
    Extract flaw information from label data and bscan axes.

    Parameters
    ----------
    label_data : pandas.Series
        A series containing label data.
    bscan_axes : BScanAxes
        The axes data of a bscan.
    Returns
    -------
    flaw_info : tuple
        A tuple containing flaw information.
    """
    ind_num = label_data['Ind num']
    classification = label_data.get('Flaw Type General', label_data.get('Flaw Type'))
    x1, x2 = label_data['rotary start'], label_data['rotary end']
    y1, y2 = label_data['frame start'], label_data['frame end']
    depth = label_data['Depth']
    disposition = label_data.get('Disposition', False)

    axial_start, length = calculate_position(y1, y2, bscan_axes.axial_pos)
    rotary_start, width = calculate_position(x1, x2, bscan_axes.rotary_pos)

    return ind_num, classification, rotary_start, width, axial_start, length, depth, disposition


def calculate_position(start, end, axis):
    """
    Calculate the start position and length based on the start and end indices.

    Parameters
    ----------
    start : int
        The start index.
    end : int
        The end index.
    axis : numpy.array
        The axis data.
    Returns
    -------
    position : float
        The start position.
    length : float
        The length.
    """
    position = axis[start]
    length = axis[end] - position
    if length < 0:  # If the length is negative, the flaw wraps around the end
        length = 360 - position + axis[end]
    return position, length


def prepare_flaw_location(label_data):
    """
    Prepare flaw location for plotting.

    Parameters
    ----------
    label_data : pandas.Series
        A series containing label data.
    Returns
    -------
    flaw_location : list
        A list containing flaw location information for plotting.
    """
    ind_num = label_data['Ind num']
    classification = label_data.get('Flaw Type General', label_data.get('Flaw Type'))
    x1, x2 = label_data['rotary start'], label_data['rotary end']
    y1, y2 = label_data['frame start'], label_data['frame end']

    label = f'{ind_num} {classification}'
    flaws = [[label, y1, y2, x1, x2, 'R']]
    return [FlawLocation(*flaw) for flaw in flaws]


def convert_to_xyxy(results):
    """
    Converts bounding box results into xyxy format (xmin, ymin, xmax, ymax).

    Parameters
    ----------
    results : pandas.DataFrame
        Dataframe containing bounding box data with 'flaw_length', 'flaw_width',
        'flaw_axial_start' and 'flaw_rotary_start' columns.
    Returns
    -------
    xyxy_results : list of tuples
        List of tuples, where each tuple represents a bounding box in the format (xmin, ymin, xmax, ymax).
    """
    return [(row['flaw_rotary_start'],
             row['flaw_axial_start'],
             row['flaw_rotary_start'] + row['flaw_width'],
             row['flaw_axial_start'] + row['flaw_length'])
            for _, row in results.iterrows()]


def retrieve_evaluated_pairs(auto_sizing_scan_summary, adjusted_labels):
    """
    Retrieves evaluated pairs from auto_sizing_scan_summary and adjusted_labels dataframes.

    This function calculates the intersection over union (IoU) for the bounding boxes 
    from auto_sizing_scan_summary and adjusted_labels, and returns a dataframe with pairs
    that have the highest IoU.

    Parameters
    ----------
    auto_sizing_scan_summary : pandas.DataFrame 
        Dataframe with predicted bounding boxes.
    adjusted_labels : pandas.DataFrame
        Dataframe with ground truth bounding boxes.
    Returns
    -------
    evaluated_results : pandas.DataFrame
        Dataframe containing pairs of predicted and ground truth bounding boxes with the highest IoU,
        along with the IoU value for each pair.
    """
    # Convert bounding boxes to xyxy format
    xyxy_auto_sizing = convert_to_xyxy(auto_sizing_scan_summary)
    xyxy_adjusted_labels = convert_to_xyxy(adjusted_labels)

    # Calculate intersection over union
    iou = box_iou_calc(np.array(xyxy_auto_sizing), np.array(xyxy_adjusted_labels))

    # Get index of predicted bounding box with the highest IoU for each ground truth bounding box
    max_iou_values, max_iou_indices = np.max(iou, axis=0), np.argmax(iou, axis=0)

    # Gather evaluated results
    evaluated_results = []
    for i, (_, adjusted_label) in enumerate(adjusted_labels.iterrows()):
        evaluated_result = pd.concat([auto_sizing_scan_summary.iloc[max_iou_indices[i], 0:1], adjusted_label,
                                      auto_sizing_scan_summary.iloc[max_iou_indices[i], 6:]], axis=0)
        evaluated_result['iou'] = max_iou_values[i]
        evaluated_result['pred_indnum'] = auto_sizing_scan_summary.iloc[max_iou_indices[i], 5]
        evaluated_results.append(evaluated_result)

    # Convert list of Series to DataFrame
    for i in range(len(evaluated_results)):
        evaluated_results[i] = evaluated_results[i].reset_index(drop=True)

    evaluated_results = pd.DataFrame(evaluated_results)

    # Adjust columns names
    # output_columns = list(adjusted_labels.columns)
    # output_columns.extend(['pred_flaw_type', 'confidence', 'pred_flaw_axial_start', 'pred_flaw_length',
    #                        'pred_flaw_rotary_start', 'pred_flaw_width', 'pred_flaw_depth', 'pred_max_amp',
    #                        'note 1', 'note 2', 'possibly_category', 'possible category flaw depth', 'flag_high_error',
    #                        'flaw_feature_amp ', 'iou'])
    # output_columns.insert(0, 'scan_name')
    evaluated_results.columns = output_columns

    return evaluated_results.reset_index(drop=True)


def error_calc(evaluated_results, iou_thres=0.1):
    """
    This function will calculate the error of the evaluated results and append to the evaluated df.

    Parameters
    ----------
    evaluated_results : pd.DataFrame
        DataFrame containing the evaluated results.
    iou_thres : float, optional
        Intersection over Union (IoU) threshold for considering a detection as valid.

    Returns
    -------
    evaluated_results : pd.DataFrame
        DataFrame with calculated errors added.
    """
    evaluated_results['detected'] = evaluated_results['iou'] > iou_thres
    evaluated_results['correct_char'] = evaluated_results['flaw_type'] == evaluated_results['pred_flaw_type']
    evaluated_results[['length_error', 'width_error', 'axial_start_error', 'circ_start_error']] = \
        evaluated_results[['pred_flaw_length', 'pred_flaw_width', 'pred_flaw_axial_start', 'pred_flaw_rotary_start']].values \
        - evaluated_results[['flaw_length', 'flaw_width', 'flaw_axial_start', 'flaw_rotary_start']].values
    evaluated_results['flaw_depth'] = pd.to_numeric(evaluated_results['flaw_depth'], errors='coerce')
    evaluated_results.loc[evaluated_results['pred_flaw_depth'].apply(np.isreal), 'depth_error'] = \
        evaluated_results.loc[evaluated_results['pred_flaw_depth'].apply(np.isreal), 'pred_flaw_depth'] - \
        evaluated_results.loc[evaluated_results['pred_flaw_depth'].apply(np.isreal), 'flaw_depth']
    return evaluated_results


def evaluate_scan(auto_sizing_scan_summary, adjusted_labels):
    """
    This function will evaluate the results of the auto_sizing_run_summary based on adjusted_labels

    Parameters
    ----------
    auto_sizing_scan_summary : DataFrame
        DataFrame containing the results of the auto sizing run summary.
    adjusted_labels : DataFrame
        DataFrame containing the adjusted labels.

    Returns
    -------
    evaluated_results: DataFrame
        DataFrame containing the evaluated results.
    """

    # Call to the optimized version of get_evaluated_pairs
    evaluated_pairs = retrieve_evaluated_pairs(
        auto_sizing_scan_summary, adjusted_labels)

    # Get axial pitch
    axial_pitch = auto_sizing_scan_summary['scan_axial_pitch'].iloc[0]
    evaluated_pairs['axial_pitch'] = axial_pitch

    # Evaluate errors using the optimized error_calc function
    evaluated_results = error_calc(evaluated_pairs)
    evaluated_results[['length_error_pixel', 'axial_start_error_pixel']] = \
        (evaluated_results[['length_error', 'axial_start_error']].values / evaluated_results['axial_pitch'][0]).round()

    return evaluated_results


def dimension_diff_histogram(evaluation_summary, save_path=None, per_class=False):
    """
    This function will plot the dimension difference histogram
    """
    # exclude disposition flaws
    flawtypes = ['Debris', 'FBBPF', 'CC', 'BM_FBBPF']
    print(
        f'Number of disposition flaws: {len(evaluation_summary[evaluation_summary["disposition"] == True])}')
    print(
        f'Number of undetected flaws: {len(evaluation_summary[evaluation_summary["detected"] != True])}')
    # evaluation_summary = evaluation_summary[evaluation_summary['disposition'].apply(lambda x: x == True)]
    evaluation_summary = evaluation_summary[evaluation_summary['disposition'].apply(
        lambda x: x != True)]
    evaluation_summary = evaluation_summary[evaluation_summary['detected'].apply(
        lambda x: x == True)]
    evaluation_summary_main = evaluation_summary[evaluation_summary['flaw_type'].apply(
        lambda x: x in flawtypes)]
    save_path = os.path.join(save_path, 'stats')
    os.makedirs(save_path, exist_ok=True)
    # save_path = os.path.join(save_path , 'dispositional')

    # get the error histograms
    evaluation_summary_main = evaluation_summary_main.sort_values(
        by=['axial_start_error_pixel'])
    ehr.error_hist_single(data=evaluation_summary_main['length_error_pixel'],
                          binwidth=1,
                          main_tile=f'Length Pixel Difference Distribution: {len(evaluation_summary_main)} cases',
                          ax_title='Stochastic Error',
                          x_title='length difference (slice/pixel)',
                          stoch_req=4,
                          save_name='Length Difference in pixels',
                          save_path=save_path)
    ehr.error_hist(data=evaluation_summary_main['length_error'],
                   binwidth=0.5,
                   main_tile=f'Length Difference Distribution: {len(evaluation_summary_main)} cases',
                   ax1_title='Non-Stochastic Error',
                   ax2_title='Stochastic Error',
                   x_title='length difference (mm)',
                   stoch_req=0.4000,
                   non_stoch_req=0.50,
                   save_name='Length Difference in mms',
                   save_path=save_path)
    
    ehr.error_hist(data=evaluation_summary_main['width_error'],
                   binwidth=0.1,
                   main_tile=f'Width Difference Distribution: {len(evaluation_summary_main)} cases',
                   ax1_title='Non-Stochastic Error',
                   ax2_title='Stochastic Error',
                   x_title='width difference (degree)',
                   stoch_req=0.4000,
                   non_stoch_req=0.20,
                   save_name='Width Difference in degrees',
                   save_path=save_path)

    ehr.error_hist_single(data=evaluation_summary_main['axial_start_error_pixel'],
                          binwidth=1,
                          main_tile=f'Axial Start Difference Distribution: {len(evaluation_summary_main)} cases',
                          ax_title='Stochatic Error Requirements',
                          x_title='axial start difference (slice/pixel)',
                          stoch_req=2,
                          save_name='axial Start Difference in pixels',
                          save_path=save_path)

    ehr.error_hist_single(data=evaluation_summary_main['circ_start_error'],
                          binwidth=0.1,
                          main_tile=f'Rotary Start Difference Distribution: {len(evaluation_summary_main)} cases',
                          ax_title='Stochatic Error Requirements',
                          x_title='width difference (degree)',
                          stoch_req=0.2,
                          save_name='Rotary Start Difference in degrees',
                          save_path=save_path)

    if per_class:
        for cls in flawtypes:
            evaluation_summary['flag_high_error'] = evaluation_summary['flag_high_error'].fillna(False)
            evaluation_summary_main = evaluation_summary[
                            (evaluation_summary['flaw_type'] == cls)]

            ehr.error_hist_single(data=evaluation_summary_main['length_error_pixel'],
                                  binwidth=1,
                                  main_tile=f'Length Pixel Difference Distribution - {cls}: {len(evaluation_summary_main)} cases',
                                  ax_title='Stochatic Error Requirements',
                                  x_title='length difference (slice/pixel)',
                                  stoch_req=4,
                                  save_name='Length Difference in pixels - ' + cls,
                                  save_path=save_path)

            ehr.error_hist(data=evaluation_summary_main['width_error'],
                           binwidth=0.1,
                           main_tile=f'Width Difference Distribution - {cls}: {len(evaluation_summary_main)} cases',
                           ax1_title='Non-Stochatic Error Requirements',
                           ax2_title='Stochatic Error Requirements',
                           x_title='width difference (degree)',
                           stoch_req=0.4,
                           non_stoch_req=0.5,
                           save_name='Width Difference in degrees - ' + cls,
                           save_path=save_path)
            evaluation_summary_main = evaluation_summary[
                (evaluation_summary['correct_char']) & 
                (evaluation_summary['detected']) & 
                (evaluation_summary['flag_high_error']!=True) &
                (evaluation_summary['flaw_type'] == cls)
            ]
            ehr.error_hist(data=evaluation_summary_main['depth_error'],
                            binwidth=0.02,
                            main_tile=f'Depth Difference Distribution - {cls}: {len(evaluation_summary_main)} cases',
                            ax1_title='Non-Stochatic Error Requirements',
                            ax2_title='Stochatic Error Requirements',
                            x_title='depth difference (mm)',
                            stoch_req=0.034,
                            non_stoch_req=0.01,
                            save_name='Depth Difference in mm - ' + cls,
                            save_path=save_path)


def format_auto_sizing_run_summary(auto_sizing_run_summary):

    auto_sizing_run_summary = pd.concat(auto_sizing_run_summary)
    # if column flaw_depth is < 0.1, then make it '< 0.1'
    auto_sizing_run_summary['flaw_depth'] = auto_sizing_run_summary['flaw_depth'].apply(
        lambda x: '< 0.1' if isinstance(x, float) and x < 0.1 else x)
    auto_sizing_run_summary['flaw_length'] = auto_sizing_run_summary['flaw_length'].apply(
        lambda x: '< 1.5' if x < 1.5 else x)
    auto_sizing_run_summary['flaw_width'] = auto_sizing_run_summary['flaw_width'].apply(
        lambda x: '< 1.5' if x < 1.5 else x)
    
    return auto_sizing_run_summary
    

def eval(validation_scans, run_name='evaluation'):
    """
    This function will evaluate the performance of the PTFAST software
    Args:
        validation_scans: list of validation scans
        run_name: name of the run
    Returns:
        evaluation_summary: summary of the evaluation
    """
    print('Starting evaluation...')
    auto_sizing_run_summary = []
    evaluation_summary = []

    # read all configurations
    config = Config.load_all_configs()

    # load models
    print('Loading models...')
    models = load_models(config)

    start_time = time.time()
    total_scans = len(validation_scans)
    current_scan_count = 0
    error = []
    for val_path in validation_scans:
        current_scan_count += 1
        print(f"\n--- Processing {current_scan_count}/{total_scans} Scans ---")

        try:
            bscan_path = Path(val_path.replace('Type D', 'Type A'))
            adjusted_labels, adjusted_plot = retrieve_adjusted_flaw_location(
                excel_path, val_path)  # Get adjusted labels, and in format for plotting
            auto_sizing_scan_summary, save_path = auto_analysis(
                bscan_path, models, config, run_name=run_name, adjusted_plot=adjusted_plot)

            if auto_sizing_scan_summary is None:
                continue
            
            auto_sizing_run_summary.append(auto_sizing_scan_summary)

            if len(adjusted_labels) != 0:
                print('looking for adjusted labels and prediction pairs...')
                # run evaluation on the results
                evaluated_results = evaluate_scan(
                    auto_sizing_scan_summary, adjusted_labels)
                evaluation_summary.append(evaluated_results)
        except Exception as e:
            error.append([val_path, e])
            print(f'Error in {val_path}: {e}')
            continue
    print("--- %s minutes %s seconds ---" %
          (int((time.time() - start_time)/60), int((time.time() - start_time) % 60)))

    auto_sizing_run_summary = format_auto_sizing_run_summary(auto_sizing_run_summary)
    evaluation_summary = pd.concat(evaluation_summary)
    save_path = Path(save_path).parent.parent
    # save the intermediate evaluation summary
    evaluation_summary.to_excel(os.path.join(
        save_path, "evaluation_summary.xlsx"), index=False)
    auto_sizing_run_summary.to_excel(os.path.join(
        save_path, "auto_sizing_run_summary.xlsx"), index=False)

    dimension_diff_histogram(
        evaluation_summary, save_path=save_path, per_class=True)
    return auto_sizing_run_summary, evaluation_summary


if __name__ == "__main__":
    # %load_ext autoreload
    # %autoreload 2
    # Mode = 'Train'
    Mode = 'Test'

    if Mode == 'Test':
        file_path_name = "file_paths.txt"
        excel_path = r"T:\Autosizing\Working Directory\3. Flaw Characterization\QA\Adjusted Labels - dispositioned\test_adjusted_labels-final-aug22.csv"
        run_name = 'validation_run_round2'
    
    elif Mode == 'Train':
        file_path_name = "train_filenames_jason.txt"
        excel_path = r"T:\Autosizing\Working Directory\3. Flaw Characterization\QA\Adjusted Labels - dispositioned\train_adjusted_labels - final - May18.csv"
        run_name = 'train_run_round2'

    with open(file_path_name, "r") as f:
        validation_scans = [line.rstrip() for line in f]
  

    # auto_sizing_run_summary, evaluation_summary = eval(
    #     validation_scans, run_name=run_name)