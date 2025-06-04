import copy
import os
import time
from enum import IntEnum
from itertools import combinations
from pathlib import Path
import cv2
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import inferencing_script
from b_scan_reader import bscan_structure_definitions as bssd
from inferencing_script import InferenceOutput
from tqdm.auto import tqdm, trange
from typing import List, Dict
from utils.folder_utils import mkdir


class WaveGroupSearchResults(IntEnum):
    Frame_name = 0
    Missing_g2 = 1
    Missing_g3 = 2
    Missing_g4 = 3
    Missing_all_labels = 4
    Labels_not_in_order = 5
    Repeating_labels = 6
    Missing_g2_impute = 7
    Missing_g3_impute = 8
    Missing_g4_impute = 9
    Missing_all_labels_impute = 10
    Labels_not_in_order_impute = 11
    g2_start = 12
    g2_end = 13
    g2_length = 14
    g2_confidence = 15
    g2_lag = 16
    g3_start = 17
    g3_end = 18
    g3_length = 19
    g3_confidence = 20
    g3_lag = 21
    g4_start = 22
    g4_end = 23
    g4_length = 24
    g4_confidence = 25
    g4_lag = 26
    avg_lag = 27
    Anomaly_G3 = 28
    Anomaly_G4 = 29
    g3_g4_overlap = 30


class SlicedResults(IntEnum):
    Frame_name = 0
    s_g4_0 = 1
    e_g4_0 = 2
    l_g4_0 = 3
    conf_g4_0 = 4
    s_g4_1 = 5
    e_g4_1 = 6
    l_g4_1 = 7
    conf_g4_1 = 8
    s_g4_2 = 9
    e_g4_2 = 10
    l_g4_2 = 11
    conf_g4_2 = 12
    s_g4_3 = 13
    e_g4_3 = 14
    l_g4_3 = 15
    conf_g4_3 = 16
    s_g4_4 = 17
    e_g4_4 = 18
    l_g4_4 = 19
    conf_g4_4 = 20
    frame_sliced = 21


class ErrorLogFrameInfo:
    """
    Object containing values tracked in the error log.
        - group_start = The position where the bounding box begins in yolo format.
        - group_end = The position where the bounding box ends in yolo format.
        - length = The length of the bounding box in yolo format.
        - confidence = The ML confidence score for the bounding box, from 0-1.
        - bbox_count = How many of times this classification of bounding box was found in the image.
    """
    group_start: np.core.numeric = np.NaN
    group_end: np.core.numeric = np.NaN
    length: np.core.numeric = np.NaN
    confidence: np.core.numeric = np.NaN
    bbox_count: int = 0

    def __int__(self, group_start: np.NaN, group_end: np.NaN, length: np.NaN, confidence: np.NaN, bbox_count: int):
        self.group_start = group_start
        self.group_end = group_end
        self.length = length
        self.confidence = confidence
        self.bbox_count = bbox_count



def iou_1d(bb1_x1: float, bb1_x2: float, bb2_x1: float, bb2_x2: float, verbose: bool):
    """
    This function takes 2 bounding boxes and calculates their iou.

    If iou_area is a positive number, the 2 bounding boxes overlap.

    Input:
        - bb1_x1: left most position of the first bounding box. Position will be relative to image and
          will be between 0 and 1 
        - bb1_x2: right most position of the first bounding box. Position will be relative to image and
          will be between 0 and 1
        - bb2_x1: left most position of the second bounding box. Position will be relative to image and
          will be between 0 and 1
        - bb2_x2: right most position of the second bounding box. Position will be relative to image and
          will be between 0 and 1
        - verbose: bool, if true outputs debugging info in console
        
    Output:
        - iou: returns the intersection over union (iou) i.e., amount of overlap of bounding boxes. 
          If there is no overlap, returns 0
    """

    x_left = max(bb1_x1, bb2_x1)  # x-coordinate of the potential bbox overlap (left side)
    x_right = min(bb1_x2, bb2_x2)  # x-coordinate of the potential bbox overlap (right side)
    iou_area = x_right - x_left  # calculate the length of bbox intersection
    if verbose:  # pragma: no cover
        print(f"X left: {x_left}")
        print(f"X right: {x_right}")
        print(f"Iou area: {iou_area}")

    if iou_area > 0:
        bb1_x = bb1_x2 - bb1_x1  # x length of bbox 1
        bb2_x = bb2_x2 - bb2_x1  # x length of bbox 2
        iou = iou_area / (bb1_x + bb2_x - iou_area)  # calculate iou value
        if verbose:  # pragma: no cover
            print(f"IOU: {iou}")
        return iou
    else:
        return 0.0


def merge_bbox(frame: list[InferenceOutput], iou_thresh: float, same_labels: bool, verbose: bool):
    """
    This function merges bounding boxes within a frame if the iou is greater than the user defined threshold.
    When looking at a frame, we take all possible pairs of bounding boxes and calculate the iou or overlap
    (iou_1d function) of the 2 bounding boxes from the pair of bounding box combinations.

    If there is an overlap, we remove the bounding box with the lowest confidence and update the other bbox with the
    highest confidence to represent the merged bboxes.

    Input:
        - frame: list of lists that contains information about all the bounding boxes in a frame
          [bbox_i][bbox info]
        - iou_thresh: value between 0 and 1. The iou_thresh represents the amount (in percentage) of bounding
          box overlap before merging. For example, an iou_thresh of 0.25 means bboxes that overlap more than 25%
          will be merged together
        - same_labels: flag (True, False) that decides how merging is done. If True, we will merge overlapping
          bboxs only if they belong to the same label/class. If false we merge bbox that overlap each other
          regardless of their label/class
        - verbose: bool, if true outputs debugging info in console
        
    Output:
        - frame: list of lists that contains information about all the merged bounding boxes in a frame 
          [bbox_i][bbox info].
    """

    pairs = list(combinations(range(len(frame)), 2))  # get all the bbox index combinations per frame

    for i, j in pairs:  # go through every combination of bboxes per frame

        # extract required information (x-coord and width) for each bbox
        bb1_x = frame[i].x
        bb1_w = frame[i].width
        bb1_conf = frame[i].confidence
        bb1_label = frame[i].classification

        bb2_x = frame[j].x
        bb2_w = frame[j].width
        bb2_conf = frame[j].confidence
        bb2_label = frame[j].classification

        # calculate the start (x1) and end (x2) of each bbox we are comparing
        bb1_x1 = bb1_x - (bb1_w / 2)
        bb1_x2 = bb1_x + (bb1_w / 2)

        bb2_x1 = bb2_x - (bb2_w / 2)
        bb2_x2 = bb2_x + (bb2_w / 2)

        if verbose:  # pragma: no cover
            print(f"\nFrame name: {frame[i].frame_number}")
            print(f"Bbox combination: \n{frame[i]} \n{frame[j]}")
            print(f"Bb1_x1: {bb1_x1}")
            print(f"Bb1_x2: {bb1_x2}")
            print(f"Bb1_conf: {bb1_conf}")
            print(f"Bb2_x1: {bb2_x1}")
            print(f"Bb2_x2: {bb2_x2}")
            print(f"Bb2_conf: {bb2_conf}")

        # calculate iou of the 2 bboxes
        iou = iou_1d(bb1_x1, bb1_x2, bb2_x1, bb2_x2, verbose)

        if iou > iou_thresh and (not same_labels or bb1_label == bb2_label):
            # Check the iou and the labels
            # Merge the bboxes if the iou exceeds the iou threshold, and if the labels match while same_labels is true
            # not(A) or (B) is the logical `implies`. Meaning we only care if bb1_label == bb2_label if same_labels is True.
            # If same_labels is False, it doesn't matter what bb1_label and bb2_label are, because `not same_labels` will be True.

            new_width = max(bb1_x2, bb2_x2) - min(bb1_x1, bb2_x1)  # calculate new width
            new_x = (max(bb1_x2, bb2_x2) + min(bb1_x1, bb2_x1)) / 2  # calculate new x center

            if verbose:  # pragma: no cover
                print(f"New x: {new_x}")
                print(f"New width: {new_width}")

            if bb1_conf >= bb2_conf:  # if bb1 has higher conf value,
                frame[i].x = new_x  # we rewrite bb1 x-center with new x-center value and new width value
                frame[i].width = new_width  # and width with new width value
                frame.pop(j)
                if verbose:  # pragma: no cover
                    print("Replacing bb1")
                    print("Removing bb2")

            else:  # if bb2 has higher conf value,
                frame[j].x = new_x  # we rewrite bb2 x-center with new x-center value and new width value
                frame[j].width = new_width
                frame.pop(i)
                if verbose:  # pragma: no cover
                    print("Removing bb1")
                    print("Replacing bb2")
            # recursively call this function until no bboxes overlap
            return merge_bbox(frame, iou_thresh, same_labels, verbose)

    return frame


def create_wavegroup_search_report(list_of_frames_with_bounding_boxes: list[list[InferenceOutput]]):
    """
    This function generates a dataframe that holds information about all the frames. Information includes errors
    such as if a frame has a g2/g3/g4 bounding box, if there are multiple bounding box for any of the labels, the
    lag for each label and so on. If we are missing any values, we will leave it black at this point in time.
    If in one frame we have multiple entries for one label (ie one frame has 2 G4 labels), we take the label
    with the highest conf value in our error log. In general, 1 would represent an error for the flags.

    Input:
        - list_of_frames_with_bounding_boxes: list of list that contains information about all bboxs in all frames
        
    Output:
        - error_log: dataframe that contains information (start, end, length, lag, etc) on all the frames and errors

    """

    error_log = pd.DataFrame(columns=[value.name for value in WaveGroupSearchResults])

    # Go through each frame in the scan
    if len(list_of_frames_with_bounding_boxes)!=0:
        for idx, frame in enumerate(list_of_frames_with_bounding_boxes):
            frame_name = f"frame_{idx + 0:03d}"

    
            # group start, end, length, confidence, bbox_count for g2 (0), g3 (1) and g4 (2)
            group_info = {bssd.WaveGroup.G2: ErrorLogFrameInfo(),
                          bssd.WaveGroup.G3: ErrorLogFrameInfo(),
                          bssd.WaveGroup.G4: ErrorLogFrameInfo()}
    
            # check to see that this frame has at least one bounding box (of any classification)
            if len(frame) != 0:
    
                # initialize all error flags to False
                missing_g2 = False
                missing_g3 = False
                missing_g4 = False
                missing_all_labels = False
                labels_not_in_order = False
                repeating_labels = False
                anomaly_g3 = False
                anomaly_g4 = False
                missing_g2_impute = False
                missing_g3_impute = False
                missing_g4_impute = False
                missing_all_labels_impute = False
                labels_not_in_order_impute = False
    
                # Iterate through each bounding box in this frame. For example, there is usually three bounding boxes per
                # frame - one for G2, G3 and G4.
                for bbox in frame:
                    # if a bbox belongs to g5, we ignore it
                    if bbox.classification == bssd.WaveGroup.G5:
                        continue
                    classification_confidence = group_info[bbox.classification].confidence
                    # Since the ML model can return multiple boxes for each wavegroup, only store the once with the highest
                    # confidence score.
                    if bbox.confidence > classification_confidence or np.isnan(classification_confidence):
                        group_info[bbox.classification].group_start = bbox.get_box_start()
                        group_info[bbox.classification].group_end = bbox.get_box_end()
                        group_info[bbox.classification].length = bbox.width
                        group_info[bbox.classification].confidence = bbox.confidence
    
                    # add 1 to bbox counter even if we don't overwrite the group with the bbox w the highest confidence
                    group_info[bbox.classification].bbox_count += 1
    
                # check if group 2 is missing in current frame
                missing_g2 = group_info[bssd.WaveGroup.G2].bbox_count == 0
                # check if group 3 is missing in current frame
                missing_g3 = group_info[bssd.WaveGroup.G3].bbox_count == 0
                # check if group 4 is missing in current frame
                missing_g4 = group_info[bssd.WaveGroup.G4].bbox_count == 0
                # check if there are no labels in current frame
                missing_all_labels = all([missing_g2, missing_g3, missing_g4])
    
                # check if there are repeating labels in current frame
                repeating_labels = any([group_info[wave_group].bbox_count > 1 for wave_group in group_info])
    
                # check all three wave groups to see if there are any invalid end positions
                any_invalid_end_positions = any([(np.isnan(group_info[wave_group].group_end)) for wave_group in group_info])
                # check to see if g2 ends after g3
                g2_ends_after_g3 = group_info[bssd.WaveGroup.G2].group_end > group_info[bssd.WaveGroup.G3].group_end
                # check to see if g3 ends after g4
                g3_ends_after_g4 = group_info[bssd.WaveGroup.G3].group_end > group_info[bssd.WaveGroup.G4].group_end
                labels_not_in_order = any_invalid_end_positions or g2_ends_after_g3 or g3_ends_after_g4
    
            else:
                # Frame has no bounding boxes, mark everything as missing.
                missing_g2 = True
                missing_g3 = True
                missing_g4 = True
                missing_all_labels = True
                labels_not_in_order = False
                repeating_labels = False
                missing_g2_impute = False
                missing_g3_impute = False
                missing_g4_impute = False
                missing_all_labels_impute = False
                labels_not_in_order_impute = False
                anomaly_g3 = False
                anomaly_g4 = False
    
            error_log.loc[len(error_log)] = [frame_name, missing_g2, missing_g3, missing_g4,
                                             missing_all_labels, labels_not_in_order, repeating_labels,
                                             missing_g2_impute, missing_g3_impute, missing_g4_impute,
                                             missing_all_labels_impute, labels_not_in_order_impute,
                                             group_info[bssd.WaveGroup.G2].group_start,
                                             group_info[bssd.WaveGroup.G2].group_end,
                                             group_info[bssd.WaveGroup.G2].length,
                                             group_info[bssd.WaveGroup.G2].confidence,
                                             np.NaN,
                                             group_info[bssd.WaveGroup.G3].group_start,
                                             group_info[bssd.WaveGroup.G3].group_end,
                                             group_info[bssd.WaveGroup.G3].length,
                                             group_info[bssd.WaveGroup.G3].confidence,
                                             np.NaN,
                                             group_info[bssd.WaveGroup.G4].group_start,
                                             group_info[bssd.WaveGroup.G4].group_end,
                                             group_info[bssd.WaveGroup.G4].length,
                                             group_info[bssd.WaveGroup.G4].confidence,
                                             np.NaN, np.NaN,
                                             anomaly_g3, anomaly_g4, 0]  # add all the data into dataframe row

    return error_log


def detect_consistent_missing_g4(wavegroup_search_results: pd.DataFrame, num_missing_g4: int):
    """
    This function will look at the error log and record the amount of g4 labels missing in consistent frames
    (comparing the current to the last frame). The data will then be saved in a text file saved in the path
    defined by the user.

    Input: 
        - path: current root path
        - wavegroup_search_results: dataframe that contains information (start, end, length, lag, etc) on all the frames and errors
        - num_missing_g4: minimum number of consecutive frames missing G4 to be flagged
    
    Output:
        - missing_g4_list: list of list. Element 1 is the subset of consecutive frames missing G4, element 2
          is either all the frames in the subset (missing_g4_list[#][0]) or the total number of frames in the 
          subset (missing_g4_list[#][1])
    """

    # create dataframe containing only the frame number and missing g4 group status
    missing_g4_log = wavegroup_search_results.loc[:, [WaveGroupSearchResults.Frame_name.name, WaveGroupSearchResults.Missing_g4.name]]
    # Filter out all frames where g4 was found
    missing_g4_log = missing_g4_log[missing_g4_log[WaveGroupSearchResults.Missing_g4.name]]
    # list that will contain index of missing g4
    missing_g4_list = []
    #error_log_path = path / 'Consecutive Missing G4.txt'

    # If there are any groups missing the g4 wave group,
    if missing_g4_log.shape[0] != 0:
        consecutive_g4 = []
        # get index of first frame -1
        last_index = missing_g4_log.index[0] - 1

        for index, row in missing_g4_log.iterrows():
            # check if the last index and current index is numerically adjacent
            if index == last_index + 1:
                # if index and last index are adjacent, append frame num
                consecutive_g4.append(int(row[WaveGroupSearchResults.Frame_name.name][6:]))

            # if index and last index are not adjacent
            else:
                # check if number of consecutive missing g4 frames is greater than user threshold
                if len(consecutive_g4) >= num_missing_g4:
                    # save the span of consecutive missing g4
                    missing_g4_list.append([consecutive_g4, len(consecutive_g4)])
                    # reset the list of consecutive missing g4 to have nothing
                    consecutive_g4.clear()
                    consecutive_g4.append(int(row[WaveGroupSearchResults.Frame_name.name][6:]))
                else:
                    # if index and last index are not adjacent and number of consecutive missing g4 is less than user
                    # threshold, reset consecutive missing g4 list
                    consecutive_g4.clear()
                    consecutive_g4.append(int(row[WaveGroupSearchResults.Frame_name.name][6:]))
            # save current index value as last index value
            last_index = index

        # check if number of consecutive missing g4 frames is greater than user threshold
        if len(consecutive_g4) >= num_missing_g4:
            # save the span of consecutive missing g4
            missing_g4_list.append([consecutive_g4, len(consecutive_g4)])

        # based on info above, populate txt file
        # with open(error_log_path, 'w') as f:
        #     f.write('This file will outline all the sets of consecutive frames that have missing G4 labels.\n')
        #     f.write(f"\ns_g3 has {wavegroup_search_results[WaveGroupSearchResults.g3_start.name].isnull().sum()} "
        #             f"missing values before interpolation")
        #     f.write(f"\ns_g4 has {wavegroup_search_results[WaveGroupSearchResults.g4_start.name].isnull().sum()} "
        #             f"missing values before interpolation\n")

        #     for row in range(len(missing_g4_list)):
        #         f.write(f'\nNumber of consecutive G4 frames missing: {missing_g4_list[row][1]}\n')
        #         f.write("consecutive G4 frames missing:\n")
        #         f.write(f'{missing_g4_list[row][0]}\n\n')
    # else:

        # with open(error_log_path, 'w') as f:
        #     f.write('This file will outline all the sets of consecutive frames that have missing G4 labels.\n')
        #     f.write(f"\ns_g3 has {wavegroup_search_results[WaveGroupSearchResults.g3_start.name].isnull().sum()} "
        #             f"missing values before interpolation")
        #     f.write(f"\ns_g4 has {wavegroup_search_results[WaveGroupSearchResults.g4_start.name].isnull().sum()} "
        #             f"missing values before interpolation\n")
        #     f.write('\nThere are no consecutive frames that have missing G4 labels.\n')
        #     f.write(f'\nCurrently the number of consecutive frames needed is {num_missing_g4}.\n')
        #     f.write('\nMaybe try setting variable "num_missing_g4" to a lower value in ML_pipeline.py.\n')

    return missing_g4_list


def outlier_detection(wave_group_idn_results: pd.DataFrame, col_name: str, save_plot: bool, verbose: bool,
                      upper_thresh: float, lower_thresh: float):
    """Looks for outliers in error_log, and for outliers based on the start & end positions of G3 and G4.

    If an outlier is found based on the upper and lower threshold value, it will be replaced by a "NaN" value.

    Input:
        - path: current root path
        - error_log: dataframe that contains information (start, end, length, lag) on all the frames and errors
        - col_name: name of the column you want to do outlier detection on. The 2 arguments accepted
          are ErrorLog.g3_start.name and ErrorLog.g4_start.name
        - verbose: bool, if true outputs debugging info in console
        - upper_thresh: the upper limit for calculating outliers, note that this value represents a percentage
          and is a value between 0 and 1
        - lower_thresh: the lower limit for calculating outliers, note that this value represents a percentage
          and is a value between 0 and 1

    Output:
        - error_log_outlier_detection: dataframe that contains information (start, end, length, lag) on all the
          frames and errors without outlier data

    """

    if col_name != WaveGroupSearchResults.g3_start.name and col_name != WaveGroupSearchResults.g4_start.name:
        raise ValueError(f'Expected g3_start or g4_start, but got {col_name}')

    # Create a copy of the wave group identification results
    outlier_detection_df = copy.deepcopy(wave_group_idn_results)
    # calculate interquartile_range
    q1 = outlier_detection_df[col_name].quantile(lower_thresh)
    q3 = outlier_detection_df[col_name].quantile(upper_thresh)
    interquartile_range = q3 - q1
    upper = outlier_detection_df[col_name] >= (q3 + 1.5 * interquartile_range)
    upper_index = list(upper[upper].index)
    lower = outlier_detection_df[col_name] <= (q1 - 1.5 * interquartile_range)
    lower_index = list(lower[lower].index)
    outlier_index = lower_index + upper_index

    # get copy of col before getting rid of outliers
    pre_outlier_df = copy.deepcopy(pd.Series(outlier_detection_df[col_name]))

    if verbose:  # pragma: no cover
        print(f"\nchecking column: {col_name}")
        print(f"upper limit: {q3 + 1.5 * interquartile_range}")
        print(f"lower limit: {q1 - 1.5 * interquartile_range}")
        print(f"outlier index: {outlier_index}")

    g3_columns_to_write = [WaveGroupSearchResults.g3_start.value,
                           WaveGroupSearchResults.g3_end.value,
                           WaveGroupSearchResults.g3_length.value,
                           WaveGroupSearchResults.g3_confidence.value,
                           WaveGroupSearchResults.Anomaly_G3.value]
    g4_columns_to_write = [WaveGroupSearchResults.g4_start.value,
                           WaveGroupSearchResults.g4_end.value,
                           WaveGroupSearchResults.g4_length.value,
                           WaveGroupSearchResults.g4_confidence.value,
                           WaveGroupSearchResults.Anomaly_G4.value]

    df_columns = g3_columns_to_write if col_name == WaveGroupSearchResults.g3_start.name else g4_columns_to_write

    for index, row in outlier_detection_df.iterrows():  # if the column contains outlier, we replace start,
        if index in outlier_index:  # end, len and conf of that col with np,NAN
            outlier_detection_df.iloc[index, df_columns[0]] = np.nan
            outlier_detection_df.iloc[index, df_columns[1]] = np.nan
            outlier_detection_df.iloc[index, df_columns[2]] = np.nan
            outlier_detection_df.iloc[index, df_columns[3]] = np.nan
            outlier_detection_df.iloc[index, df_columns[4]] = True

    # if save_plot:
    #     post_outlier_df = pd.Series(outlier_detection_df[col_name])  # get a copy of col after getting rid of outliers

    #     df = pd.DataFrame(columns=['Before', "After"])  # make a df with before and after results
    #     df["Before"] = pre_outlier_df
    #     df["After"] = post_outlier_df

    #     df.plot.box()  # make a boxplot
    #     plt.title(f"Outlier Detection: {col_name}")

    #     out_path = path / 'graphs'
    #     mkdir(out_path)

    #     plt.savefig(out_path / f'Outlier Detection for {col_name}')

    return outlier_detection_df


def second_round(probe_data, tf_model, autoanalysis_config, merge_bbox_config: dict):
    """This function will take the sliced images of frames that need to be inferenced on a second time, do
    inferencing and then merge the overlapping bounding boxes from the inferenced data, creating an error_log.

    Input: 
        - output_path: directory to save inference results
        - probe_data: array with img data using 1 channel
        - inference_config: InferenceConfig class which contains:
            - image_size: size of image used in inferencing
            - model_dir: Path that contains the object detection model
            - iou_threshold: iou threshold used for merging bounding boxes
            - confidence_threshold: confidence threshold used to filter out any low confidence bboxes
            - max_wh: max width and height
            - device: optional argument allowing user to actively force inferencing on CPU instead of GPU, default is "gpu", input "cpu" if using on CPU
            - save_output: whether to save the inference output to file
            - verbose: if true, prints optional diagnostic values to the console
        - merge_bbox_config: Dictionary of settings, which contains:
            - iou_thresh: Boxes with an iou overlap above the threshold will be merged
            - same_labels: If true, will only merge overlapping boxes if they have the same label. If False, will merge all overlapping boxes
            - verbose: if true, prints optional diagnostic values to the console

    Output:
        - error_log: dataframe that contains information (start, end, length, lag, etc) on all the frames that 
          are in current folder

    """

    # print("\nSecond round of inference")
    # calling inferencing tool/script
    bbox_data = inferencing_script.inference(probe_data, tf_model, autoanalysis_config)

    # print("\nSecond round of merging bounding boxes")
    bbox_data_merged = []  # list of lists that contains non-overlapping bounding boxes [frame/frame_numb][bbox][bbox info]
    for frame_number, frame in enumerate(bbox_data[:]):
        bbox_data_merged.append(merge_bbox(copy.deepcopy(frame), **merge_bbox_config.__dict__))

    # print("\nCreating second round error log")
    # create error_log
    error_log = create_wavegroup_search_report(bbox_data_merged)

    return error_log


def interpolation(probe_data: np.ndarray, tf_model, error_log_outlier_detection: pd.DataFrame, missing_g4_list, settings: dict):
    """Calculates an interpolated point wherever there is missing data, so there is a prediction for all bbox in every
    frame.

    Input:
        - path: root path for this script
        - probe_data: array with img data using 1 channel
        - error_log_outlier_detection: dataframe that contains information on all the frames after merging and outlier
          detection
        - missing_g4_list: list of list. Element 1 is the subset of consecutive frames missing G4, element 2 
          is either all the frames in the subset (missing_g4_list[#][0]) or the total number of frames in the 
          subset (missing_g4_list[#][1])
        - settings: dictionary containing settings for all the other functions
          
    Output:
        - wavegroup_idn_results: Complete dataframe that contains imputed bbox start and end values with no missing data
        - error_log_impute_slices: Complete dataframe that contains imputed bbox start and end values for G4 
          with no missing data

    Order of operations:
        1. Create a copy of error_log_outlier_detection, to preserve the original error log.
        2. Check for any overlapping G3 / G4 wavegroups
        3. Go through all frames with consecutive missing G4 predictions and select every 'x' frame
            - 'x' is the 'skip_frame' variable from the config
            - The frames that are selected will be used for a second round of inferencing
        4. Begin second round of inferencing. Iterate through every frame in the scan and run inferencing on any frame
           that either:
            - Was selected by the consecutive missing G4 prediction check
            - Has G3G4 overlap which exceeds the configured limit, which indicates that the first round of imputation
              may have placed the G3 or G4 box incorrectly
        5. Rename the inferenced frames output with the source frame number and slice number
            - ie: frame_000 may become frame_031_2
            - Only runs if Save Output is True
        6. Begin breaking up error_log_second_round into smaller pandas dataframes, so that there is one error_log
           dataframe for each frame that was sliced & inferenced
        7. Create error_log_impute_slices.
            1. Iterate through every frame in the scan.
            2. If the current frame went through a second round of inferencing, pull the slice info from error_log_second_round
            3. If the current frame did not go through second inferencing, pull the G4 box from error_log_impute and
               slice it into 5 bboxes, so the bbox list lengths match
        8. Perform outlier detection on error_log_impute_slices to remove:
            - Any slices where the start location lines up poorly with other start locations
            - Any slices where g4 occurs before g3
        9. Perform interpolation on the remaining columns in error_log_impute_slices
            1. Iterate through every frame in the scan
            2. If all slices in the current frame are imputed, and there is a full G4 box prediction in
               error_log_impute, use the full G4 box as the wavegroup bounds
            3. If there are different slices for the current frame, make the G4 box start at the left-most slice
               prediction and end at the right-most slice prediction
        10. Impute all columns in error_log_impute and fill out calculated columns, such as lag
        11. Perform an error check and a second G3G4 overlap check
        12. Return wavegroup_idn_results and error_log_impute_slices

    """
    num_missing_g4 = settings.interpolation.num_missing_g4
    skip_frame = settings.interpolation.skip_frame
    overlap_thresh = settings.interpolation.overlap_thresh
    verbose = settings.interpolation.verbose

    lower_thresh = settings.outlier_rejection.lower_thresh
    upper_thresh = settings.outlier_rejection.upper_thresh

    if verbose:  # pragma: no cover
        print("\n\nImputing missing values from error_log")
        print("Calculating lag from error_log")

    # make a deep copy of error log
    wavegroup_idn_results = copy.deepcopy(error_log_outlier_detection)

    g3_g4_overlap = get_wavegroup_overlap(wavegroup_idn_results, verbose)

    wavegroup_idn_results[WaveGroupSearchResults.g3_g4_overlap.name] = g3_g4_overlap
    wavegroup_idn_results[WaveGroupSearchResults.g3_g4_overlap.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g3_g4_overlap.name].replace(np.NaN, 0)

    # list of frames that we will inference on again
    for_inference = []

    if skip_frame != -1:
        # for every set of consecutive missing g4 frames
        for group in missing_g4_list:
            # look at all the frames in the set of consecutive missing g4 frames
            for frame in range(len(group[0])):
                # if the frame is the 3 frame in the set
                if frame % skip_frame == skip_frame - 1:
                    # append it to list which will later be sliced up and inferenced on
                    for_inference.append(group[0][frame])
    else:
        for index, row in wavegroup_idn_results.iterrows():
            # inference all frames
            for_inference.append(get_frame_number(row))

    # make folder for frames that need to be inferenced again
    # second_inf_path = path / 'frames for second round of inference'
    # mkdir(second_inf_path)

    counter = 0

    # list of frames for second inference
    frames_for_second_infer = []
    # list of frame names for second inference
    frames_for_second_infer_names = []

    # loop through each row in
    for index, row in wavegroup_idn_results.iterrows():
        # get frame number
        frame_num = get_frame_number(row)

        # if frame is in need of second inference or has G3 G4 overlap
        if frame_num in for_inference or row[WaveGroupSearchResults.g3_g4_overlap.name] > overlap_thresh:
            # counter used to split error_log_second_round into sub_df
            counter = counter + 1
            # get frame img data
            img = probe_data[index]
            img_height = img.shape[0]
            section_height = img_height // 5

            for section in range(5):
                # slice frame into 5 equal parts
                start = section * section_height
                end = (section + 1) * section_height
                section_img = img[start:end, :]

                frames_for_second_infer_names.append(f'{row[WaveGroupSearchResults.Frame_name]}_{section}.png')
                frames_for_second_infer.append(section_img)

                # if settings.interpolation.save_second_round:
                #     # save copy of img
                #     cv2.imwrite(second_inf_path / f'{row[WaveGroupSearchResults.Frame_name]}_{section}.png', section_img)

    if len(frames_for_second_infer_names) != 0:
        # There are images that require a second round of inferencing

        # covert data structure to replicate probe_data
        frames_for_second_infer = np.stack(frames_for_second_infer)

        # Run second round of inferencing
        error_log_second_round = second_round(frames_for_second_infer, tf_model, settings, settings.merge_bbox)

        # if settings.inference.save_output:
        #     # Rename files from 'frame_000.png', 'frame_001.png' to the original filename, ie: 'frame_026_1.png'
        #     second_ind_results = second_inf_path / 'inference results'
        #     for index, fn in enumerate(second_ind_results.iterdir()):
        #         fn.rename(fn.with_stem(frames_for_second_infer_names[index]))

        for index, name in enumerate(frames_for_second_infer_names):
            error_log_second_round.iloc[index, 0] = name

        # take the second round error log and brake into sub dfs so each sub df holds data for one frame
        error_log_second_round = np.array_split(error_log_second_round, counter)

        # now we will make new df, that will have frame_name, s_g4_0, e_g4_0, l_g4_0, conf_g4_0, s_g4_1...        
        frame_name = []  # list that will hold data that will be used to make new df
        s_g4_0 = []
        e_g4_0 = []
        l_g4_0 = []
        conf_g4_0 = []
        s_g4_1 = []
        e_g4_1 = []
        l_g4_1 = []
        conf_g4_1 = []
        s_g4_2 = []
        e_g4_2 = []
        l_g4_2 = []
        conf_g4_2 = []
        s_g4_3 = []
        e_g4_3 = []
        l_g4_3 = []
        conf_g4_3 = []
        s_g4_4 = []
        e_g4_4 = []
        l_g4_4 = []
        conf_g4_4 = []
        frame_sliced = []

        for index, row in wavegroup_idn_results.iterrows():  # loop through every row
            frame_num = int(row[WaveGroupSearchResults.Frame_name.name].split("_")[1])  # get frame num
            if frame_num in for_inference or row["g3_g4_overlap"] > overlap_thresh:  # if frame was in second inference
                for sub_df in error_log_second_round:  # loop through all second round dfs
                    if row[WaveGroupSearchResults.Frame_name.name] == sub_df.iloc[0, 0][:-6]:  # if the frame name is same as sub_df frame name
                        frame_name.append(row[WaveGroupSearchResults.Frame_name.name])  # save frame name
                        s_g4_0.append(sub_df[WaveGroupSearchResults.g4_start.name].iloc[0])  # save new start of slice 0 of g4
                        e_g4_0.append(sub_df[WaveGroupSearchResults.g4_end.name].iloc[0])  # save new end of slice 0 of g4
                        l_g4_0.append(sub_df[WaveGroupSearchResults.g4_length.name].iloc[0])  # save new length of slice 0 of g4
                        conf_g4_0.append(sub_df[WaveGroupSearchResults.g4_confidence.name].iloc[0])  # save new conf of slice 0 of g4
                        s_g4_1.append(sub_df[WaveGroupSearchResults.g4_start.name].iloc[1])  # save new start of slice 1 of g4
                        e_g4_1.append(sub_df[WaveGroupSearchResults.g4_end.name].iloc[1])  # save new end of slice 1 of g4
                        l_g4_1.append(sub_df[WaveGroupSearchResults.g4_length.name].iloc[1])  # save new len of slice 1 of g4
                        conf_g4_1.append(sub_df[WaveGroupSearchResults.g4_confidence.name].iloc[1])  # save new conf of slice 1 of g4
                        s_g4_2.append(sub_df[WaveGroupSearchResults.g4_start.name].iloc[2])
                        e_g4_2.append(sub_df[WaveGroupSearchResults.g4_end.name].iloc[2])
                        l_g4_2.append(sub_df[WaveGroupSearchResults.g4_length.name].iloc[2])
                        conf_g4_2.append(sub_df[WaveGroupSearchResults.g4_confidence.name].iloc[2])
                        s_g4_3.append(sub_df[WaveGroupSearchResults.g4_start.name].iloc[3])
                        e_g4_3.append(sub_df[WaveGroupSearchResults.g4_end.name].iloc[3])
                        l_g4_3.append(sub_df[WaveGroupSearchResults.g4_length.name].iloc[3])
                        conf_g4_3.append(sub_df[WaveGroupSearchResults.g4_confidence.name].iloc[3])
                        s_g4_4.append(sub_df[WaveGroupSearchResults.g4_start.name].iloc[4])
                        e_g4_4.append(sub_df[WaveGroupSearchResults.g4_end.name].iloc[4])
                        l_g4_4.append(sub_df[WaveGroupSearchResults.g4_length.name].iloc[4])
                        conf_g4_4.append(sub_df[WaveGroupSearchResults.g4_confidence.name].iloc[4])
                        frame_sliced.append(1)

            else:  # if frame was only in first inference
                frame_name.append(row[WaveGroupSearchResults.Frame_name.name])  # save frame name
                s_g4_0.append(row[WaveGroupSearchResults.g4_start.name])  # save original s_g4
                e_g4_0.append(row[WaveGroupSearchResults.g4_end.name])  # save original e_g4
                l_g4_0.append(row[WaveGroupSearchResults.g4_length.name])  # save original l_g4
                conf_g4_0.append(row[WaveGroupSearchResults.g4_confidence.name])  # save original conf_g4
                s_g4_1.append(row[WaveGroupSearchResults.g4_start.name])
                e_g4_1.append(row[WaveGroupSearchResults.g4_end.name])
                l_g4_1.append(row[WaveGroupSearchResults.g4_length.name])
                conf_g4_1.append(row[WaveGroupSearchResults.g4_confidence.name])
                s_g4_2.append(row[WaveGroupSearchResults.g4_start.name])
                e_g4_2.append(row[WaveGroupSearchResults.g4_end.name])
                l_g4_2.append(row[WaveGroupSearchResults.g4_length.name])
                conf_g4_2.append(row[WaveGroupSearchResults.g4_confidence.name])
                s_g4_3.append(row[WaveGroupSearchResults.g4_start.name])
                e_g4_3.append(row[WaveGroupSearchResults.g4_end.name])
                l_g4_3.append(row[WaveGroupSearchResults.g4_length.name])
                conf_g4_3.append(row[WaveGroupSearchResults.g4_confidence.name])
                s_g4_4.append(row[WaveGroupSearchResults.g4_start.name])
                e_g4_4.append(row[WaveGroupSearchResults.g4_end.name])
                l_g4_4.append(row[WaveGroupSearchResults.g4_length.name])
                conf_g4_4.append(row[WaveGroupSearchResults.g4_confidence.name])
                frame_sliced.append(0)

        # colum that tells us if a frame was slices and went through second inference
        wavegroup_idn_results["Frame_sliced"] = frame_sliced

        # create new dataframe
        error_log_impute_slices = pd.DataFrame(list(zip(frame_name,
                                                        s_g4_0, e_g4_0, l_g4_0, conf_g4_0,
                                                        s_g4_1, e_g4_1, l_g4_1, conf_g4_1,
                                                        s_g4_2, e_g4_2, l_g4_2, conf_g4_2,
                                                        s_g4_3, e_g4_3, l_g4_3, conf_g4_3,
                                                        s_g4_4, e_g4_4, l_g4_4, conf_g4_4, frame_sliced)),
                                               columns=['Frame_name',
                                                        's_g4_0', 'e_g4_0', 'l_g4_0', 'conf_g4_0',
                                                        's_g4_1', 'e_g4_1', 'l_g4_1', 'conf_g4_1',
                                                        's_g4_2', 'e_g4_2', 'l_g4_2', 'conf_g4_2',
                                                        's_g4_3', 'e_g4_3', 'l_g4_3', 'conf_g4_3',
                                                        's_g4_4', 'e_g4_4', 'l_g4_4', 'conf_g4_4', 'frame_sliced'])

        # outlier detection on slices data
        q1 = error_log_impute_slices["s_g4_0"].quantile(lower_thresh)  # calculate interquartile_range
        q3 = error_log_impute_slices["s_g4_0"].quantile(upper_thresh)
        interquartile_range = q3 - q1
        upper = error_log_impute_slices["s_g4_0"] >= (q3 + 1.5 * interquartile_range)
        upper_index = list(upper[upper].index)
        lower = error_log_impute_slices["s_g4_0"] <= (q1 - 1.5 * interquartile_range)
        lower_index = list(lower[lower].index)
        outlier_index = lower_index + upper_index
        for row in range(error_log_impute_slices.shape[0]):  # if the column contains outlier, we replace start,
            if row in outlier_index:
                if wavegroup_idn_results.iloc[row, 31] == 1:
                    error_log_impute_slices.iloc[row, 1] = np.nan
                    error_log_impute_slices.iloc[row, 2] = np.nan
                    error_log_impute_slices.iloc[row, 3] = np.nan
                    error_log_impute_slices.iloc[row, 4] = np.nan
                    wavegroup_idn_results.iloc[row, 29] = 2

        q1 = error_log_impute_slices["s_g4_1"].quantile(lower_thresh)  # calculate interquartile_range
        q3 = error_log_impute_slices["s_g4_1"].quantile(upper_thresh)
        interquartile_range = q3 - q1
        upper = error_log_impute_slices["s_g4_1"] >= (q3 + 1.5 * interquartile_range)
        upper_index = list(upper[upper].index)
        lower = error_log_impute_slices["s_g4_1"] <= (q1 - 1.5 * interquartile_range)
        lower_index = list(lower[lower].index)
        outlier_index = lower_index + upper_index
        for row in range(error_log_impute_slices.shape[0]):  # if the column contains outlier, we replace start,
            if row in outlier_index:
                if wavegroup_idn_results.iloc[row, 31] == 1:
                    error_log_impute_slices.iloc[row, 5] = np.nan
                    error_log_impute_slices.iloc[row, 6] = np.nan
                    error_log_impute_slices.iloc[row, 7] = np.nan
                    error_log_impute_slices.iloc[row, 8] = np.nan
                    wavegroup_idn_results.iloc[row, 29] = 2

        q1 = error_log_impute_slices["s_g4_2"].quantile(lower_thresh)  # calculate interquartile_range
        q3 = error_log_impute_slices["s_g4_2"].quantile(upper_thresh)
        interquartile_range = q3 - q1
        upper = error_log_impute_slices["s_g4_2"] >= (q3 + 1.5 * interquartile_range)
        upper_index = list(upper[upper].index)
        lower = error_log_impute_slices["s_g4_2"] <= (q1 - 1.5 * interquartile_range)
        lower_index = list(lower[lower].index)
        outlier_index = lower_index + upper_index
        for row in range(error_log_impute_slices.shape[0]):  # if the column contains outlier, we replace start,
            if row in outlier_index:
                if wavegroup_idn_results.iloc[row, 31] == 1:
                    error_log_impute_slices.iloc[row, 9] = np.nan
                    error_log_impute_slices.iloc[row, 10] = np.nan
                    error_log_impute_slices.iloc[row, 11] = np.nan
                    error_log_impute_slices.iloc[row, 12] = np.nan
                    wavegroup_idn_results.iloc[row, 29] = 2

        q1 = error_log_impute_slices["s_g4_3"].quantile(lower_thresh)  # calculate interquartile_range
        q3 = error_log_impute_slices["s_g4_3"].quantile(upper_thresh)
        interquartile_range = q3 - q1
        upper = error_log_impute_slices["s_g4_3"] >= (q3 + 1.5 * interquartile_range)
        upper_index = list(upper[upper].index)
        lower = error_log_impute_slices["s_g4_3"] <= (q1 - 1.5 * interquartile_range)
        lower_index = list(lower[lower].index)
        outlier_index = lower_index + upper_index
        for row in range(error_log_impute_slices.shape[0]):  # if the column contains outlier, we replace start,
            if row in outlier_index:
                if wavegroup_idn_results.iloc[row, 31] == 1:
                    error_log_impute_slices.iloc[row, 13] = np.nan
                    error_log_impute_slices.iloc[row, 14] = np.nan
                    error_log_impute_slices.iloc[row, 15] = np.nan
                    error_log_impute_slices.iloc[row, 16] = np.nan
                    wavegroup_idn_results.iloc[row, 29] = 2

        q1 = error_log_impute_slices["s_g4_4"].quantile(lower_thresh)  # calculate interquartile_range
        q3 = error_log_impute_slices["s_g4_4"].quantile(upper_thresh)
        interquartile_range = q3 - q1
        upper = error_log_impute_slices["s_g4_4"] >= (q3 + 1.5 * interquartile_range)
        upper_index = list(upper[upper].index)
        lower = error_log_impute_slices["s_g4_4"] <= (q1 - 1.5 * interquartile_range)
        lower_index = list(lower[lower].index)
        outlier_index = lower_index + upper_index
        for row in range(error_log_impute_slices.shape[0]):  # if the column contains outlier, we replace start,
            if row in outlier_index:
                if wavegroup_idn_results.iloc[row, 31] == 1:
                    error_log_impute_slices.iloc[row, 17] = np.nan
                    error_log_impute_slices.iloc[row, 18] = np.nan
                    error_log_impute_slices.iloc[row, 19] = np.nan
                    error_log_impute_slices.iloc[row, 20] = np.nan
                    wavegroup_idn_results.iloc[row, 29] = 2

        for row in range(error_log_impute_slices.shape[0]):
            if wavegroup_idn_results.iloc[row, 18] > error_log_impute_slices.iloc[row, 2]:  # if e_g3 > e_g4_0, remove data
                error_log_impute_slices.iloc[row, 1] = np.nan
                error_log_impute_slices.iloc[row, 2] = np.nan
                error_log_impute_slices.iloc[row, 3] = np.nan
                error_log_impute_slices.iloc[row, 4] = np.nan
                wavegroup_idn_results.iloc[row, 29] = 2

            if wavegroup_idn_results.iloc[row, 18] > error_log_impute_slices.iloc[row, 6]:  # if e_g3 > e_g4_1, remove data
                error_log_impute_slices.iloc[row, 5] = np.nan
                error_log_impute_slices.iloc[row, 6] = np.nan
                error_log_impute_slices.iloc[row, 7] = np.nan
                error_log_impute_slices.iloc[row, 8] = np.nan
                wavegroup_idn_results.iloc[row, 29] = 2

            if wavegroup_idn_results.iloc[row, 18] > error_log_impute_slices.iloc[row, 10]:  # if e_g3 > e_g4_2, remove data
                error_log_impute_slices.iloc[row, 9] = np.nan
                error_log_impute_slices.iloc[row, 10] = np.nan
                error_log_impute_slices.iloc[row, 11] = np.nan
                error_log_impute_slices.iloc[row, 12] = np.nan
                wavegroup_idn_results.iloc[row, 29] = 2

            if wavegroup_idn_results.iloc[row, 18] > error_log_impute_slices.iloc[row, 14]:  # if e_g3 > e_g4_3, remove data
                error_log_impute_slices.iloc[row, 13] = np.nan
                error_log_impute_slices.iloc[row, 14] = np.nan
                error_log_impute_slices.iloc[row, 15] = np.nan
                error_log_impute_slices.iloc[row, 16] = np.nan
                wavegroup_idn_results.iloc[row, 29] = 2

            if wavegroup_idn_results.iloc[row, 18] > error_log_impute_slices.iloc[row, 18]:  # if e_g3 > e_g4_4, remove data
                error_log_impute_slices.iloc[row, 17] = np.nan
                error_log_impute_slices.iloc[row, 18] = np.nan
                error_log_impute_slices.iloc[row, 19] = np.nan
                error_log_impute_slices.iloc[row, 20] = np.nan
                wavegroup_idn_results.iloc[row, 29] = 2

        # interpolation with all data
        error_log_impute_slices["s_g4_0"] = error_log_impute_slices["s_g4_0"].interpolate(limit_direction='both')
        error_log_impute_slices["e_g4_0"] = error_log_impute_slices["e_g4_0"].interpolate(limit_direction='both')
        error_log_impute_slices["s_g4_1"] = error_log_impute_slices["s_g4_1"].interpolate(limit_direction='both')
        error_log_impute_slices["e_g4_1"] = error_log_impute_slices["e_g4_1"].interpolate(limit_direction='both')
        error_log_impute_slices["s_g4_2"] = error_log_impute_slices["s_g4_2"].interpolate(limit_direction='both')
        error_log_impute_slices["e_g4_2"] = error_log_impute_slices["e_g4_2"].interpolate(limit_direction='both')
        error_log_impute_slices["s_g4_3"] = error_log_impute_slices["s_g4_3"].interpolate(limit_direction='both')
        error_log_impute_slices["e_g4_3"] = error_log_impute_slices["e_g4_3"].interpolate(limit_direction='both')
        error_log_impute_slices["s_g4_4"] = error_log_impute_slices["s_g4_4"].interpolate(limit_direction='both')
        error_log_impute_slices["e_g4_4"] = error_log_impute_slices["e_g4_4"].interpolate(limit_direction='both')

        error_log_impute_slices["l_g4_0"] = error_log_impute_slices["e_g4_0"] - error_log_impute_slices["s_g4_0"]
        error_log_impute_slices["l_g4_1"] = error_log_impute_slices["e_g4_1"] - error_log_impute_slices["s_g4_1"]
        error_log_impute_slices["l_g4_2"] = error_log_impute_slices["e_g4_2"] - error_log_impute_slices["s_g4_2"]
        error_log_impute_slices["l_g4_3"] = error_log_impute_slices["e_g4_3"] - error_log_impute_slices["s_g4_3"]
        error_log_impute_slices["l_g4_4"] = error_log_impute_slices["e_g4_4"] - error_log_impute_slices["s_g4_4"]

        error_log_impute_slices["conf_g4_0"] = error_log_impute_slices["conf_g4_0"].replace(np.NaN, "impute")
        error_log_impute_slices["conf_g4_1"] = error_log_impute_slices["conf_g4_1"].replace(np.NaN, "impute")
        error_log_impute_slices["conf_g4_2"] = error_log_impute_slices["conf_g4_2"].replace(np.NaN, "impute")
        error_log_impute_slices["conf_g4_3"] = error_log_impute_slices["conf_g4_3"].replace(np.NaN, "impute")
        error_log_impute_slices["conf_g4_4"] = error_log_impute_slices["conf_g4_4"].replace(np.NaN, "impute")

    else:
        # There are no images that require a second round of inferencing

        frame_name = []  # list that will hold data that will be used to make new df
        s_g4_0 = []
        e_g4_0 = []
        l_g4_0 = []
        conf_g4_0 = []
        s_g4_1 = []
        e_g4_1 = []
        l_g4_1 = []
        conf_g4_1 = []
        s_g4_2 = []
        e_g4_2 = []
        l_g4_2 = []
        conf_g4_2 = []
        s_g4_3 = []
        e_g4_3 = []
        l_g4_3 = []
        conf_g4_3 = []
        s_g4_4 = []
        e_g4_4 = []
        l_g4_4 = []
        conf_g4_4 = []
        frame_sliced = []

        for index, row in wavegroup_idn_results.iterrows():  # loop through every row

            frame_name.append(row[WaveGroupSearchResults.Frame_name.name])  # save frame name
            s_g4_0.append(row[WaveGroupSearchResults.g4_start.name])  # save original s_g4
            e_g4_0.append(row[WaveGroupSearchResults.g4_end.name])  # save original e_g4
            l_g4_0.append(row[WaveGroupSearchResults.g4_length.name])  # save original l_g4
            conf_g4_0.append(row[WaveGroupSearchResults.g4_confidence])  # save original conf_g4
            s_g4_1.append(row[WaveGroupSearchResults.g4_start.name])
            e_g4_1.append(row[WaveGroupSearchResults.g4_end.name])
            l_g4_1.append(row[WaveGroupSearchResults.g4_length.name])
            conf_g4_1.append(row[WaveGroupSearchResults.g4_confidence])
            s_g4_2.append(row[WaveGroupSearchResults.g4_start.name])
            e_g4_2.append(row[WaveGroupSearchResults.g4_end.name])
            l_g4_2.append(row[WaveGroupSearchResults.g4_length.name])
            conf_g4_2.append(row[WaveGroupSearchResults.g4_confidence])
            s_g4_3.append(row[WaveGroupSearchResults.g4_start.name])
            e_g4_3.append(row[WaveGroupSearchResults.g4_end.name])
            l_g4_3.append(row[WaveGroupSearchResults.g4_length.name])
            conf_g4_3.append(row[WaveGroupSearchResults.g4_confidence])
            s_g4_4.append(row[WaveGroupSearchResults.g4_start.name])
            e_g4_4.append(row[WaveGroupSearchResults.g4_end.name])
            l_g4_4.append(row[WaveGroupSearchResults.g4_length.name])
            conf_g4_4.append(row[WaveGroupSearchResults.g4_confidence])
            frame_sliced.append(0)

        wavegroup_idn_results[
            "Frame_sliced"] = frame_sliced  # colum that tells us if a frame was slices and went through second inference

        # make new dataframe        
        error_log_impute_slices = pd.DataFrame(list(zip(frame_name,
                                                        s_g4_0, e_g4_0, l_g4_0, conf_g4_0,
                                                        s_g4_1, e_g4_1, l_g4_1, conf_g4_1,
                                                        s_g4_2, e_g4_2, l_g4_2, conf_g4_2,
                                                        s_g4_3, e_g4_3, l_g4_3, conf_g4_3,
                                                        s_g4_4, e_g4_4, l_g4_4, conf_g4_4, frame_sliced)),
                                               columns=['Frame_name',
                                                        's_g4_0', 'e_g4_0', 'l_g4_0', 'conf_g4_0',
                                                        's_g4_1', 'e_g4_1', 'l_g4_1', 'conf_g4_1',
                                                        's_g4_2', 'e_g4_2', 'l_g4_2', 'conf_g4_2',
                                                        's_g4_3', 'e_g4_3', 'l_g4_3', 'conf_g4_3',
                                                        's_g4_4', 'e_g4_4', 'l_g4_4', 'conf_g4_4', 'frame_sliced'])

    # rewrite start and end location of bbox based on slices data
    for index, row in wavegroup_idn_results.iterrows():

        s_g4_0 = error_log_impute_slices.iloc[index, 1]
        e_g4_0 = error_log_impute_slices.iloc[index, 2]
        conf_g4_0 = error_log_impute_slices.iloc[index, 4]
        s_g4_1 = error_log_impute_slices.iloc[index, 5]
        e_g4_1 = error_log_impute_slices.iloc[index, 6]
        conf_g4_1 = error_log_impute_slices.iloc[index, 8]
        s_g4_2 = error_log_impute_slices.iloc[index, 9]
        e_g4_2 = error_log_impute_slices.iloc[index, 10]
        conf_g4_2 = error_log_impute_slices.iloc[index, 12]
        s_g4_3 = error_log_impute_slices.iloc[index, 13]
        e_g4_3 = error_log_impute_slices.iloc[index, 14]
        conf_g4_3 = error_log_impute_slices.iloc[index, 16]
        s_g4_4 = error_log_impute_slices.iloc[index, 17]
        e_g4_4 = error_log_impute_slices.iloc[index, 18]
        conf_g4_4 = error_log_impute_slices.iloc[index, 20]

        # if all the slices in a frame are imputed and we have a full bbox for G4, base slices of full bbox
        if conf_g4_0 == "impute" and conf_g4_1 == "impute" and conf_g4_2 == "impute" and conf_g4_3 == "impute" and conf_g4_4 == "impute":
            if not pd.isna(row[WaveGroupSearchResults.g4_confidence]):
                error_log_impute_slices.iloc[index, 1] = row[WaveGroupSearchResults.g4_start.name]
                error_log_impute_slices.iloc[index, 5] = row[WaveGroupSearchResults.g4_start.name]
                error_log_impute_slices.iloc[index, 9] = row[WaveGroupSearchResults.g4_start.name]
                error_log_impute_slices.iloc[index, 13] = row[WaveGroupSearchResults.g4_start.name]
                error_log_impute_slices.iloc[index, 17] = row[WaveGroupSearchResults.g4_start.name]

                error_log_impute_slices.iloc[index, 2] = row[WaveGroupSearchResults.g4_end.name]
                error_log_impute_slices.iloc[index, 6] = row[WaveGroupSearchResults.g4_end.name]
                error_log_impute_slices.iloc[index, 10] = row[WaveGroupSearchResults.g4_end.name]
                error_log_impute_slices.iloc[index, 14] = row[WaveGroupSearchResults.g4_end.name]
                error_log_impute_slices.iloc[index, 18] = row[WaveGroupSearchResults.g4_end.name]

        # otherwise base new full bbox of slices
        else:
            s_g4_min = min(s_g4_0, s_g4_1, s_g4_2, s_g4_3, s_g4_4, )
            e_g4_max = max(e_g4_0, e_g4_1, e_g4_2, e_g4_3, e_g4_4, )

            row[WaveGroupSearchResults.g4_start.name] = s_g4_min
            row[WaveGroupSearchResults.g4_end.name] = e_g4_max

    # interpolation with all data
    wavegroup_idn_results[WaveGroupSearchResults.g2_start.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g2_start.name].interpolate(
        limit_direction='both')  # interpolate start and end of each label
    wavegroup_idn_results[WaveGroupSearchResults.g2_end.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g2_end.name].interpolate(
        limit_direction='both')  # limit_direction='both' will fill consecutive NaNs at both ends
    wavegroup_idn_results[WaveGroupSearchResults.g3_start.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g3_start.name].interpolate(
        limit_direction='both')
    wavegroup_idn_results[WaveGroupSearchResults.g3_end.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g3_end.name].interpolate(limit_direction='both')
    wavegroup_idn_results[WaveGroupSearchResults.g4_start.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g4_start.name].interpolate(
        limit_direction='both')
    wavegroup_idn_results[WaveGroupSearchResults.g4_end.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g4_end.name].interpolate(limit_direction='both')

    # calculate distance from end to start of label
    wavegroup_idn_results[WaveGroupSearchResults.g2_length.name] = wavegroup_idn_results[WaveGroupSearchResults.g2_end.name] - \
                                                                   wavegroup_idn_results[WaveGroupSearchResults.g2_start.name]
    wavegroup_idn_results[WaveGroupSearchResults.g3_length.name] = wavegroup_idn_results[WaveGroupSearchResults.g3_end.name] - \
                                                                   wavegroup_idn_results[WaveGroupSearchResults.g3_start.name]
    wavegroup_idn_results[WaveGroupSearchResults.g4_length.name] = wavegroup_idn_results[WaveGroupSearchResults.g4_end.name] - \
                                                                   wavegroup_idn_results[WaveGroupSearchResults.g4_start.name]

    wavegroup_idn_results[WaveGroupSearchResults.g2_lag.name] = wavegroup_idn_results[
        WaveGroupSearchResults.g2_start.name].diff()  # calculate lag of each label by subtracting current label with same label from previous frame
    wavegroup_idn_results[WaveGroupSearchResults.g3_lag.name] = wavegroup_idn_results[WaveGroupSearchResults.g3_start.name].diff()
    wavegroup_idn_results[WaveGroupSearchResults.g4_lag.name] = wavegroup_idn_results[WaveGroupSearchResults.g4_start.name].diff()
    wavegroup_idn_results[WaveGroupSearchResults.avg_lag.name] = (wavegroup_idn_results[WaveGroupSearchResults.g2_lag.name] +
                                                                  wavegroup_idn_results[WaveGroupSearchResults.g3_lag.name] +
                                                                  wavegroup_idn_results[WaveGroupSearchResults.g4_lag.name]) / 3

    # for imputed bbox, make conf = " impute" so we know it's not based on the model predictions
    wavegroup_idn_results[WaveGroupSearchResults.g2_confidence.name] = wavegroup_idn_results[WaveGroupSearchResults.g2_confidence.name].replace(np.NaN, "impute")
    wavegroup_idn_results[WaveGroupSearchResults.g3_confidence.name] = wavegroup_idn_results[WaveGroupSearchResults.g3_confidence.name].replace(np.NaN, "impute")
    wavegroup_idn_results[WaveGroupSearchResults.g4_confidence.name] = wavegroup_idn_results[WaveGroupSearchResults.g4_confidence.name].replace(np.NaN, "impute")

    # error flag calculations
    for index, row in wavegroup_idn_results.iterrows():
        missing_g2_impute = 0  # preset all error flags
        missing_g3_impute = 0
        missing_g4_impute = 0
        missing_all_labels_impute = 0
        labels_not_in_order_impute = 0

        if pd.isnull(row[WaveGroupSearchResults.g2_start.name]):  # check flag condition
            missing_g2_impute = 1
        if pd.isnull(row[WaveGroupSearchResults.g3_start.name]):
            missing_g3_impute = 1
        if pd.isnull(row[WaveGroupSearchResults.g4_start.name]):
            missing_g4_impute = 1
        if missing_g2_impute == 1 and missing_g3_impute == 1 and missing_g4_impute == 1:
            missing_all_labels_impute = 1
        if missing_g2_impute == 1 or missing_g3_impute == 1 or missing_g4_impute == 1 or \
                (row[WaveGroupSearchResults.g2_start.name] >= row[WaveGroupSearchResults.g3_start.name] or row[
                    WaveGroupSearchResults.g3_start.name] >= row[
                     WaveGroupSearchResults.g4_start.name]):
            labels_not_in_order_impute = 1

        wavegroup_idn_results.iloc[index, 7] = missing_g2_impute  # update flag value
        wavegroup_idn_results.iloc[index, 8] = missing_g3_impute
        wavegroup_idn_results.iloc[index, 9] = missing_g4_impute
        wavegroup_idn_results.iloc[index, 10] = missing_all_labels_impute
        wavegroup_idn_results.iloc[index, 11] = labels_not_in_order_impute

    consistent_missing_g4 = []
    missing_g4_frames = []
    for group in missing_g4_list:
        for frame in range(len(group[0])):
            missing_g4_frames.append(group[0][frame])
    for index, row in wavegroup_idn_results.iterrows():
        if get_frame_number(row) in missing_g4_frames:
            consistent_missing_g4.append(1)
        else:
            consistent_missing_g4.append(0)
    # make new col to identify which frames were part of consecutive frames that were missing G4
    wavegroup_idn_results["Consistent Missing G4"] = consistent_missing_g4

    g3_g4_overlap = get_wavegroup_overlap(wavegroup_idn_results, verbose)
    wavegroup_idn_results["G3 G4 Overlap"] = g3_g4_overlap

    # print(f"s_g3 has {wavegroup_idn_results[WaveGroupSearchResults.g3_start.name].isnull().sum()} missing values after interpolation")
    # if wavegroup_idn_results[WaveGroupSearchResults.g3_start.name].isnull().sum() != 0:
    #     print("There are missing s_g3 after interpolation, please check error_log_impute!")
    # print(f"s_g4 has {wavegroup_idn_results[WaveGroupSearchResults.g4_start.name].isnull().sum()} missing values after interpolation")
    # if wavegroup_idn_results[WaveGroupSearchResults.g4_start.name].isnull().sum() != 0:
    #     print("There are missing s_g4 after interpolation, please check error_log_impute!")

    return wavegroup_idn_results, error_log_impute_slices


def get_wavegroup_overlap(wavegroup_idn_results: pd.DataFrame, verbose: bool):
    """Calculates the amount of overlap between the G3 and G4 wavegroups, using the intersection-over-union (iou) method

    Args:
        wavegroup_idn_results: pandas dataframe of results, ie: error_log
        verbose: if true, prints coordinates and overlap to the console

    Returns:
        g3_g4_overlap: list of G3-G4 iou for every frame
    """
    g3_g4_overlap = []  # check for overlap in error_log
    for row in range(wavegroup_idn_results.shape[0]):
        bb1_x1 = wavegroup_idn_results.iloc[row, WaveGroupSearchResults.g3_start]  # s_g3
        bb1_x2 = wavegroup_idn_results.iloc[row, WaveGroupSearchResults.g3_end]  # e_g3
        bb2_x1 = wavegroup_idn_results.iloc[row, WaveGroupSearchResults.g4_start]  # s_g4
        bb2_x2 = wavegroup_idn_results.iloc[row, WaveGroupSearchResults.g4_end]  # e_g4
        len_g4 = wavegroup_idn_results.iloc[row, WaveGroupSearchResults.g4_length]  # l_g4
        frame_num = wavegroup_idn_results.iloc[row, WaveGroupSearchResults.Frame_name]  # frame_name

        x_left = max(bb1_x1, bb2_x1)  # x-coordinate of the bbox overlap (left side)
        x_right = min(bb1_x2, bb2_x2)  # x-coordinate of the bbox overlap (right side)
        iou_area = x_right - x_left  # calculate the length of bbox intersection
        if verbose:  # pragma: no cover
            print("\n")
            print(f"X left: {x_left}")
            print(f"X right: {x_right}")
            print(f"Iou area: {iou_area}")

        if iou_area > 0:
            iou = iou_area / len_g4  # calculate iou value
            if verbose:  # pragma: no cover
                print(f"G3 G4 overlap for frame {frame_num}: {iou}")
        else:
            iou = 0
            if verbose:  # pragma: no cover
                print(f"G3 G4 overlap for frame {frame_num}: {iou}")

        g3_g4_overlap.append(iou)
    return g3_g4_overlap


def get_frame_number(row):
    """Extracts the frame number from the frame name"""
    return int(row[WaveGroupSearchResults.Frame_name.name].split("_")[1])


def plot_b_scans(probe_data, path, error_log_impute, error_log_impute_slices, frames_ranges, scale_percent, show_b_scans, save_b_scans):
    """This function will extract information from error_log_impute & error_log_impute_slices, and optionally displays and / or saves it.
    
    Input:
        - probe_data: array with img data using 1 channel
        - path: path to root folder
        - error_log_impute: dataframe that contains information on all the frames with no missing values 
        - error_log_impute_slices: dataframe that contains information on all the frames for G4 with no 
          missing values 
        - frames_ranges: A list of range objects containing the range of G4 locations after Imputations.
        - scale_percent: float 0 <= x <= 1 to scale the size of the B-scan to display on the screen
        - show_bscans: bool, if true creates a pop-up window to show every plot then saves to path
        - save_plot: bool, if true save all plots if false only saves necessary plots
            
    Output:
        None

    """
    if not show_b_scans:
        plt.ioff()

    compare_path = path / 'comparing results'  # create new folder called 'comparing results'
    mkdir(compare_path, delete_existing=True)

    if save_b_scans:
        postprocessing_path = path / 'postprocessing results'  # create new folder called 'postprocessing results'
        mkdir(postprocessing_path, delete_existing=True)
        slices_path = path / 'postprocessing slices results'  # create new folder called 'postprocessing slicing results'
        mkdir(slices_path, delete_existing=True)

    counter = 0

    with trange(error_log_impute.shape[0], ascii=True) as pbar:
        pbar.set_description("Plotting BScans")
        for index, frame in error_log_impute.iterrows():  # for every row in error_log_impute which represents every frame

            # (group_number, start, end, confidence, color) these are our pointers to the error logs
            postprocessing_group_info = (
                ("2", WaveGroupSearchResults.g2_start.name, WaveGroupSearchResults.g2_end.name,
                 WaveGroupSearchResults.g2_confidence.name, (0, 255, 0)),
                ("3", WaveGroupSearchResults.g3_start.name, WaveGroupSearchResults.g3_end.name,
                 WaveGroupSearchResults.g3_confidence.name, (255, 0, 0)),
                ("4", WaveGroupSearchResults.g4_start.name, WaveGroupSearchResults.g4_end.name,
                 WaveGroupSearchResults.g4_confidence, (0, 0, 255)))

            slices_group_info = (("0", 1, 2, 4, (0, 0, 255)),
                                 ("1", 5, 6, 8, (0, 0, 255)),
                                 ("2", 9, 10, 12, (0, 0, 255)),
                                 ("3", 13, 14, 16, (0, 0, 255)),
                                 ("4", 17, 18, 20, (0, 0, 255)))

            img_name = frame.iloc[0]
            img_grayscale = probe_data[index]
            # original img is grayscale so it only has one channel. Give it 3 channels so it acts like an rgb image.
            img = cv2.merge((img_grayscale, img_grayscale, img_grayscale))
            height, width, channel = img.shape  # get dimensions of B-scan image
            y1 = 0  # set y values of bbox we plan on displaying spanning the full height of the B-scan
            y2 = height
            g3g4_overlap = frame["G3 G4 Overlap"]

            # Add the bounding box and text for postprocessing_result
            postprocess_img = copy.deepcopy(img)
            for group in postprocessing_group_info:
                if not np.isnan(frame[group[1]]):
                    start = frame[group[1]]  # s_gx
                    end = frame[group[2]]  # e_gx
                    conf = frame[group[3]]  # conf_gx

                    if conf == "impute":
                        text = f"G{group[0]}, imputed"
                    elif conf == "2nd inference":
                        text = f"G{group[0]}, 2nd inference"
                    else:
                        text = f"G{group[0]}, {round(float(conf) * 100, 1)}%"

                    x1 = round(start * width)
                    x2 = round(end * width)
                    y1_text = y1 + int(group[0]) * 100  # stagger the text for each group to increase readability
                    cv2.rectangle(postprocess_img, (x1, y1), (x2, y2), group[-1], 5)  # draw bbox and text
                    cv2.putText(postprocess_img, text, (x1, y1_text), cv2.FONT_HERSHEY_SIMPLEX, 2, group[-1], 4,
                                cv2.LINE_AA)

            postprocess_img_width = int(postprocess_img.shape[1] * scale_percent)
            postprocess_img_height = int(postprocess_img.shape[0] * scale_percent)
            # resize postprocessor img
            postprocess_img = cv2.resize(postprocess_img, (postprocess_img_width, postprocess_img_height),
                                         interpolation=cv2.INTER_AREA)
            # show plot in new window if flag is true
            if show_b_scans:
                cv2.imshow(f"{img_name}.png", postprocess_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # save plot if flag is true
            if save_b_scans:
                cv2.imwrite(os.path.join(postprocessing_path, f"{img_name}.png"), postprocess_img)

            # Add bounding box and text for postprocessor_slices
            slices_img = copy.deepcopy(img)
            section_height = height // 5

            s_g3 = error_log_impute.iloc[index, 17]  # get G3 location
            e_g3 = error_log_impute.iloc[index, 18]
            conf_g3 = error_log_impute.iloc[index, 20]

            y1_g3 = 0
            y2_g3 = height

            if conf_g3 == "impute":
                text = "G3, imputed"
            elif conf_g3 == "2nd inference":
                text = "G3, 2nd inference"
            else:
                text = f"G3, {round(float(conf_g3) * 100, 1)}%"

            if not np.isnan(s_g3):  # if G3 bbox exist
                x1_g3 = round(s_g3 * width)
                x2_g3 = round(e_g3 * width)
                y1_g3_text = y1_g3 + 200  # stagger the text for each group to increase readability
                cv2.rectangle(slices_img, (x1_g3, y1_g3), (x2_g3, y2_g3), (255, 0, 0), 5)  # draw bbox and text
                cv2.putText(slices_img, text, (x1_g3, y1_g3_text), cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 4, cv2.LINE_AA)

            # Add the bounding box and text for each group
            for group in slices_group_info:

                start = error_log_impute_slices.iloc[index, group[1]]  # s_gx
                end = error_log_impute_slices.iloc[index, group[2]]  # e_gx
                conf = error_log_impute_slices.iloc[index, group[3]]  # conf_gx

                if not np.isnan(start):  # if bbox exist
                    y1 = section_height * int(group[0])
                    y2 = section_height * (int(group[0]) + 1)

                    if conf == "impute":
                        text = f"g4_{group[0]}, imputed"
                    elif conf == "2nd inference":
                        text = f"g4_{group[0]}, 2nd inference"
                    else:
                        text = f"g4_{group[0]}, {round(float(conf) * 100, 1)}%"

                    x1 = round(start * width)
                    x2 = round(end * width)
                    y1_text = y1 + 100  # stagger the text for each group to increase readability
                    cv2.rectangle(slices_img, (x1, y1), (x2, y2), group[-1], 5)  # draw bbox and text
                    cv2.putText(slices_img, text, (x1, y1_text), cv2.FONT_HERSHEY_SIMPLEX, 2, group[-1], 4, cv2.LINE_AA)

            slice_width = int(slices_img.shape[1] * scale_percent)
            slice_height = int(slices_img.shape[0] * scale_percent)
            # resize slices img
            slices_img = cv2.resize(slices_img, (slice_width, slice_height), interpolation=cv2.INTER_AREA)
            # show plot in new window if flag is true
            if show_b_scans:
                cv2.imshow(f"{img_name}.png", slices_img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()
            # save plot if flag is true
            if save_b_scans:
                cv2.imwrite(os.path.join(slices_path, f"{img_name}.png"), slices_img)

            # Add combined img
            # create figure
            plt.figure(counter)
            fig = plt.figure(figsize=(20, 20))

            # setting values to rows and column variables
            rows = 1
            columns = 4

            # Adds a subplot at the 1st position
            fig.add_subplot(rows, columns, 1)

            # showing inference image
            inference_img = cv2.imread(str(path / "inference results" / f'{img_name}.png'))
            infer_width = int(inference_img.shape[1] * scale_percent)
            infer_height = int(inference_img.shape[0] * scale_percent)
            inference_img = cv2.resize(inference_img, (infer_width, infer_height), interpolation=cv2.INTER_AREA)
            plt.imshow(inference_img)
            plt.axis('off')
            plt.title("Inference", fontsize=40)

            # Adds a subplot at the 2nd position
            fig.add_subplot(rows, columns, 2)

            # showing postprocessor image
            plt.imshow(postprocess_img)
            plt.axis('off')
            plt.title(f"Postprocessor\nOverlap: {round(g3g4_overlap * 100, 2)}%", fontsize=30)

            # Adds a subplot at the 3rd position
            fig.add_subplot(rows, columns, 3)

            # showing postprocessor image
            plt.imshow(slices_img)
            plt.axis('off')
            plt.title("Slices", fontsize=40)

            # Adds a subplot at the 4th position
            fig.add_subplot(rows, columns, 4)

            # showing postprocessor image
            if len(frames_ranges[counter]) == 1:
                plt.imshow(postprocess_img)
                plt.axis('off')
                plt.title("Final Result", fontsize=40)
            if len(frames_ranges[counter]) == 5:
                plt.imshow(slices_img)
                plt.axis('off')
                plt.title("Final Result", fontsize=40)
            # save img
            plt.savefig(compare_path / f'{img_name}', bbox_inches="tight")
            plt.clf()
            plt.close('all')

            counter = counter + 1  # counter existed so we can create new figures rather than overwriting our figures
            pbar.update(1)


def plot_all_wavegroup_start_and_end(path: Path, error_log: pd.DataFrame, title: str):
    """
    This function will extract information from error_log and plot the start and end points of G2, G3, G4 for all frames in the error_log.
    The image will then be saved to a graph folder in the user specified path.

    Input:
        - path: path to images you want to inference 
        - error_log: dataframe that contains information on all the frames 
        - title: title of plot
    Output:
        - displays/saves graph of start and end points of frames based on error_log 

    """
    out_path = path / 'graphs'
    mkdir(out_path)

    # Get the wavegroup start and end locations for each wavegroup
    df = pd.concat([error_log[WaveGroupSearchResults.g2_start.name], error_log[WaveGroupSearchResults.g2_end.name],
                    error_log[WaveGroupSearchResults.g3_start.name], error_log[WaveGroupSearchResults.g3_end.name],
                    error_log[WaveGroupSearchResults.g4_start.name], error_log[WaveGroupSearchResults.g4_end.name]],
                   axis=1)
    # Create array from 0 to the max frame number. Ex. 0 ,1 ,2 ... 64, 65
    x = np.arange(0, error_log.shape[0], 1)
    plt.ioff()
    plt.figure()
    plt.xlabel("Frame")
    plt.ylabel("Value")
    plt.title(title)
    plt.plot(x, df[WaveGroupSearchResults.g2_start.name], color='g', linestyle='solid', label="G2")
    plt.plot(x, df[WaveGroupSearchResults.g2_end.name], color='g', linestyle='solid')
    plt.plot(x, df[WaveGroupSearchResults.g3_start.name], color='b', linestyle='solid', label="G3")
    plt.plot(x, df[WaveGroupSearchResults.g3_end.name], color='b', linestyle='solid')
    plt.plot(x, df[WaveGroupSearchResults.g4_start.name], color='r', linestyle='solid', label="G4")
    plt.plot(x, df[WaveGroupSearchResults.g4_end.name], color='r', linestyle='solid')
    plt.legend(loc="best")

    plt.savefig(str(out_path / f'{title}'))


def postprocessor(
    probe_data: np.ndarray,
    tf_model,
    bbox_data: List[InferenceOutput],
    autoanalysis_config: Dict
) -> tuple:
    """
    Executes postprocessing on probe data using the specified TensorFlow model and bounding box data.

    Parameters
    ----------
    probe_data : np.ndarray
        Image data using a single channel.
    tf_model : object
        TensorFlow model used for processing.
    bbox_data : List[InferenceOutput]
        List of bounding box data per frame. Each entry contains information about 
        bounding boxes in the format [x_c, y_c, width, height, conf, label, bbox_number, frame name].
    autoanalysis_config : Dict
        Configuration dictionary containing settings.

    Returns
    -------
    tuple
        frames_ranges : List
            Ranges of G4 locations post-imputation.
        bbox_data : List[InferenceOutput]
            Original bounding box data.
        bbox_data_merged : List[InferenceOutput]
            Merged bounding box data per frame.
        error_log : pd.DataFrame
            Information about frames after merging/selecting bounding boxes.
        missing_g4_list : List
            Frames with missing G4 box, formatted as [subset of consecutive frames, frame count].
        error_log_outlier_detection : pd.DataFrame
            Information about frames post-outlier detection.
        error_log_impute : pd.DataFrame
            Information about frames post-interpolation.
        error_log_impute_slices : pd.DataFrame
            Information about frames sliced into equal parts post-interpolation.
    """

    # list of lists that contains non-overlapping bounding boxes [frame/frame_numb][bbox][bbox info]
    bbox_data_merged = []
    # merge overlapping bounding boxes
    for frame_number, frame in enumerate(bbox_data[:]):
        bbox_data_merged.append(merge_bbox(copy.deepcopy(frame), **autoanalysis_config.merge_bbox.__dict__))

    error_log = create_wavegroup_search_report(bbox_data_merged)

    # plot_all_wavegroup_start_and_end(output_path, error_log, "Starting and End locations for Wavegroups")

    # check for consecutive missing G4 frames

    missing_g4_list = detect_consistent_missing_g4(error_log, autoanalysis_config.interpolation.num_missing_g4)


    error_log_outlier_detection = outlier_detection(error_log,
                                                    WaveGroupSearchResults.g3_start.name,
                                                    **autoanalysis_config.outlier_rejection.__dict__)
    error_log_outlier_detection = outlier_detection(error_log_outlier_detection,
                                                    WaveGroupSearchResults.g4_start.name,
                                                    **autoanalysis_config.outlier_rejection.__dict__)

   
    error_log_impute, error_log_impute_slices = interpolation(probe_data, tf_model, error_log_outlier_detection,
                                                              missing_g4_list, autoanalysis_config)

    
    # plot_all_wavegroup_start_and_end(output_path, error_log_impute,
    #                                  "Starting and End locations for Wavegroups after Imputation")



    # error_log_impute.to_csv(os.path.join(output_path, "General_info_1.csv"), index=False)
    # error_log_impute_slices.to_csv(os.path.join(output_path, "General_info_2.csv"), index=False)

    # get a range object that tells us the location of G4 for every frame
    frames_ranges = []

    for i in range(len(error_log_impute[WaveGroupSearchResults.g4_start.name])):
        if error_log_impute['G3 G4 Overlap'][i] > autoanalysis_config.interpolation.overlap_thresh:
            frame_range_0 = [np.round(error_log_impute_slices['s_g4_0'][i] * probe_data.shape[2]),
                             np.round(error_log_impute_slices['e_g4_0'][i] * probe_data.shape[2])]
            frame_range_1 = [np.round(error_log_impute_slices['s_g4_1'][i] * probe_data.shape[2]),
                             np.round(error_log_impute_slices['e_g4_1'][i] * probe_data.shape[2])]
            frame_range_2 = [np.round(error_log_impute_slices['s_g4_2'][i] * probe_data.shape[2]),
                             np.round(error_log_impute_slices['e_g4_2'][i] * probe_data.shape[2])]
            frame_range_3 = [np.round(error_log_impute_slices['s_g4_3'][i] * probe_data.shape[2]),
                             np.round(error_log_impute_slices['e_g4_3'][i] * probe_data.shape[2])]
            frame_range_4 = [np.round(error_log_impute_slices['s_g4_4'][i] * probe_data.shape[2]),
                             np.round(error_log_impute_slices['e_g4_4'][i] * probe_data.shape[2])]
            frame_range = np.asarray([frame_range_0, frame_range_1, frame_range_2, frame_range_3, frame_range_4])

        else:
            frame_start, frame_end = error_log_impute[WaveGroupSearchResults.g4_start.name][i], \
                                     error_log_impute[WaveGroupSearchResults.g4_end.name][i]
            if np.isnan(frame_start):
                frame_start = 0
            if np.isnan(frame_end):
                frame_end = 0

            frame_range = np.asarray([round(frame_start * probe_data.shape[2]),
                                      round(frame_end * probe_data.shape[2])], dtype=np.int16)
        frames_ranges.append(frame_range)

    # if autoanalysis_config.plot_b_scans.save_b_scans or autoanalysis_config.plot_b_scans.show_b_scans:
    #     plot_b_scans(probe_data, output_path, error_log_impute, error_log_impute_slices, frames_ranges, **autoanalysis_config.plot_b_scans.__dict__)

    return frames_ranges, bbox_data, bbox_data_merged, error_log, missing_g4_list, error_log_outlier_detection, \
           error_log_impute, error_log_impute_slices
