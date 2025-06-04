# -*- coding: utf-8 -*-
"""
Inferencing Script for Auto-Analysis project
"""

import cv2
import os
import numpy as np
import time
from tensorflow.python.client import device_lib
from tensorflow import saved_model
from tqdm.auto import trange
from typing import List
from pathlib import Path
from b_scan_reader import bscan_structure_definitions as bssd
from utils.folder_utils import mkdir
from utils.logger_init import create_dual_loggers

# create loggers
dual_logger, _  = create_dual_loggers()

class InferenceOutput:
    """
        Results of inferencing. Box location is stored yolo-style, as (x_mid, y_mid, width, height)
    """
    classification: bssd.WaveGroup = bssd.WaveGroup.NA
    x: float = -1.0
    y: float = -1.0
    width: float = -1.0
    height: float = -1.0
    confidence: float = -1.0
    frame_number: int = -1

    def __init__(self, classification, x, y, width, height, confidence, frame_number, image_shape):
        self.classification = classification
        self.x = x
        self.y = y
        self.width = width
        self.height = height
        self.confidence = confidence
        self.frame_number = frame_number
        self.image_shape = image_shape

    def __str__(self):
        format_string = 'Classification: {0}, X: {1}, Y: {2}, Width: {3}, Height: {4}, Confidence: {5}, Frame Number {6}'
        pretty_inf_output = format_string.format(self.classification, self.x, self.y, self.width, self.height,
                                                 self.confidence, self.frame_number)
        return pretty_inf_output

    def __repr__(self):
        return '[' + self.__str__() + ']'

    def __eq__(self, other):
        if isinstance(other, self.__class__):
            return self.__dict__ == other.__dict__
        else:
            return False

    def get_box_start(self):
        return self.x - (self.width / 2)

    def get_box_end(self):
        return self.x + (self.width / 2)

    def as_bounds(self, normalize=False, as_range=False, true_height=0):
        """
        Returns the box in bounding-box style, ie: (left, top, right, bottom)

        The as_range parameter changes determines how the right & bottom (ie: x2, y2) elements are returned.
        Eg: if the image is (100, 100) and the flaw lies in the bottom-right corner,
            as_range = False: (x2, y2) will equal (99, 99)
            as_range = True: (x2, y2) will equal (100, 100)

        Set as_range to False if you intend to index the elements directly, or in functions such as cv2.rectangle()
        Set as_range to True if you intend to index the elements using [x1:x2, y1:y2], or use the elements in functions such as range().

        Args:
            normalize: If False, returns pixel-coordinates based on self.image_shape. If True, returns coordinates as a 0-1 float
            as_range: If True, returns x2 and y2 as values such that they can be used easily in range().
            true_height: If normalize is True, this is the true height of the image. If 0, the image height from array is used.
        Returns:
            left, top, right, bottom - positions of the bounding box
        """
        if true_height == 0:
            max_height = self.image_shape[0]
        else:
            max_height = true_height
        max_width = self.image_shape[1]
        img_width = 1 if normalize else max_width
        left = (self.x - (self.width / 2)) * img_width
        right = (self.x + (self.width / 2)) * img_width

        img_height = 1 if normalize else max_height
        top = (self.y - (self.height / 2)) * img_height
        bottom = (self.y + (self.height / 2)) * img_height

        # clipping the values ensures that the signal does not get multiplied outside the bounds of the frame
        left = np.clip(left, 0, max_width)
        right = np.clip(right, 0, max_width)
        top = np.clip(top, 0, max_height)
        bottom = np.clip(bottom, 0, max_height)

        if normalize:
            return left, top, right, bottom
        else:
            if not as_range:
                # subtract each value by one to convert from pixel number to index ie. 3600 was set as the right bound
                # for a frame, and that is an invalid index
                right -= 1.0
                bottom -= 1.0
            # obtaining the conservative extent of flaw after rounding
            left = np.floor(left) if np.floor(left) >= 0 else 0
            top = np.floor(top) if np.floor(top) >= 0 else 0
            right = np.ceil(right) if np.ceil(right) < max_width else max_width
            bottom = np.ceil(bottom) if np.ceil(bottom) < max_height else max_height
            return round(left), round(top), round(right), round(bottom)

    def convert_to_pixel(self):
        self.x *= self.image_shape[1]
        self.width *= self.image_shape[1]
        self.y *= self.image_shape[0]
        self.height *= self.image_shape[0]

    def convert_to_relative(self):
        self.x /= self.image_shape[1]
        self.width /= self.image_shape[1]
        self.y /= self.image_shape[0]
        self.height /= self.image_shape[0]


class InferenceCharOutput(InferenceOutput):
    flaw_name_to_id = {"Debris": 0,
                        "FBBPF": 1,
                        "BM_FBBPF": 2,
                        "Axial_Scrape": 3,
                        "Circ_Scrape": 4,
                        "ID_scratch": 5,
                        "CC": 6,
                        "Others": 7}
    flaw_id_to_name = dict(zip(flaw_name_to_id.values(), flaw_name_to_id.keys()))

    def __init__(self, classification, x, y, width, height, confidence, frame_number, image_shape):
        super().__init__(classification, x, y, width, height, confidence, frame_number, image_shape)
        self.axial_start = None
        self.axial_end = None
        self.rotary_start = None
        self.rotary_end = None
        self.flaw_name = self.flaw_id_to_name[classification]

    def get_real_positions(self, bscan_axis_info: bssd.BScanAxis,true_height=0, width_buffer = 0.1, length_buffer = 0.3):
        rotary_start_i, axial_start_i, rotary_end_i, axial_end_i = self.as_bounds(true_height=true_height)
        self.rotary_start = bscan_axis_info.rotary_pos[rotary_start_i]
        self.axial_start = bscan_axis_info.axial_pos[axial_start_i]
        self.rotary_end = bscan_axis_info.rotary_pos[rotary_end_i] + width_buffer
        self.axial_end = bscan_axis_info.axial_pos[axial_end_i]+ length_buffer
        return self.rotary_start, self.axial_start, self.rotary_end, self.axial_end, self.flaw_id_to_name[self.classification], self.confidence, [axial_start_i,axial_end_i]


    def get_flaw_name(self):
        return self.flaw_id_to_name[self.classification]


def get_available_gpus():
    '''
    Helper function that outputs number of GPU
    Returns:
        names of gpus
    '''
    local_device_protos = device_lib.list_local_devices()
    return [x.name for x in local_device_protos if x.device_type == 'GPU']


def convert_xyxy(x):
    """
    Converts yolo-style (x_mid, y_mid, width, height) to a (x_min, y_min, x_max, y_max) bounding box
    Input:
        - x: list of bbox predictions in xywh format
    Output:
        - z: list of bbox predictions in xyxy format
    """

    z = x.copy()
    z[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
    z[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
    z[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
    z[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
    return z


def non_max_suppression(pred, confidence_threshold, iou_threshold, max_wh, resized_width, resized_length, padding_value, ratio):
    """
    This function will filter out unwanted bounding boxes, including those with low confidence score and those that are overlapping given an iou_threshold.

    Input:
        - pred: predictions returned from model inferencing
        - confidence_threshold: Float value; confidence threshold used to filter out any low confidence bboxes.
        - iou_threshold: Float value; iou threshold used for merging bounding boxes.
        - max_wh: Int value; min and max width and height.
        - resized_width: resized width of the image
        - resized_length: resized length of the image
        - padding_value: added padding pixels either vertically or horizontally
        - ratio: ratio for the original image to prevent distorting when resizing
    Output:
        - out_numpy: numpy array of predictions after NonMaxSupression
    """
    pred_over_conf = pred[
                         ..., 4] > confidence_threshold  # number of predicted bboxes on specified threshold (Boolean values)
    pred_filter = pred[0]

    # this will filter out any prediction under the specified confidence threshold
    pred_filter = pred_filter[pred_over_conf[0]]

    # Prepare an output variable
    output = np.zeros((0, 6))

    # Get confidence score
    # Scale classification probabilities by the score, using object confidence multiplied by class confidence
    pred_filter[:, 5:] *= pred_filter[:, 4:5]
    # Obtain bbox in xyxy format after converting from xywh
    bbox = convert_xyxy(pred_filter[:, :4])
    # Get confidence score as conf; and class as c

    # conf, c = pred_filter[:, 5:].max(1, keepdim=True)
    conf = np.amax(pred_filter[:, 5:], axis=1, keepdims=True)
    c = np.argmax(pred_filter[:, 5:], axis=1)
    c = np.expand_dims(c, axis=1)

    pred_filter = np.concatenate((bbox, conf, c), 1)[conf.flatten() > confidence_threshold]

    boxes = pred_filter[:, :4] + (pred_filter[:, 5:6] * max_wh)
    scores = pred_filter[:, 4]

    # Conduct NMS
    i = nms_cpu(boxes, scores, iou_threshold)

    # put results after NMS in output
    # Putting information in prepared output for possible usage of CPU or GPU
    output = pred_filter[i]
    # Turn output tensor to a numpy array
    # out_numpy = output[0].numpy()

    # Shift bboxes so their positions are with respect to the original B-scans rather than the padded images
    if resized_width < resized_length:
        output[:, [0, 2]] -= padding_value.get('l_pad'), padding_value.get('r_pad')
    elif resized_length < resized_width:
        output[:, [1, 3]] -= padding_value.get('t_pad'), padding_value.get('b_pad')
    else:
        pass

    # Resize bboxes to original B-scan size. e.g. 640 -> 3600
    output[:, :4] /= ratio
    return output

def nms_cpu(boxes, scores, overlap_threshold=0.5, min_mode=False):
    x1 = boxes[:, 0]
    y1 = boxes[:, 1]
    x2 = boxes[:, 2]
    y2 = boxes[:, 3]

    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    order = scores.argsort()[::-1]

    keep = []
    while order.size > 0:
        keep.append(order[0])
        xx1 = np.maximum(x1[order[0]], x1[order[1:]])
        yy1 = np.maximum(y1[order[0]], y1[order[1:]])
        xx2 = np.minimum(x2[order[0]], x2[order[1:]])
        yy2 = np.minimum(y2[order[0]], y2[order[1:]])

        w = np.maximum(0.0, xx2 - xx1 + 1)
        h = np.maximum(0.0, yy2 - yy1 + 1)
        inter = w * h

        if min_mode:
            ovr = inter / np.minimum(areas[order[0]], areas[order[1:]])
        else:
            ovr = inter / (areas[order[0]] + areas[order[1:]] - inter)

        inds = np.where(ovr <= overlap_threshold)[0]
        order = order[inds + 1]
    return keep


def save_inference_image(image, results: List[InferenceOutput], results_path: Path, frame_number, task='wave_det'):
    """
    Adds inference overlays to the image and saves it to file.

    Args:
        image: Current BScan
        results: Inference results for the BScan, as a list of InferenceOutput classes
        results_path: Path to save the image to
        frame_number: Current frame number, used to determine the filename.

    Returns:
        None
    """
    if task == 'wave_det':
        for result in results:
    
            # Get position and label
            left, top, right, bottom = result.as_bounds()
            start = (left, top)
            end = (right, bottom)
    
            # label = bssd.WaveGroup(result.classification).name
            label = result.classification
    
            text = f'{label}, {round(result.confidence * 100, 1)}%'
    
            # Choose color and label offset based on the classification
            if result.classification == 2:
                color = (0, 0, 255) # Red
                offset = 100
            elif result.classification == 1:
                color = (255, 0, 0) # Blue
                offset = 200
            elif result.classification == 0:
                color = (0, 255, 0) # Green
                offset = 300
            else:
                color = (255, 0, 255) # yellow
                offset = 400
    
            # Draw the bounding box and label
            image = cv2.rectangle(image, start, end, color, 4)
            image = cv2.putText(image, text, (left, top + offset), cv2.FONT_HERSHEY_SIMPLEX, 2, color, 4, cv2.LINE_AA)
    
        cv2.imwrite(os.path.join(results_path, f'frame_{frame_number:03d}.png'), image)

    else:
        
        for result in results:
    
            # Get position and label
            left, top, right, bottom = result.as_bounds()
            start = (left, top)
            end = (right, bottom)
    
            # label = bssd.WaveGroup(result.classification).name
            label = result.classification
    
            text = f'{label}, {round(result.confidence * 100, 1)}%'
            color = (0, 0, 255)

            # Draw the bounding box and label
            image = cv2.rectangle(image, start, end, color, 1)
            image = cv2.putText(image, text, (left, top -5), cv2.FONT_HERSHEY_SIMPLEX, 0.3, color, 1, cv2.LINE_AA)
    
        cv2.imwrite(os.path.join(results_path, f'slice_{frame_number:03d}.png'), image)


def inference(
    probe_data: np.ndarray,
    tf_model,
    autoanalysis_config
) -> List[List[InferenceOutput]]:
    """
    Runs model inferencing on the BScan images and predicts the locations of the wavegroups.

    Parameters
    ----------
    probe_data : np.ndarray
        3D numpy array of data from the BScan file.
    tf_model : object
        TensorFlow model for inference.
    autoanalysis_config : object
        InferenceConfig class of settings for inferencing.

    Returns
    -------
    List[List[InferenceOutput]]
        List of InferenceOutput classes for all detected wavegroups in each frame.
    """
    def add_padding(img, target_width, target_height):
        """Add padding to the image to reach target dimensions."""
        resized_width, resized_length = img.shape[1], img.shape[0]
        if resized_width < resized_length:
            padding = target_width - resized_width
            l_pad, r_pad = int(round(padding / 2 - 0.1)), int(round(padding / 2 + 0.1))
            t_pad, b_pad = 0, 0
        elif resized_length < resized_width:
            padding = target_height - resized_length
            l_pad, r_pad = 0, 0
            t_pad, b_pad = int(round(padding / 2 - 0.1)), int(round(padding / 2 + 0.1))
        else:
            return img, {"l_pad": 0, "r_pad": 0, "t_pad": 0, "b_pad": 0}

        color = (114, 114, 114)
        padded_img = cv2.copyMakeBorder(img, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=color)
        return padded_img, {"l_pad": l_pad, "r_pad": r_pad, "t_pad": t_pad, "b_pad": b_pad}
    def preprocess_image(img):
        """Preprocess the image for model input."""
        img = img.transpose((2, 0, 1))[::-1]
        img = np.ascontiguousarray(img)
        img = img / 255.0
        img = img[None]
        return np.transpose(img, (0, 2, 3, 1)).astype('float32')

    def normalize_predictions(pred, img_width, img_height):
        """Normalize predictions to image dimensions."""
        pred[..., 0] *= img_width
        pred[..., 1] *= img_height
        pred[..., 2] *= img_width
        pred[..., 3] *= img_height
        return pred

    def clip_coordinates(result, og_shape):
        """Clip coordinates to image boundaries."""
        result[:, 0] = np.clip(result[:, 0], 0, og_shape[1])
        result[:, 1] = np.clip(result[:, 1], 0, og_shape[0])
        result[:, 2] = np.clip(result[:, 2], 0, og_shape[1])
        result[:, 3] = np.clip(result[:, 3], 0, og_shape[0])
        return result

    def process_frame_results(result, frame_number, og_shape):
        """Process results for a single frame."""
        frame_results = []
        for r in result:
            x1, y1, x2, y2 = map(int, r[:4])
            classification = int(r[5])
            x = ((x1 + x2) / 2) / og_shape[1]
            y = ((y1 + y2) / 2) / og_shape[0]
            width = (x2 - x1) / og_shape[1]
            height = (y2 - y1) / og_shape[0]
            confidence = r[4]
            frame_result = InferenceOutput(bssd.WaveGroup(classification), x, y, width, height, confidence, frame_number, og_shape)
            frame_results.append(frame_result)
        return frame_results
    start_time = time.time()

    if autoanalysis_config.inference.device != 'gpu':
        os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

    inference_results = []

    for frames in trange(len(probe_data), ascii=True, desc='Wavegroup Detection', position=0):
        img0 = cv2.merge((probe_data[frames, :, :], probe_data[frames, :, :], probe_data[frames, :, :]))
        og_shape = img0.shape[:2]

        resize_ratio = min(autoanalysis_config.inference.img_width / og_shape[0],
                           autoanalysis_config.inference.img_height / og_shape[1])
        resize_shape = int(og_shape[1] * resize_ratio), int(og_shape[0] * resize_ratio)
        img_resized = cv2.resize(img0, resize_shape, interpolation=cv2.INTER_LINEAR)

        im, padding_value = add_padding(img_resized, autoanalysis_config.inference.img_width, autoanalysis_config.inference.img_height)

        if im.shape[0] != 640 or im.shape[1] != 640:
            im = cv2.resize(im, (640, 640), interpolation=cv2.INTER_LINEAR)

        input_img = preprocess_image(im)

        pred = tf_model(input_img, training=False).numpy()
        pred = normalize_predictions(pred, autoanalysis_config.inference.img_width, autoanalysis_config.inference.img_height)

        result = non_max_suppression(pred, autoanalysis_config.inference.confidence_threshold, 
                                     autoanalysis_config.inference.iou_threshold, autoanalysis_config.inference.max_wh,
                                     resize_shape[0], resize_shape[1], padding_value, resize_ratio)
        result = clip_coordinates(result, og_shape)

        frame_results = process_frame_results(result, frames, og_shape)

        inference_results.append(sorted(frame_results, key=lambda wavegroup: wavegroup.classification))

    end_time = time.time()
    duration = end_time - start_time

    return inference_results

def inference_char(cscan: list, tf_model, inference_config, img_size, conf, binary_det):
    """
    Runs model inferencing on the C-scan representations and predicts the locations of the indications.

    Args:
        cscan (list): List of C-scan representations to be inferred.
        tf_model: Pre-trained TensorFlow model for inferencing.
        inference_config: Configuration parameters for inferencing.
        output_path (Path): Root directory for saving the results.
        img_size (int): Size of the input image for the model.
        conf (float): Confidence threshold for predictions.
        binary_det (bool): Indicator for binary detection.
        verbose (bool, optional): Indicator for printing additional information. Defaults to False.

    Returns:
        List[InferenceCharOutput]: List of detected indications with elements (classification, x, y, width, height, confidence, frames).
    """

    inference_results = []

    desc = 'Binary Detection' if binary_det else 'Identifying & Characterizing Flaws'
    dual_logger.info(desc)
    for slices in trange(len(cscan), ascii=True, desc=desc, position=0): # loop through B-scans and inference.
        img0 = cscan[slices]
        img0 = np.float32(img0)
        og_shape = img0.shape[:2] # Store the image shape

        # Taking the min value for resizing ratio, this will be ratio for the longer side (length or width).
        resize_ratio = min(img_size / og_shape[0],
                            img_size / og_shape[1])
        resize_shape = int(og_shape[1] * resize_ratio), int(og_shape[0] * resize_ratio)  # Calculate resized shape
        resized_width = img0.shape[1]
        resized_length = img0.shape[0]
        img_resized = cv2.resize(img0, resize_shape, interpolation=cv2.INTER_LINEAR)  # Resize image

        # add padding to image depending on shape
        if resized_width < resized_length:
            # Calculate padding size
            padding = img_size - resized_width
            # adding +0.1 and -0.1 because so that the added padding won't exceed or below the resized length or width.
            l_pad, r_pad = int(round(padding / 2 - 0.1)), int(round(padding / 2 + 0.1))
            t_pad, b_pad = 0, 0
            color = (114, 114, 114)
            im = cv2.copyMakeBorder(img_resized, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=color)
        elif resized_length < resized_width:
            padding = img_size - resized_length
            l_pad, r_pad = 0, 0
            t_pad, b_pad = int(round(padding / 2 - 0.1)), int(round(padding / 2 + 0.1))
            color = (114, 114, 114)  # adding padding in grey color, being the color model was trained on
            im = cv2.copyMakeBorder(img_resized, t_pad, b_pad, l_pad, r_pad, cv2.BORDER_CONSTANT, value=color)
        else:
            l_pad, r_pad, t_pad, b_pad = 0, 0, 0, 0
            im = img_resized

        padding_value = {"l_pad": l_pad, "r_pad": r_pad, "t_pad": t_pad, "b_pad": b_pad}

        # resize again in case it is not specified image size
        if im.shape[0] != img_size or im.shape[1] != img_size:
            im = cv2.resize(im, (img_size, img_size), interpolation=cv2.INTER_LINEAR)

        # IMG preprocessing
        img = im.transpose((2, 0, 1))[::-1]  # HWC to CHW
        img = np.ascontiguousarray(img)  # convert to contiguous array
        img = img / 255.0  # Normalize to 0-1
        img = img[None]  # Extend Dimension
        input_img = np.transpose(img, (0, 2, 3, 1)).astype('float32')

        # pred variable columns is as followed: (x,y,w,h,conf,label, for column [5:] is probability for each class)
        pred = tf_model(input_img, training=False).numpy()  # Inference
        # Normalize back bbox result to according image size 
        pred[..., 0] *= img_size
        pred[..., 1] *= img_size
        pred[..., 2] *= img_size
        pred[..., 3] *= img_size

        # get the prediction results
        result = non_max_suppression(pred, conf, inference_config.nms_iou, inference_config.max_img_size, resized_width, resized_length, padding_value, resize_ratio)

        frame_results = []

        if result is not None or len(result) != 0:

            # frame_number = frames
            img0 = cv2.merge((cscan[slices][:, :, 0], cscan[slices][:, :, 1], cscan[slices][:, :, 2]))
            for r in result:
                # grab bbox coordinates
                x1 = r[0]
                y1 = r[1]
                x2 = r[2]
                y2 = r[3]

                # convert to yolo format
                classification = int(r[5])
                x = ((x1 + x2) / 2) / og_shape[1]
                y = ((y1 + y2) / 2) / og_shape[0]
                width = (x2 - x1) / og_shape[1]
                height = (y2 - y1) / og_shape[0]
                confidence = r[4]

                # Build up frame results
                frame_result = InferenceCharOutput(classification, x, y, width, height, confidence, slices, og_shape)
                frame_results.append(frame_result)
                inference_results.append(frame_result)

    return inference_results

