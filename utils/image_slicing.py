from empatches import EMPatches
import matplotlib.pyplot as plt
import imgviz
import numpy as np
from itertools import combinations
from inferencing_script import InferenceCharOutput
import cv2


def convert(size, box):
    """
    This function converts the absolute bounding box coordinates to relative coordinates.

    Parameters:
    -----------
    size: tuple
        The size of the image in pixels, as (width, height).
    box: tuple
        The bounding box in absolute coordinates, as (xmin, xmax, ymin, ymax).

    Returns:
    --------
    tuple
        The bounding box in relative coordinates, as (x_center, y_center, width, height).
    """
    width_ratio = 1.0 / size[0]
    height_ratio = 1.0 / size[1]

    x_center = (box[0] + box[1]) * width_ratio / 2.0
    y_center = (box[2] + box[3]) * height_ratio / 2.0

    relative_width = (box[1] - box[0]) * width_ratio
    relative_height = (box[3] - box[2]) * height_ratio

    return (x_center, y_center, relative_width, relative_height)

def resize_image(image, target_size):
    """
    This function resizes an image to a specified target size while maintaining the aspect ratio.

    Parameters:
    -----------
    image: numpy.ndarray
        The original image to resize.
    target_size: int
        The size of the larger dimension in the resized image (width or height).

    Returns:
    --------
    resized_image: numpy.ndarray
        The resized image.
    """
    # Get the current dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the scale factor while maintaining the aspect ratio
    scale_factor = target_size / max(original_height, original_width)

    # Calculate the new dimensions for the image
    new_width = int(original_width * scale_factor)
    new_height = int(original_height * scale_factor)

    # Resize the image using OpenCV
    resized_image = cv2.resize(image, (new_width, new_height))

    return resized_image

def overlaps(x: range, y: range):
    """
    Checks if two ranges overlap and returns the overlapping range.

    Parameters:
    -----------
    x: range
        The first range.
    y: range
        The second range.

    Returns:
    --------
    tuple
        A boolean indicating whether the ranges overlap,
        and a list representing the overlapping range (or an empty list if there is no overlap).
    """
    intersection = set(x).intersection(y)

    if intersection:
        flaw_loc = [min(intersection) - y.start, max(intersection) - y.start + 1]
    else:
        flaw_loc = []

    overlap = max(x.start, y.start) < min(x.stop, y.stop)

    return overlap, flaw_loc
        
def create_img_patches(img: np.ndarray, 
                       patchsize: int, 
                       overlap: float, 
                       verbose: bool = False):
    """
    Creates image patches of a specified size with a specified overlap.
    
    Parameters
    ----------
    img : np.ndarray
        The source image from which to create patches.
    patchsize : int
        The size of the patches to be created.
    overlap : float
        The amount of overlap between patches as a decimal (e.g., 0.5 for 50% overlap).
    verbose : bool, optional
        If True, visualizes the created patches. Default is False.

    Returns
    -------
    Tuple[List[np.ndarray], np.ndarray]
        The created patches and their indices.
    """
    emp = EMPatches()
    img_patches, indices = emp.extract_patches(img, patchsize=patchsize, overlap=overlap)

    if verbose and len(img_patches) > 1:  # pragma: no cover
        tiled = imgviz.tile(list(map(np.uint8, img_patches[:-1])), border=(255, 0, 0))
        plt.figure(figsize=(20, 20))
        plt.imshow(tiled)

    return img_patches, indices


def convert_label(labels,shape,verbose=True):
    
    dh, dw, _ = shape
    
    xyxy = []
    
    for l in labels:

        if len(l) != 0:
            _, x, y, w, h = map(float, l.split(' ')[:5])
            
            l = round((x - w / 2) * dw)
            r = round((x + w / 2) * dw)
            t = round((y - h / 2) * dh)
            b = round((y + h / 2) * dh)
            
            if l < 0:
                l = 0
            if r > dw - 1:
                r = dw - 1
            if t < 0:
                t = 0
            if b > dh - 1:
                b = dh - 1
            
            # img_overlay = copy.deepcopy(img)
            # img_overlay = cv2.rectangle(img, (l, t), (r, b), (0, 0, 255), 1)
            
            xyxy.append(((l,t,r,b,_)))
    
    # plt.Figure(figsize=(25,25))
    # plt.imshow(img_overlay)

    return xyxy


def create_patch_package(indices,xyxy_all,img_patches,patchsize):
    
    patches = []
    shape = (img_patches[0].shape[1],img_patches[0].shape[0])
    for index,patch in enumerate(indices):
        flaw_loc_all = ''
        
        for xyxy in xyxy_all:
            
            # overlap_bool = do_overlap(Point(xyxy[0],xyxy[1]),Point(xyxy[2],xyxy[3]), Point(patch[2],patch[0]), Point(patch[3],patch[1]))
            overlap_bool_circ, flaw_loc_circ = overlaps(range(xyxy[0],xyxy[2]), range(patch[2],patch[3]))
            overlap_bool_axial, flaw_loc_axial = overlaps(range(xyxy[1],xyxy[3]), range(patch[0],patch[1]))
            
            
            
            if overlap_bool_circ and overlap_bool_axial:
                box = [flaw_loc_circ[0],flaw_loc_circ[1],flaw_loc_axial[0],flaw_loc_axial[1]]
                bb = convert(shape, box)
                flaw_loc = str(int(xyxy[4])) + " " + " ".join([("%.6f" % a) for a in bb]) + '\n'
                # flaw_loc = [xyxy[4],flaw_loc_axial[0],flaw_loc_axial[1],flaw_loc_circ[0],flaw_loc_circ[1]]
                flaw_loc_all = flaw_loc_all + flaw_loc
            else:
                flaw_loc = ''
            
        patches.append([str(index+1),img_patches[index],flaw_loc_all])
    
    return patches


# def patch_wrapper(img,folder_name:str,fname:str,patchsize:int,overlap:float):
    

#     label_path = os.path.join(root,fname+'.txt')
    
#     fl = open(label_path, 'r')
#     labels = fl.readlines()
#     fl.close()
    
#     xyxy_all = convert_label(labels,img.shape)
        
#     img_patches, indices = create_img_patches(img,patchsize,overlap)
    
#     patches = create_patch_package(indices,xyxy_all,img_patches,patchsize)

#     return patches


def calc_flaw_locations(prediction_lst, indices, img_size):

    flaw_locations = []
    for flaw in prediction_lst:
        index = flaw.frame_number
        x_start = indices[index][2]
        y_start = indices[index][0]
        flaw.convert_to_pixel()
        flaw.x += x_start
        flaw.y += y_start
        flaw.image_shape = img_size
        flaw.convert_to_relative()
        flaw_locations.append(flaw)

    return flaw_locations


def get_iou(pred_box, gt_box):
    """
    pred_box : the coordinate for predict bounding box
    gt_box :   the coordinate for ground truth bounding box
    return :   the iou score
    the  left-down coordinate of  pred_box:(pred_box[0], pred_box[1])
    the  right-up coordinate of  pred_box:(pred_box[2], pred_box[3])
    """
    # 1.get the coordinate of inters
    ixmin = max(pred_box[0], gt_box[0])
    ixmax = min(pred_box[2], gt_box[2])
    iymin = max(pred_box[1], gt_box[1])
    iymax = min(pred_box[3], gt_box[3])

    iw = np.maximum(ixmax-ixmin+1., 0.)
    ih = np.maximum(iymax-iymin+1., 0.)

    # 2. calculate the area of inters
    inters = iw*ih

    # 3. calculate the area of union
    uni = ((pred_box[2]-pred_box[0]+1.) * (pred_box[3]-pred_box[1]+1.) +
           (gt_box[2] - gt_box[0] + 1.) * (gt_box[3] - gt_box[1] + 1.) -
           inters)
    
    area_left = (pred_box[2]-pred_box[0]+1.)*(pred_box[3]-pred_box[1]+1.)
    area_right = (gt_box[2]-gt_box[0]+1.)*(gt_box[3]-gt_box[1]+1.)
    # 4. calculate the overlaps between pred_box and gt_box
    iou = inters / uni
    
    # 5. calculate the overlaps of iou_right, and iou_left
    iou_left = inters / area_left
    iou_right = inters / area_right

    return iou, iou_left, iou_right


def filter_empty_item(lst):
    return list(filter(None,lst))


def flatten_lst(lst):
    return [item for sublist in lst  for item in sublist]


def merge_bboxes(all_labels: list[InferenceCharOutput], iou_threshold, iou_thresh_alt):
    # get all the bbox index combinations per frame
    pairs = list(combinations(range(len(all_labels)), 2))

    for a, b in pairs:  # go through every combination of bboxes per frame

        # Calculate IOUs
        iou, iou_left, iou_right = get_iou(all_labels[a].as_bounds(), all_labels[b].as_bounds())

        # if either iou_left or right is over 0.9 iou, merge
        if iou_left >= iou_thresh_alt or iou_right >= iou_thresh_alt:
            merge_bool = True
        else:
            merge_bool = False

        if iou >= iou_threshold or merge_bool:

            # Calculate new coordinates
            x1_a, y1_a, x2_a, y2_a = all_labels[a].as_bounds(normalize=True)
            x1_b, y1_b, x2_b, y2_b = all_labels[b].as_bounds(normalize=True)
            x1, x2 = min(x1_a, x2_a, x1_b, x2_b), max(x1_a, x2_a, x1_b, x2_b)
            y1, y2 = min(y1_a, y2_a, y1_b, y2_b), max(y1_a, y2_a, y1_b, y2_b)
            x_mid = (x2 + x1) / 2
            y_mid = (y2 + y1) / 2
            width = (x2 - x1)
            height = (y2 - y1)
            conf_a = all_labels[a].confidence
            conf_b = all_labels[b].confidence
            conf = max(conf_a, conf_b)

            # if it is overlapping boxes with difference classifications, don't merge
            if all_labels[a].classification == all_labels[b].classification:
                # Choose the label with higher confidence, and discard the other
                if conf_a > conf_b:
                    all_labels[a].x = x_mid
                    all_labels[a].y = y_mid
                    all_labels[a].width = width
                    all_labels[a].height = height
                    all_labels.pop(b)

                else:
                    all_labels[b].x = x_mid
                    all_labels[b].y = y_mid
                    all_labels[b].width = width
                    all_labels[b].height = height
                    all_labels.pop(a)

                # Recursively call this function until no boxes overlap
                return merge_bboxes(all_labels, iou_threshold, iou_thresh_alt)

    return all_labels