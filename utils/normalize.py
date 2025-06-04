import numpy as np
from axial_pitch_align import align


def NormalizeData(data: np.ndarray) -> np.ndarray:
    """
    Normalizes the given data to the range [0, 255].

    Parameters:
        data (np.ndarray): The input data array.

    Returns:
        np.ndarray: The normalized data array scaled to the range [0, 255].
    """
    return (data - np.min(data)) / (np.max(data) - np.min(data)) * 255


def preprocessing(result: np.ndarray, axial_pitch: float, alignment: bool = True) -> np.ndarray:
    """
    Preprocesses the given result by normalizing it and optionally aligning it.

    Parameters:
        result (np.ndarray): The input result array.
        bscan: B-scan data object.
        bscan_path (str): Path to the B-scan data file.
        alignment (bool, optional): Whether to align the result. Defaults to True.

    Returns:
        np.ndarray: The preprocessed result, normalized and optionally aligned.
    """
    normalized_result = NormalizeData(result)
    if alignment:
        aligned_cscan = align(axial_pitch, normalized_result, align_to=0.3)
    else:
        aligned_cscan = normalized_result

    return aligned_cscan