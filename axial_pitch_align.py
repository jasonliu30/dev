import cv2

# def get_resolution(bscan, bscan_path):
#     """
#     Calculate and return the resolution of the B-Scan.

#     Parameters:
#     ----------
#         bscan (object): The B-Scan object.
#         bscan_path (str): The path of the B-Scan file.

#     Returns:
#     -------
#         float: The calculated resolution of the B-Scan.
#     """
#     axial_pitch = round(bscan._read_header().AxialPitch, 1)
#     is_daq_file = bscan_path.endswith('.daq')
#     return (axial_pitch * 10 if is_daq_file else axial_pitch) / 10

def align(axial_pitch, cscan, align_to = 0.3):
    """
    Align the C-Scan to a specified resolution.

    Parameters:
    bscan (object): The B-Scan object.
    bscan_path (str): The path of the B-Scan file.
    cscan (np.array): The C-Scan image.
    align_to (float, optional): The resolution to align to. Defaults to 0.3.

    Returns:
    np.array: The aligned C-Scan image.
    """
    # obtain c-scan length and width
    w,l = cscan.shape[1], cscan.shape[0]

    # Calculate ratio and aligned image length
    ratio = axial_pitch/align_to
    aligned_l = round(l * (ratio))

    # align axial pitch
    aligned_cscan = cv2.resize(cscan,[w,aligned_l])

    return aligned_cscan