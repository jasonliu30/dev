from enum import IntEnum
import numpy as np

class Flaw_Types(IntEnum):
    Debris = 0
    FBBPF = 1
    BM_FBBPF = 2
    Axial_Scrape = 3
    Circ_Scrape = 4
    ID_scratch = 5
    CC = 6


class Flaw_Information:
    def __init__(self, type: str, axial_start_mm: float, length_mm: float, rotary_start_deg: float, width_deg: float, depth_mm: float, flaw_location: np.ndarray):
        self.type = type
        self.axial_start_mm = axial_start_mm
        self.length_mm = length_mm
        self.rotary_start_deg = rotary_start_deg
        self.width_deg = width_deg
        self.depth_mm = depth_mm
        self.flaw_location = flaw_location

