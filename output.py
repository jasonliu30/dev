import numpy as np
import pandas as pd
from utils.datamap_utils import FlawLocation


class Label:
    """
    A class to represent a flaw label with its associated parameters.
    """
    def __init__(self, flaw_type, frame_start, frame_end, rotary_start, rotary_end):
        """
        Initializes the Label with the given parameters.
        
        Parameters:
            flaw_type (str): The type of the flaw.
            frame_start (float): The starting frame of the flaw.
            frame_end (float): The ending frame of the flaw.
            rotary_start (float): The starting rotary of the flaw.
            rotary_end (float): The ending rotary of the flaw.
        """
        self.flaw_type = flaw_type
        self.frame_start = frame_start
        self.frame_end = frame_end
        self.rotary_start = rotary_start
        self.rotary_end = rotary_end

    def get_xyxy(self):
        """
        Returns the coordinates of the flaw.

        Returns:
            list: The coordinates of the flaw in the format [rotary_start, frame_start, rotary_end, frame_end].
        """
        return [self.rotary_start, self.frame_start, self.rotary_end, self.frame_end]


class Auto_sizing_summary:
    """
    A class to represent the summary of auto-sizing a flaw.
    """
    def __init__(self, flaw_id: int, fn, header, pred_coords, depth, comments, is_daq_format=False):
        """
        Initializes the Auto_sizing_summary with the given parameters.
        
        Parameters:
            flaw_id (int): The ID of the flaw.
            fn (str): The filename of the scan.
            header (Header): The header information of the scan.
            pred_coords (np.array): The predicted coordinates of the flaw.
            depth (float): The depth of the flaw.
            comments (str): Any additional comments.
        """
        self.scan_name = fn
        self.scan_unit = header.UnitNumber
        self.scan_station = str(header.GeneratingStation).split(".")[1]
        self.scan_channel = header.ChannelLetter + str(header.ChannelNumber)
        self.scan_axial_pitch = header.AxialPitch if is_daq_format else header.AxialPitch/10
        self.flaw_id = flaw_id + 1
        self.flaw_type = pred_coords[:, 4][flaw_id]
        self.confidence = pred_coords[:, 5][flaw_id]
        self.flaw_axial_start = pred_coords[:, 1][flaw_id]
        self.flaw_length = pred_coords[:,3][flaw_id] - self.flaw_axial_start
        self.flaw_rotary_start = pred_coords[:, 0][flaw_id]
        # if rotary start begins before 0 degree, change width calculation
        self.flaw_width = self.calculate_flaw_width(pred_coords, flaw_id)
        self.frame_start = pred_coords[:, 6][flaw_id][0]
        self.frame_end = pred_coords[:, 6][flaw_id][1]
        self.probe_depth = None
        self.flaw_depth = depth[flaw_id] if depth != None else "N/A"
        self.flaw_feature_amp = None
        self.flaw_max_amp = None
        self.chatter_amplitude = None
        self.flag_high_error = None
        self.note_1 = comments[flaw_id]
        self.note_2 = "N/A"
        self.possible_category = pred_coords[:, 7][flaw_id]
    @staticmethod
    def calculate_flaw_width(pred_coords, flaw_id):
        """
        Calculates the width of the flaw.

        Parameters:
            pred_coords (np.array): The predicted coordinates of the flaw.
            flaw_id (int): The ID of the flaw.

        Returns:
            float: The calculated width of the flaw.
        """
        if pred_coords[:, 0][flaw_id] > pred_coords[:, 2][flaw_id]:
            return 360 - pred_coords[:, 0][flaw_id] + pred_coords[:, 2][flaw_id]
        else:
            return pred_coords[:, 2][flaw_id] - pred_coords[:, 0][flaw_id]
        
    def get_df(self):
        return pd.DataFrame.from_dict(self.__dict__, orient='index').T

    def get_FlawLocation(self, prediction):
        """
        Generates a list of FlawLocation objects from the given prediction.

        Parameters:
            prediction (np.array): The flaw prediction.

        Returns:
            list: A list of FlawLocation objects.
        """
        label = f"{self.flaw_type} {self.confidence:.1%}"
        flaws = [[label, prediction[1], prediction[3] + 1, prediction[0], prediction[2], 'D']]
        return [FlawLocation(*flaw) for flaw in flaws]

    def filter_confidence(self, conf_config):
        """
        Filters the flaw based on its confidence value.

        Parameters:
            conf_config (Config): The confidence configuration.

        Returns:
            Auto_sizing_summary: The filtered Auto_sizing_summary object.
            None: If the confidence is lower than the threshold.
        """
        confidence_thresholds = {
            "Debris": conf_config.debris,
            "CC": conf_config.CC,
            "FBBPF": conf_config.FBBPF,
            "Axial_Scrape": conf_config.scrape,
            "Circ_Scrape": conf_config.scrape,
            "ID_scratch": conf_config.scratch,
            "Note 1": conf_config.debris,
            "Note 1 - overlapped": conf_config.debris,
            "other": conf_config.other
        }
        
        confidence = next((v for k, v in confidence_thresholds.items() if k in self.flaw_type), None)
        return self if confidence is None or self.confidence >= confidence else None
    


