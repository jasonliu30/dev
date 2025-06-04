from typing import List, Optional, Dict, Any
import numpy as np

VERSION = 'v1.2.0'

class Flaw:
    """
    Represents a flaw detected in a scan.

    Attributes:
    -----------
    Various attributes representing flaw properties (e.g., width_buffer, length_buffer, ind_num, etc.)
    """

    def __init__(self, prediction: Optional[Any] = None, index: Optional[int] = None, 
                 scan: Optional[Any] = None, config: Optional[Any] = None):
        """
        Initialize a Flaw object.

        Parameters
        ----------
        prediction : Optional[Any]
            Prediction object containing flaw information.
        index : Optional[int]
            Index of the flaw.
        scan : Optional[Any]
            Scan object containing scan information.
        config : Optional[Any]
            Configuration object.
        """
        self.width_buffer = config.characterization.inference.width_buffer if config else None
        self.length_buffer = config.characterization.inference.length_buffer if config else None
        
        if all(arg is not None for arg in (prediction, index, scan, config)):
            self.from_prediction(prediction, index, scan, config)
        self.flaw_id: Optional[int] = None
        self.flaw_type: Optional[str] = None
        self.ind_num: Optional[str] = None
        self.possible_category: List[str] = []
        self.note1: Optional[str] = None
        self.note2: Optional[str] = None
        self.depth: Optional[float] = None
        self.probe_depth: Optional[float] = None
        self.depth_nb: Optional[float] = None
        self.depth_pc: Optional[float] = None
        self.depth_nb1: Optional[float] = None
        self.depth_nb2: Optional[float] = None
        self.depth_apc: Optional[float] = None
        self.depth_cpc: Optional[float] = None
        self.probe_used: Optional[str] = None
        self.feature_amp: Optional[float] = None
        self.max_amp: Optional[float] = None
        self.stats: Optional[Any] = None,
        self.chatter_amplitude: Optional[float] = None
        self.depth_array: Optional[np.ndarray] = None
        self.max_depth_info: Optional[Dict[str, Any]] = None
        self.is_reported: bool = False
        self.is_flagged: bool = False
        self.version: str = VERSION

    def __repr__(self) -> str:
        """
        Return a string representation of the Flaw object.

        Returns
        -------
        str
            A detailed string representation of the Flaw object.
        """
        return (f"type={self.flaw_type}, "
                f"axial_start={self.axial_start}, rotary_start={self.rotary_start}, "
                f"length={self.length}, width={self.width}, depth={self.depth}, "
                f"confidence={self.confidence:.2f})")

    def __str__(self) -> str:
        """
        Return a human-readable string representation of the Flaw object.

        Returns
        -------
        str
            A concise string representation of the Flaw object.
        """
        return f"Flaw {self.flaw_id}: {self.flaw_type} {self.depth}  at ({self.axial_start}, {self.rotary_start})"
    
    @classmethod
    def from_prediction(cls, prediction: Any, index: int, scan: Any, config: Any):
        """
        Create a Flaw object from a prediction.

        Parameters
        ----------
        prediction : Any
            Prediction object containing flaw information.
        index : int
            Index of the flaw.
        scan : Any
            Scan object containing scan information.
        config : Any
            Configuration object containing buffer values.

        Returns
        -------
        Flaw
            A new Flaw object populated with prediction data.
        """
        flaw = cls()
        flaw.width_buffer = config.characterization.inference.width_buffer
        flaw.length_buffer = config.characterization.inference.length_buffer
        
        pred_coord = prediction.get_real_positions(scan.axes, true_height=scan.num_frames, 
                                                   width_buffer=flaw.width_buffer, 
                                                   length_buffer=flaw.length_buffer)
        bounds = prediction.as_bounds(true_height=scan.num_frames)
        
        flaw.flaw_id = index
        flaw.X = prediction.x
        flaw.Y = prediction.y
        flaw.width_yolo = prediction.width
        flaw.height_yolo = prediction.height
        flaw.flaw_type_id = prediction.classification
        flaw.flaw_type = prediction.flaw_name
        flaw.confidence = prediction.confidence
        flaw.axial_start = pred_coord[1]
        flaw.length = pred_coord[3] - pred_coord[1]
        flaw.rotary_start = pred_coord[0]
        if pred_coord[2] > pred_coord[0]:
            flaw.width = pred_coord[2] - pred_coord[0]
        else:
            flaw.width = (360 - pred_coord[0]) + pred_coord[2]
        flaw.x_start, flaw.y_start, flaw.x_end, flaw.y_end = bounds
        flaw.frame_start = bounds[1] + 1
        flaw.frame_end = bounds[3] + 1
        flaw.is_predicted = True
        
        return flaw

    def update_possible_category(self, new_categories: List[str]) -> None:
        """
        Update the possible categories for this flaw.

        Parameters
        ----------
        new_categories : List[str]
            List of new possible categories.
        """
        self.possible_category = new_categories

    @staticmethod
    def convert_xyxy(x: np.ndarray) -> np.ndarray:
        """
        Converts yolo-style (x_mid, y_mid, width, height) to a (x_min, y_min, x_max, y_max) bounding box.

        Parameters
        ----------
        x : np.ndarray
            Array of bbox predictions in xywh format.

        Returns
        -------
        np.ndarray
            Array of bbox predictions in xyxy format.
        """
        z = x.copy()
        z[:, 0] = x[:, 0] - x[:, 2] / 2  # x1
        z[:, 1] = x[:, 1] - x[:, 3] / 2  # y1
        z[:, 2] = x[:, 0] + x[:, 2] / 2  # x2
        z[:, 3] = x[:, 1] + x[:, 3] / 2  # y2
        return z

    def update_depth(self, new_depth: float) -> None:
        """
        Update the depth of the flaw.

        Parameters
        ----------
        new_depth : float
            New depth value.
        """
        self.depth = new_depth

    def to_dict(self) -> Dict[str, Any]:
        """
        Convert the Flaw object to a dictionary.

        Returns
        -------
        Dict[str, Any]
            Dictionary representation of the Flaw object.
        """
        return self.__dict__

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Flaw':
        """
        Create a Flaw object from a dictionary.

        Parameters
        ----------
        data : Dict[str, Any]
            Dictionary containing flaw data.

        Returns
        -------
        Flaw
            A new Flaw object populated with the dictionary data.
        """
        flaw = cls()
        for key, value in data.items():
            setattr(flaw, key, value)
        return flaw