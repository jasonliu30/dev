import pandas as pd
import numpy as np
import os
from b_scan_reader import bscan_structure_definitions as bssd
from b_scan_reader.bscan_structure_definitions import ROI
from b_scan_reader.BScan_reader import load_bscan
from typing import List, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, field
from flaw_data_structure import Flaw
from utils import results_saving
import itertools
from collections import namedtuple
from typing import Dict, List, Callable
from collections import defaultdict

FlawLocation = namedtuple('FlawLocation', ['name', 'frame_start', 'frame_end', 'rotary_start', 'rotary_end', 'dr'])

@dataclass
class Scan:
    bscan_path: Path
    local_path: Path
    bscan: object
    bscan_d: object
    scan_name: str
    scan_unit: str
    scan_station: str
    scan_channel: str
    year: int
    month: int
    day: int
    outage_number: str
    scan_axial_pitch: float
    first_axial: int
    last_axial: int
    first_rotary: int
    last_rotary: int
    val_train: str = None
    axes: Optional[object] = None
    num_frames: Optional[int] = None
    num_rotary: Optional[int] = None
    is_daq: bool = field(init=False)
    roi: Optional[object] = None
    save_name: Optional[str] = None
    cscan: Optional[object] = None
    probe_names: List[str] = field(default_factory=lambda: ['APC', 'CPC', 'NB1', 'NB2'])
    flaws_plot: List[List[FlawLocation]] = field(default_factory=list)
    flaws: List[Flaw] = field(default_factory=list)
    reported_flaws: List[Flaw] = field(default_factory=list)

    def __getstate__(self):
        """Return a picklable state of the object."""
        state = self.__dict__.copy()
        # Remove the non-picklable entries
        state['bscan'] = None
        state['bscan_d'] = None
       
        return state

    def __post_init__(self):
        self.is_daq = self.is_daq_file(self.bscan_path)
        if self.save_name is None:
            self.save_name = self.scan_name
    @classmethod
    def from_bscan_path(cls, bscan_path: str):
        bscan_path = Path(bscan_path)
        bscan_path_d = bscan_path.with_name(bscan_path.name.replace('Type A', 'Type D'))
        bscan_a = load_bscan(str(bscan_path))
        bscan_d = load_bscan(str(bscan_path_d))
        header = cls.get_header(bscan_a)
        
        return cls(
            bscan_path=bscan_path,
            local_path=bscan_path,
            bscan=bscan_a,
            bscan_d=bscan_d,
            scan_name=bscan_path.stem,
            scan_unit=header.UnitNumber,
            scan_station=header.GeneratingStation.name,
            year = int(header.Year),
            month = int(header.Month),
            day = int(header.Day),
            outage_number = f"{header.GeneratingStation.name[0]}{header.Year}{header.UnitNumber}1",
            scan_channel=f"{header.ChannelLetter}{header.ChannelNumber}",
            scan_axial_pitch=cls.get_resolution(bscan_a, str(bscan_path)),
            first_axial=int(header.FirstAxialPosition),
            last_axial=int(header.LastAxialPosition),
            first_rotary=int(header.FirstRotaryPosition),
            last_rotary=int(header.LastRotaryPosition)
        )
           
    def add_flaw(self, flaw: Flaw, reported: bool = False):
        if reported:
            self.reported_flaws.append(flaw)
        else:
            self.flaws.append(flaw)

    def remove_flaw(self, flaw: Flaw):
        self.flaws.remove(flaw)

    def generate_ind_num(self):
        # Initialize counters for each flaw type
        counters = {'Indication': 0, 'FBBPF': 0, 'BM_FBBPF': 0}
        prefixes = {'Indication': 'Ind', 'FBBPF': 'MP', 'BM_FBBPF': 'BM'}

        # Iterate over each flaw in the scan
        for flaw in self.flaws:
            flaw_type = flaw.flaw_type

            # Increment the counter for the flaw type
            if flaw_type in counters:
                counters[flaw_type] += 1
                flaw.ind_num = f"{prefixes[flaw_type]} {counters[flaw_type]}"
            else:
                counters['Indication'] += 1
                flaw.ind_num = f"Ind {counters['Indication']}"

    def categorize_flaws(self) -> Dict[str, List['Flaw']]:
        """
        Categorize flaws based on their type and required depth calculation method.
        A flaw can be categorized into multiple categories if it has multiple possible categories.

        Returns
        -------
        Dict[str, List[Flaw]]
            A dictionary of categorized flaws, where keys are category names and values are lists of Flaw objects.
        """
        categories = {
            'Debris': ['nb_debris', 'pc_cc'],
            'FBBPF': ['nb_fbbpf', 'pc_other'],
            'BM_FBBPF': ['nb_fbbpf', 'pc_other'],
            'CC': ['pc_cc', 'nb_debris'],
        }

        flaw_categories = defaultdict(list)

        for flaw in self.flaws:
            categorized = False
            for category, assignments in categories.items():
                if category in flaw.possible_category:
                    for assignment in assignments:
                        flaw_categories[assignment].append(flaw)
                    categorized = True
            
            if not categorized:
                flaw_categories['nb_other'].append(flaw)
                flaw_categories['pc_other'].append(flaw)

        return dict(flaw_categories)
    
    def get_predicted_flaw_types(self):
        flaw_types = set(flaw.flaw_type for flaw in self.flaws)
        possible_cats = set(cat for flaw in self.flaws 
                            for cat in flaw.possible_category)
        return flaw_types | possible_cats

    def get_plot(self):
        image_width, image_height = self.cscan.shape[1], self.cscan.shape[0]
        for flaw in self.flaws:
            # Handle case where flaw.depth might be None or a string
            if flaw.depth is not None:
                try:
                    # Try to convert to float and format
                    depth_value = float(flaw.depth)
                    label = f"{flaw.flaw_type} {depth_value:.2f} mm"
                except (ValueError, TypeError):
                    # If conversion fails, use the original value as is
                    label = f"{flaw.flaw_type} {flaw.depth} mm"
            else:
                label = f"{flaw.flaw_type}"
            # xyxy = convert_xyxy(np.array([[flaw.X, flaw.Y, flaw.width_yolo, flaw.height_yolo]]))[0]
            frame_start = flaw.frame_start -1
            frame_end = flaw.frame_end -1
            rotary_start = int(flaw.rotary_start*10 - self.first_rotary)
            rotary_end = int(flaw.width*10 + rotary_start)
            flaws = [[label, 
                      frame_start, 
                      frame_end, 
                      rotary_start, 
                      rotary_end, 'D']]
            self.flaws_plot.append([FlawLocation(*flaw) for flaw in flaws])

    def plot_flaws(self, adjusted_plot: Optional[List[Tuple]] = None):
        self.get_plot()
        if adjusted_plot:
            self.flaws_plot += adjusted_plot
        fig = results_saving.plot_c_scan(np.uint8(self.cscan), [item[0] for item in self.flaws_plot], self.axes, 
                                        title=self.scan_name,flaw_ids = self.get_flaw_ids())
        return fig
    
    def get_flaw_ids(self):
        return [flaw.ind_num for flaw in self.flaws]
    
    def get_flaws_by_type(self, flaw_type: str) -> List[Flaw]:
        return [flaw for flaw in self.flaws if flaw.flaw_type == flaw_type]

    def get_flaws_by_confidence(self, confidence_threshold: float) -> List[Flaw]:
        return [flaw for flaw in self.flaws if flaw.confidence >= confidence_threshold]
    
    def delete_flaws_by_confidence(self, confidence_threshold: float) -> None:
        self.flaws = [flaw for flaw in self.flaws if flaw.confidence >= confidence_threshold]
    
    @staticmethod
    def is_daq_file(bscan_path: Path) -> bool:
        return bscan_path.suffix.lower() == '.daq'
    
    @staticmethod
    def get_header(bscan):
        return bscan._read_header()
    
    @staticmethod
    def get_resolution(bscan, bscan_path: str) -> float:
        """
        Calculate and return the resolution of the B-Scan.

        Parameters
        ----------
        bscan : object
            The B-Scan object.
        bscan_path : str
            The path of the B-Scan file.

        Returns
        -------
        float
            The calculated resolution of the B-Scan.
        """
        axial_pitch = round(bscan._read_header().AxialPitch, 1)
        is_daq_file = bscan_path.endswith('.daq')
        return (axial_pitch * 10 if is_daq_file else axial_pitch) / 10
    
    def get_bscan_data(self) -> Tuple:
        """
        Get the B-scan data and axes for the given channel.

        Parameters
        ----------
        channel : str
            The channel to get the data for.

        Returns
        -------
        Tuple
            A tuple containing the B-scan data and axes.
        """
        
        probes = [bssd.Probe['APC'], bssd.Probe['CPC'], bssd.Probe['NB1'], bssd.Probe['NB2']]
        probes_data = {probe.name: None for probe in probes}
        for probe in probes:
            if probe.name == 'NB2':
                probe_data = self.bscan_d.read_channel_data_subset(probe, self.roi) if self.roi else self.bscan_d.read_channel_data(probe)
            else:
                probe_data = self.bscan.read_channel_data_subset(probe, self.roi) if self.roi else self.bscan.read_channel_data(probe)
            if self.is_daq:
                probe_data.data = daq_to_anf(probe_data.data, probe.name)
                probe_data.probe = probe
            elif not self.is_daq and self.roi:
                probe_data.probe = probe
            
            probes_data[probe.name] = probe_data
       
        
        return probes_data
    
    def validate_axial_range(self, axial_range: Optional[List[int]]) -> None:
        """
        Validate the given axial range.

        Parameters
        ----------
        axial_range : Optional[List[int]]
            The axial range to validate

        Raises
        ------
        ValueError
            If the axial range is invalid or if the B-scan file exceeds 225 mm without a specified range
        """
        
        
        if len(axial_range) < 2 or not (self.first_axial <= axial_range[0] and self.last_axial >= axial_range[1]):
            raise ValueError(f'Axial range {axial_range} is not within the B-scan range {self.first_axial} - {self.last_axial}, please specify a valid axial range')
        elif (self.last_axial - self.first_axial) > 225:
            raise ValueError('B-scan file exceeds 225 mm. Specify a smaller axial range.')
    
    def update_axial_range(self, axial_range: Optional[List[int]]) -> None:
        """
        Update the axial range of the B-scan and the save name.

        Parameters
        ----------
        axial_range : Optional[List[int]]
            The axial range to update. If None, keeps the current range.

        Raises
        ------
        ValueError
            If the axial range is invalid
        """
        if axial_range is not None:
            self.validate_axial_range(axial_range)
            self.first_axial, self.last_axial = map(int, axial_range[:2])
            self.save_name = f'{self.scan_name}_A{self.first_axial}-{self.last_axial}'
        else:
            self.save_name = self.scan_name

        self.update_bscan_axes(axial_range)

    def update_bscan_axes(self, axial_range: Optional[List[int]] = None) -> Tuple:
        """
        Update the bscan axes based on the given axial range.

        Parameters
        ----------
        axial_range : Optional[List[int]], optional
            The axial range to use for ROI, by default None

        """
        roi = None
        if axial_range:
            self.validate_axial_range(axial_range)
            roi = ROI(axial_start_mm=int(axial_range[0]), axial_end_mm=int(axial_range[1]))
        
        if roi:
            self.axes = self.bscan.get_channel_axes(bssd.Probe['APC'], roi=roi)
            self.roi = roi
        else:
            self.axes = self.bscan.get_channel_axes(bssd.Probe['APC'])

        self.num_frames = len(self.axes.axial_pos)
        self.num_rotary = len(self.axes.rotary_pos)

    def __repr__(self):
        return f"Scan(name={self.scan_name}, unit={self.scan_unit}, flaws={len(self.flaws)})"

    def __str__(self):
        return f"Scan {self.scan_name} from unit {self.scan_unit} with {len(self.flaws)} flaws"
    
    def to_dataframe(self):
        flaw_dicts = [flaw.to_dict() for flaw in self.flaws]
        df = pd.DataFrame(flaw_dicts)
        scan_info = {
            'scan_name': self.scan_name,
            'scan_unit': self.scan_unit,
            'scan_station': self.scan_station,
            'scan_channel': self.scan_channel,
            'scan_axial_pitch': self.scan_axial_pitch
        }
        for key, value in scan_info.items():
            df[key] = value
        return df

    @classmethod
    def from_dataframe(cls, df):
        scan = cls(
            df['scan_name'].iloc[0],
            df['scan_unit'].iloc[0],
            df['scan_station'].iloc[0],
            df['scan_channel'].iloc[0],
            df['scan_axial_pitch'].iloc[0]
        )
        for _, row in df.iterrows():
            flaw = Flaw.from_dict(row.to_dict())
            scan.add_flaw(flaw)
        return scan
    
    @staticmethod
    def closest_to_0_or_360(rotary_position: float) -> int:
        return 0 if rotary_position % 360 < 180 else 360

    def merge_flaws_cross_0(self, config):
        """
        Merges flaws that cross the 0/360 degree boundary.
        
        This method identifies flaws that appear at both ends of the rotary scan
        (near 0 and 360 degrees) and merges them into a single flaw record based on 
        their proximity in axial position and depth characteristics.
        
        Parameters:
        ----------
        config : object
            Configuration object containing thresholds for merging.
        """
        # Check if there are enough flaws to potentially merge
        if len(self.flaws) <= 1:
            return
        
        rotary_threshold = config.characterization.flaw_merging.rotary_threshold_to_0
        has_flaws_near_boundary = any(
            ((flaw.rotary_start <= rotary_threshold) or 
            (flaw.rotary_start + flaw.width >= 360 - rotary_threshold)) and
            ("Scrape" in flaw.flaw_type)
            for flaw in self.flaws
        )
        
        # Only proceed if there are flaws near the boundary
        if not has_flaws_near_boundary:
            return
            
        # Convert flaws to DataFrame format
        df = self.to_dataframe()
        
        # Apply the merging algorithm
        merged_df, ind_to_drop = self._merge(config, df)
        
        # Update flaws from the merged DataFrame
        self._update_flaws_from_dataframe(ind_to_drop, merged_df)
        
        # Regenerate flaw IDs
        self.generate_ind_num()
        
    def _merge(self, config, auto_sizing_summary: pd.DataFrame) -> pd.DataFrame:
        axial_threshold = config.characterization.flaw_merging.axial_threshold
        rotary_threshold_to_0 = config.characterization.flaw_merging.rotary_threshold_to_0
        auto_sizing_summary = auto_sizing_summary.copy()
        auto_sizing_summary['original_order'] = range(len(auto_sizing_summary))
        auto_sizing_summary['flaw_axial_end'] = auto_sizing_summary['flaw_axial_start'] + auto_sizing_summary['flaw_length']
        auto_sizing_summary['flaw_rotary_end'] = auto_sizing_summary['flaw_rotary_start'] + auto_sizing_summary['flaw_width']
        df_sorted = auto_sizing_summary.sort_values(by='flaw_rotary_start').reset_index(drop=True)
        ind_to_drop: List[int] = []
        
        for i1, i2 in itertools.combinations(range(len(df_sorted)), 2):
            # Axial check
            current_start = df_sorted.at[i1, 'flaw_axial_start']
            current_end = df_sorted.at[i1, 'flaw_axial_end']
            next_start = df_sorted.at[i2, 'flaw_axial_start']
            next_end = df_sorted.at[i2, 'flaw_axial_end']
            if abs(next_start - current_start) <= axial_threshold and abs(next_end - current_end) <= axial_threshold and (df_sorted.at[i1, 'flaw_type'] == df_sorted.at[i2, 'flaw_type']):
                
                current_rotary_start = df_sorted.at[i1, 'flaw_rotary_start']
                current_rotary_end = df_sorted.at[i1, 'flaw_rotary_end']
                next_rotary_start = df_sorted.at[i2, 'flaw_rotary_start']
                next_rotary_end = df_sorted.at[i2, 'flaw_rotary_end']
                # Rotary check
                if self.closest_to_0_or_360(current_rotary_start) == 0:
                    if abs(current_rotary_start - 0) <= rotary_threshold_to_0 and abs(next_rotary_end - 360) <= rotary_threshold_to_0:
                        df_sorted, ind_to_drop = self._merge_flaws_based_on_depth(df_sorted, i1, i2, ind_to_drop, current_rotary_end, next_rotary_start, current_start, next_start)
                elif self.closest_to_0_or_360(current_rotary_start) == 360:
                    if abs(current_rotary_end - 360) <= rotary_threshold_to_0 and abs(next_rotary_start - 0) <= rotary_threshold_to_0:
                        df_sorted, ind_to_drop = self._merge_flaws_based_on_depth(df_sorted, i2, i1, ind_to_drop, current_rotary_end, next_rotary_start, current_start, next_start)

        cols_to_nan = df_sorted.columns.difference(['scan_name', 'scan_unit', 'scan_station', 'scan_channel', 'scan_axial_pitch', 'flaw_id', 'flaw_type','original_order'])
        df_sorted.loc[ind_to_drop, cols_to_nan] = ''

        # remove the columns that were added for the merging process
        df_sorted = df_sorted.sort_values(by='original_order').drop(columns=['flaw_axial_end', 'flaw_rotary_end', 'original_order'])

        df_sorted = df_sorted[~df_sorted['flaw_id'].str.contains("merged into", na=False)]

        return df_sorted, ind_to_drop

    @staticmethod
    def _merge_flaws_based_on_depth(df_sorted: pd.DataFrame, i1: int, i2: int, ind_to_drop: List[int], 
                                   current_rotary_end: float, next_rotary_start: float, 
                                   current_start: float, next_start: float) -> Tuple[pd.DataFrame, List[int]]:
        # Determine which flaw has the greater depth and store relevant info
        if df_sorted.at[i1, 'flaw_depth'] > df_sorted.at[i2, 'flaw_depth']:
            chosen_index, delete_index = i1, i2
        elif df_sorted.at[i1, 'flaw_depth'] < df_sorted.at[i2, 'flaw_depth']:
            chosen_index, delete_index = i2, i1
        else:
            # If there are strings, choose whichever one is a float
            if isinstance(df_sorted.at[i1, 'flaw_depth'], float) and not isinstance(df_sorted.at[i2, 'flaw_depth'], float):
                chosen_index, delete_index = i1, i2
            elif not isinstance(df_sorted.at[i1, 'flaw_depth'], float) and isinstance(df_sorted.at[i2, 'flaw_depth'], float):
                chosen_index, delete_index = i2, i1
            else:
                # Default to i1 if both are floats or neither is a float
                chosen_index, delete_index = i1, i2

        # Update the flaw at index with merged information
        df_sorted.at[chosen_index, 'flaw_rotary_start'] = next_rotary_start
        df_sorted.at[chosen_index, 'flaw_axial_start'] = min(current_start, next_start)
        df_sorted.at[chosen_index, 'flaw_width'] = (current_rotary_end - 0) + 360 - next_rotary_start
        df_sorted.at[chosen_index, 'flaw_length'] = max(df_sorted.at[i1, 'flaw_axial_end'], df_sorted.at[i2, 'flaw_axial_end']) - df_sorted.at[chosen_index, 'flaw_axial_start']
        df_sorted.at[chosen_index, 'frame_start'] = min(df_sorted.at[i1, 'frame_start'], df_sorted.at[i2, 'frame_start'])
        df_sorted.at[chosen_index, 'frame_end'] = max(df_sorted.at[i1, 'frame_end'], df_sorted.at[i2, 'frame_end'])
        df_sorted.at[chosen_index, 'confidence'] = (df_sorted.at[chosen_index, 'confidence'] + df_sorted.at[delete_index, 'confidence']) / 2
        df_sorted.at[delete_index, 'flaw_id'] = df_sorted.at[delete_index, 'flaw_id'] + ' (merged into ' + df_sorted.at[chosen_index, 'flaw_id'] + ')'

        ind_to_drop.append(delete_index)
        return df_sorted, ind_to_drop

    def to_dataframe(self):
        data = [{
            'scan_name': self.scan_name,
            'scan_unit': self.scan_unit,
            'scan_station': self.scan_station,
            'scan_channel': self.scan_channel,
            'scan_axial_pitch': self.scan_axial_pitch,
            'flaw_id': flaw.ind_num,
            'flaw_type': flaw.flaw_type,
            'flaw_axial_start': flaw.axial_start,
            'flaw_rotary_start': flaw.rotary_start,
            'flaw_length': flaw.length,
            'flaw_width': flaw.width,
            'flaw_depth': flaw.depth,
            'frame_start': flaw.frame_start,
            'frame_end': flaw.frame_end,
            'confidence': flaw.confidence
            } for flaw in self.flaws]
        df = pd.DataFrame(data)
        return df

    def _update_flaws_from_dataframe(self, ind_to_drop, df):
        # Keep track of flaws to be removed
        flaws_to_remove = []
        
        for id, row in df.iterrows():
            if row['flaw_axial_start'] != '':  # Skip merged flaws
                flaw_found = False
                for flaw in self.flaws:
                    if flaw.flaw_id == id:
                        flaw_found = True
                        flaw.flaw_type = row['flaw_type']
                        flaw.axial_start = row['flaw_axial_start']
                        flaw.rotary_start = row['flaw_rotary_start']
                        flaw.length = row['flaw_length']
                        flaw.width = row['flaw_width']
                        flaw.depth = row['flaw_depth']
                        flaw.frame_start = row['frame_start']
                        flaw.frame_end = row['frame_end']
                        flaw.confidence = row['confidence']
                 
        for flaw in self.flaws:

            # Add the active attribute if it doesn't exist
            if not hasattr(flaw, 'active'):
                flaw.active = True
            
            if flaw.flaw_id in ind_to_drop:
                flaw.active = False
        
        self.flaws = [flaw for flaw in self.flaws if flaw.active]

def daq_to_anf(data, probe_name):
    """
    Apply a specified gain in dB to each element in a 12-bit data array based on the probe name and convert it to 8-bit.
    The function subtracts 2048 from the input, applies a bit shift based on the probe name for the gain, and then clips the result to 0-255.

    Parameters:
    ----------
    data (np.array): The input array of data values in 12-bit range (0-4095).
    probe_name (str): The name of the probe, which determines the gain to be applied. Accepts 'APC', 'CPC', 'NB1', or 'NB2'.

    Returns:
    -------
    np.array: The data array after applying the gain and converting to 8-bit.
    """
    # Subtract 2048 from the data
    data -= 2048

    # Apply the gain using bit shift based on the probe name
    if probe_name in ['APC', 'CPC']:
        data >>= 2  # +12 dB gain for APC and CPC
    elif probe_name in ['NB1', 'NB2']:
        data >>= 3  # +6 dB gain for NB1 and NB2
    else:
        raise ValueError("Invalid probe name. Accepted values are 'APC', 'CPC', 'NB1', 'NB2'.")

    # Add 128 and clip the data to 0-255
    data = np.clip(data + 128, 0, 255)

    # Convert to 8-bit
    return data.astype(np.int16)