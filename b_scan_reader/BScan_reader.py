import pandas as pd
import BScan_anf
import BScan_daq
from pathlib import Path
from typing import Union


def load_bscan(b_scan_file: Union[Path, str]):
    if isinstance(b_scan_file, str):
        b_scan_file = Path(b_scan_file)

    if b_scan_file.suffix == '.anf':
        b_scan = BScan_anf.BScan_anf(b_scan_file)
    elif b_scan_file.suffix == '.daq':
        b_scan = BScan_daq.BScan_daq(b_scan_file)
    else:
        exception_text = b_scan_file.name + " is not a .daq or .anf file."
        raise FileNotFoundError(exception_text)
    return b_scan


def build_path(row: pd.DataFrame, raw_data_root: Path):
    """
    Obtain the correct format of the directory of the .anf or .daq files.

    Args:
        row: Current row being analyzed
        raw_data_root: Path to the raw data root folder, common to all data files.

    Returns: Path to the Type A and Type D BScan files

    """

    if row["Outage Number"][0] == "D":
        station = "Scan Data - Darlington"
    elif row["Outage Number"][0] == "P":
        station = "Scan Data - Pickering"
    else:
        raise ValueError(f"Unsupported Outage Number. Supported Values are 'D' or 'P', but {row['Outage Number'][0]} was received.")

    outage_dir = row["Outage Number"]

    # Get the right channel format. e.g. D6 -> D06
    row_channel = row["Channel"].split(' ')[0]
    if len(row_channel.replace('-', '')) == 2:
        channel_dir = row_channel[0] + "0" + row_channel[2]
    else:
        channel_dir = row_channel.replace('-', '')

    type_d_fn = row["Filename"]
    type_a_fn = row["Filename"].replace('BSCAN Type D', 'BSCAN Type A')

    # all scan files at the P2251 outage were daq files except channel N09, which were .anf files
    if row["Outage Number"] == "P2251" and channel_dir != 'N09':
        b_scan_a_dir = raw_data_root / station / outage_dir / channel_dir / (type_a_fn + '.daq')
        b_scan_d_dir = None
    else:
        b_scan_a_dir = raw_data_root / station / outage_dir / channel_dir / (type_a_fn + '.anf')
        b_scan_d_dir = raw_data_root / station / outage_dir / channel_dir / (type_d_fn + '.anf')

    return b_scan_a_dir, b_scan_d_dir
