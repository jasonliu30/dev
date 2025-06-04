import pandas as pd
from scan_data_structure import Scan
from flaw_data_structure import Flaw
from db import ScanDatabase
from pathlib import Path
from typing import List, Tuple
import json

def generate_actual_path(filename: str) -> Tuple[Path, str, str]:
    """Generate the actual path for a given filename and verify its existence."""
    parts = filename.split()
    
    if 'Type D' not in filename:  # .daq format
        outage_number = parts[1]
        channel = parts[2].strip('-')[:3]
        station = 'Darlington' if outage_number.startswith('D') else 'Pickering'
        base_path = Path(f"S:\\Scan Data - {station}")
        full_path = base_path / outage_number / channel / filename.replace('Type D', 'Type A')
        full_path = full_path.with_suffix('.daq')
    else:
        # Common operations for other formats
        channel = parts[3].replace('-', '')[:3]
        station = 'Darlington' if 'Darlington' in parts[4] else 'Pickering'
        base_path = Path(f"S:\\Scan Data - {station}")

        if len(parts) == 9:  # Edge case for Pickering
            outage_number = parts[3]
            channel = parts[4].strip('-')[:3]
        else:  # Pickering (11 parts) or Darlington
            year = parts[-3].split('-')[-1]
            unit_number = parts[6 if len(parts) == 11 else 5].split('-')[-1]
            outage_number = f"{'D' if station == 'Darlington' else 'P'}{year[-2:]}{unit_number}1"
            if outage_number == 'D2111':
                outage_number = 'D2011'
        full_path = base_path / outage_number / channel / filename.replace('Type D', 'Type A')
        full_path = full_path.with_suffix('.anf')

    # Verify that the generated path exists
    if not full_path.exists():
        raise FileNotFoundError(f"Generated path does not exist: {full_path}")

    return full_path, outage_number, channel

def import_excel_to_database(excel_path: str, database_path: str):
    """Import data from Excel to the database."""
    # Initialize the database
    db = ScanDatabase(database_path)

    # Read the Excel file
    df = pd.read_excel(excel_path)

    # Group the dataframe by filename
    grouped = df.groupby('Filename')

    for filename, group in grouped:
        print(f"Importing {filename}...")
        # Parse filename to get scan information
        # Create a Scan object
        local_path, outage_number, channel = generate_actual_path(filename)

        scan = Scan(
            bscan_path=Path(filename),  
            local_path=generate_actual_path(filename),
            bscan=None,  
            bscan_d=None,  
            scan_name=None,
            scan_unit=None,
            scan_station=None,  # Assuming station is the first part of the unit name
            scan_channel=channel,
            year=None,
            month=None,
            day=None,
            outage_number=outage_number,
            scan_axial_pitch=None,  # Assuming default value, adjust if available
            first_axial=None,
            last_axial=None,
            first_rotary=None,
            last_rotary=None
        )

        # Create Flaw objects for each row in the group
        for _, row in group.iterrows():
            flaw = Flaw()
            flaw.ind_num = row['reported ind num']
            flaw.axial_start = row['Axial Start']
            flaw.rotary_start = row['Rotary Start']
            flaw.length = row['Length (mm)']
            flaw.width = row['Width (deg)']
            flaw.depth = row['Depth (mm)']
            flaw.flaw_type = row['Flaw-group']
            flaw.is_reported = True  # Set this flaw as reported
            flaw.is_predicted = False  # Ensure this is not marked as predicted
            flaw.VERSION = None
            # Add the flaw to the scan
            scan.add_flaw(flaw, reported=True)

        # Add the scan to the database
        db.add_scan(scan)

    print(f"Import completed. {len(grouped)} scans imported to the database.")

def annotate_train_val(json_path='UT_db.json', val_path=None):
    """
    Annotate scans as 'val' if their paths are in the provided list,
    otherwise annotate as None.
    """
    with open(json_path, 'r') as f:
        data = json.load(f)
    # Create a set of scan_ids from the val_paths for faster lookup
    val_scan_ids = set()
    for path in val_path:
        filename = Path(path).name
        scan_id = filename.rsplit('.', 1)[0]
        val_scan_ids.add(scan_id)

    # Annotate scans
    for scan in data.get('scans', []):
        if scan.get('scan_id') in val_scan_ids:
            scan['train_val'] = 'val'
            print(f"Annotated {scan.get('scan_id')} as 'val'.")
        else:
            scan['train_val'] = None
    with open(json_path, 'w') as f:
            json.dump(data, f, indent=2)


def remove_duplicate_scans(json_path='UT_db.json'):
    """
    Remove scans with duplicate scan_ids from the JSON file.
    """
    from collections import OrderedDict

    # Load the JSON file
    try:
        with open(json_path, 'r') as f:
            data = json.load(f)
    except FileNotFoundError:
        print(f"Error: The file '{json_path}' was not found.")
        return
    except json.JSONDecodeError:
        print(f"Error: The file '{json_path}' is not a valid JSON file.")
        return

    # Remove duplicate scan_ids
    unique_scans = OrderedDict()
    duplicates_removed = 0
    for scan in data.get('scans', []):
        scan_id = scan.get('scan_id')
        if scan_id not in unique_scans:
            unique_scans[scan_id] = scan
        else:
            duplicates_removed += 1
            print(f"Removing duplicate scan_id: {scan_id}")

    # Update the data with unique scans
    data['scans'] = list(unique_scans.values())

    # Save the changes back to the JSON file
    try:
        with open('UT_db2.json', 'w') as f:
            json.dump(data, f, indent=2)
        print(f"JSON file updated and saved successfully to {json_path}")
        print(f"Total duplicates removed: {duplicates_removed}")
    except IOError as e:
        print(f"Error saving to JSON file: {str(e)}")

# Usage
if __name__ == '__main__':
    excel_path= 'K-620416-CD-0001-R01, data_map-2024.xlsx'
    database_path= 'UT_db.json'
    import_excel_to_database(excel_path, database_path)
