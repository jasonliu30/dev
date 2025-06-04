import json
from typing import List, Dict, Optional, Any
from pathlib import Path
from scan_data_structure import Scan
from flaw_data_structure import Flaw
import numpy as np


class ScanDatabase:
    def __init__(self, database_path: str):
        self.database_path = Path(database_path)
        self.scans: List[Dict] = []
        self.load_database()

    def load_database(self):
        if self.database_path.exists():
            with open(self.database_path, 'r') as f:
                data = json.load(f)
                self.scans = data.get('scans', [])
        else:
            self.scans = []

    def save_database(self):
        with open(self.database_path, 'w') as f:
            json.dump({'scans': self.scans}, f, indent=2)

    def add_scan(self, scan: Scan):
        scan_dict = self._scan_to_dict(scan)
        self.scans.append(scan_dict)
        self.save_database()

    def update_scan(self, scan_id: str, updated_scan: Scan):
        """
        Update a scan in the database, preserving existing non-null values.
        For VERSION scans, preserves the original depth value and adds depth2 and depth_category.

        Parameters
        ----------
        scan_id : str
            The ID of the scan to update.
        updated_scan : Scan
            The updated Scan object.

        Raises
        ------
        ValueError
            If the scan with the given ID is not found.
        """
        for i, s in enumerate(self.scans):
            if s['scan_id'] == scan_id:
                # Convert the Scan object to a dictionary
                updated_dict = self._scan_to_dict(updated_scan)
                
                # First, preserve reported_id and add new fields for each flaw
                for updated_flaw in updated_dict['flaws']:
                    # Find matching flaw in existing scan
                    matching_flaw = next((f for f in s['flaws'] 
                                        if f['ind_num'] == updated_flaw['ind_num']), None)
                    if matching_flaw and 'reported_id' in matching_flaw:
                        # Explicitly preserve reported_id
                        updated_flaw['reported_id'] = matching_flaw['reported_id']
                    
                    # Ensure new fields exist
                    if 'depth2' not in updated_flaw:
                        updated_flaw['depth2'] = None
                    if 'depth_category' not in updated_flaw:
                        updated_flaw['depth_category'] = None
                
                # Then do the general preservation of non-null values
                for key in s.keys():
                    if key == 'reported_flaws':
                        updated_dict[key] = s[key]
                    elif key not in updated_dict or updated_dict[key] is None:
                        updated_dict[key] = s[key]
                
                self.scans[i] = updated_dict
                print(f'database updated for {scan_id}')
                self.save_database()
                return
        raise ValueError(f"Scan with id {scan_id} not found")

    def delete_scan(self, scan_id: str):
        self.scans = [s for s in self.scans if s['scan_id'] != scan_id]
        self.save_database()

    def get_scan(self, scan_id: str) -> Optional[Scan]:
        for s in self.scans:
            if s['scan_id'] == scan_id:
                return self._dict_to_scan(s)
        return None

    def get_all_scans(self) -> List[Scan]:
        return [self._dict_to_scan(s) for s in self.scans]

    def query_scans(self, **kwargs) -> List[Scan]:
        result = []
        for s in self.scans:
            if all(s.get(k) == v for k, v in kwargs.items()):
                result.append(self._dict_to_scan(s))
        return result

    def _scan_to_dict(self, scan: Scan) -> Dict:
        scan_dict = {
            'scan_id': scan.bscan_path.stem,  # Using bscan_path.stem as unique identifier
            'bscan_path': str(scan.bscan_path),
            'local_path': str(scan.local_path),
            'scan_name': scan.scan_name,
            'scan_unit': scan.scan_unit,
            'scan_station': scan.scan_station,
            'scan_channel': scan.scan_channel,
            'year': scan.year,
            'month': scan.month,
            'day': scan.day,
            'outage_number': scan.outage_number,
            'scan_axial_pitch': scan.scan_axial_pitch,
            'first_axial': scan.first_axial,
            'last_axial': scan.last_axial,
            'first_rotary': scan.first_rotary,
            'last_rotary': scan.last_rotary,
            'flaws': [self._flaw_to_dict(f) for f in scan.flaws],
            'reported_flaws': [self._flaw_to_dict(f) for f in scan.reported_flaws]
        }
        return scan_dict

    def _dict_to_scan(self, scan_dict: Dict) -> Scan:
        scan = Scan(
            bscan_path=Path(scan_dict['bscan_path']),
            local_path=Path(scan_dict['local_path']),
            bscan=None,  # These objects can't be serialized, so we'll need to load them separately
            bscan_d=None,
            scan_name=scan_dict['scan_name'],
            scan_unit=scan_dict['scan_unit'],
            scan_station=scan_dict['scan_station'],
            scan_channel=scan_dict['scan_channel'],
            year=scan_dict['year'],
            month=scan_dict['month'],
            day=scan_dict['day'],
            outage_number=scan_dict['outage_number'],
            scan_axial_pitch=scan_dict['scan_axial_pitch'],
            first_axial=scan_dict['first_axial'],
            last_axial=scan_dict['last_axial'],
            first_rotary=scan_dict['first_rotary'],
            last_rotary=scan_dict['last_rotary']
        )
        scan.flaws = [self._dict_to_flaw(f) for f in scan_dict['flaws']]
        scan.reported_flaws = [self._dict_to_flaw(f) for f in scan_dict['reported_flaws']]
        return scan

    def _flaw_to_dict(self, flaw: Flaw) -> Dict:
        try:
            flaw_dict = {
            'ind_num': flaw.ind_num,
            'axial_start': flaw.axial_start,
            'rotary_start': flaw.rotary_start,
            'frame_start':flaw.frame_start,
            'frame_end':flaw.frame_end,
            'length': flaw.length,
            'width': flaw.width,
            'depth': flaw.depth,
            'depth2': flaw.depth2,
            'depth_category': flaw.depth_category,
            'flaw_type': flaw.flaw_type,
            'is_reported': flaw.is_reported,
            'is_predicted': flaw.is_predicted,
            'VERSION': flaw.version,
            'reported_id': getattr(flaw, 'reported_id', None)
        }
        except AttributeError:
            flaw_dict = {
            'ind_num': flaw.ind_num,
            'axial_start': flaw.axial_start,
            'rotary_start': flaw.rotary_start,
            'length': flaw.length,
            'width': flaw.width,
            'depth': flaw.depth,
            'flaw_type': flaw.flaw_type,
            'is_reported': flaw.is_reported,
            'is_predicted': flaw.is_predicted,
            'VERSION': flaw.version,
            'reported_id': getattr(flaw, 'reported_id', None)
        }
            

        try:
            flaw_dict['reported_id'] = flaw.reported_id
        except AttributeError:
            return flaw_dict
        return flaw_dict

    def _dict_to_flaw(self, flaw_dict: Dict) -> Flaw:
        flaw = Flaw()
        for key, value in flaw_dict.items():
            setattr(flaw, key, value)
        return flaw
    
    def get_total_scans(self) -> int:
        """Return the total number of scans in the database."""
        return len(self.scans)

    def get_total_flaws(self) -> int:
        """Return the total number of flaws across all scans."""
        return sum(len(scan['flaws']) for scan in self.scans)

    def get_total_reported_flaws(self) -> int:
        """Return the total number of reported flaws across all scans."""
        return sum(len(scan['reported_flaws']) for scan in self.scans)

    def get_flaw_types(self, reported=True) -> List[str]:
        """Return a list of unique flaw types in the database."""
        flaw_types = set()
        if reported:
            type_key = 'reported_flaws'
        else:
            type_key = 'flaws'
        for scan in self.scans:
            for flaw in scan[type_key]:
                flaw_types.add(flaw['flaw_type'])
        return list(flaw_types)

    def get_flaws_by_type(self, flaw_type: str, reported=True) -> List[Dict]:
        """Return a list of flaws of a specific type across all scans."""
        flaws = []
        for scan in self.scans:
            for flaw in scan['flaws']:
                if flaw['flaw_type'] == flaw_type:
                    flaws.append({**flaw, 'scan_id': scan['scan_id']})
        return flaws

    def get_scans_by_year(self, year: int) -> List[Scan]:
        """Return a list of scans from a specific year."""
        return self.query_scans(year=year)

    def get_scans_by_unit(self, unit: str) -> List[Scan]:
        """Return a list of scans for a specific unit."""
        return self.query_scans(scan_unit=unit)

    def get_flaw_statistics(self) -> Dict[str, int]:
        """Return statistics about flaws in the database."""
        stats = {
            'total_predicted_flaws': self.get_total_flaws(),
            'total_reported_flaws': self.get_total_reported_flaws()
        }
        return stats

    def get_scan_statistics(self) -> Dict[str, int]:
        """Return statistics about scans in the database."""
        stats = {
            'total_scans': self.get_total_scans(),
            'unique_units': len(set(scan['scan_unit'] for scan in self.scans)),
            'unique_stations': len(set(scan['scan_station'] for scan in self.scans)),
            'unique_years': len(set(scan['year'] for scan in self.scans)),
            'unique_outages': len(set(scan['outage_number'] for scan in self.scans))
        }
        return stats

    def flexible_query(self, query_type: str, scan_filters: Dict[str, Any] = None, flaw_filters: Dict[str, Any] = None) -> List[Dict[str, Any]]:
        """
        Perform a flexible query on the ScanDatabase.
        
        :param query_type: Type of query ('scans', 'flaws', or 'reported_flaws')
        :param scan_filters: Filters to apply on scan attributes (lists are treated as OR conditions)
        :param flaw_filters: Filters to apply on flaw attributes (string values use partial matching)
        :return: List of dictionaries containing the query results
        """
        results = []
        scan_filters = scan_filters or {}
        flaw_filters = flaw_filters or {}

        for scan in self.scans:
            # Check scan filters with support for list values (OR condition)
            scan_match = True
            for k, v in scan_filters.items():
                scan_value = scan.get(k)
                
                # Handle list of possible values (OR condition)
                if isinstance(v, list):
                    if scan_value not in v:
                        scan_match = False
                        break
                # Handle single value
                elif scan_value != v:
                    scan_match = False
                    break
            
            if scan_match:
                if query_type == 'scans':
                    results.append(scan)  # scan is already a dict, no need for conversion
                elif query_type in ['flaws', 'reported_flaws']:
                    flaw_list = scan[query_type]
                    matching_flaws = []
                    
                    for flaw in flaw_list:
                        # Check each flaw against all filters
                        matches_all_filters = True
                        
                        for k, v in flaw_filters.items():
                            flaw_value = flaw.get(k)
                            
                            # Skip None values
                            if flaw_value is None:
                                matches_all_filters = False
                                break
                                
                            # For strings, use partial matching (contains)
                            if isinstance(v, str) and isinstance(flaw_value, str):
                                if v.lower() not in flaw_value.lower():  # Case-insensitive contains
                                    matches_all_filters = False
                                    break
                            # For lists, check if any value is contained
                            elif isinstance(v, list) and isinstance(flaw_value, str):
                                if not any(val.lower() in flaw_value.lower() for val in v if isinstance(val, str)):
                                    matches_all_filters = False
                                    break
                            # For other types, use exact matching
                            elif flaw_value != v:
                                matches_all_filters = False
                                break
                        
                        if matches_all_filters:
                            matching_flaws.append(flaw)
                    
                    if matching_flaws:
                        results.append({
                            'scan': scan
                        })

        return QueryResult(self, results)

    def match_flaws_to_reported(self, scan_id: str, iou_threshold: float = 0.2) -> None:
        """
        Match predicted flaws to reported flaws based on IoU.

        Parameters
        ----------
        scan_id : str
            The ID of the scan to process.
        iou_threshold : float, optional
            The minimum IoU required for a match (default is 0.2).

        Returns
        -------
        None
            The method modifies the flaws in place.
        """
        scan = self.get_scan(scan_id)
        if not scan:
            raise ValueError(f"Scan with id {scan_id} not found")

        def calculate_iou(box1: Dict[str, float], box2: Dict[str, float]) -> float:
            """Calculate IoU between two bounding boxes."""
            x1 = max(box1['rotary_start'], box2['rotary_start'])
            y1 = max(box1['axial_start'], box2['axial_start'])
            x2 = min(box1['rotary_start'] + box1['width'], box2['rotary_start'] + box2['width'])
            y2 = min(box1['axial_start'] + box1['length'], box2['axial_start'] + box2['length'])

            intersection = max(0, x2 - x1) * max(0, y2 - y1)
            area1 = box1['width'] * box1['length']
            area2 = box2['width'] * box2['length']
            union = area1 + area2 - intersection

            return intersection / union if union > 0 else 0

        for predicted_flaw in scan.flaws:
            best_iou = 0
            best_match = None

            for reported_flaw in scan.reported_flaws:
                try:
                    box1 = {
                            'rotary_start': predicted_flaw.rotary_start,
                            'axial_start': predicted_flaw.axial_start,
                            'width': predicted_flaw.width,
                            'length': predicted_flaw.length
                        }
                    box2 =                 {
                            'rotary_start': reported_flaw.rotary_start,
                            'axial_start': reported_flaw.axial_start,
                            'width': float(str(reported_flaw.width).replace('<','').replace('*','').split('(')[0]),
                            'length': float(str(reported_flaw.length).replace('<','').replace('*','').split('(')[0])
                        }



                    iou = calculate_iou(box1,box2)
                
                except ValueError as e:
                    print(e)
                    continue

                if iou > best_iou:
                    best_iou = iou
                    best_match = reported_flaw

            if best_iou >= iou_threshold:
                predicted_flaw.reported_id = best_match.ind_num

        # Update the scan in the database
        self.update_scan(scan_id, scan)

    def calculate_flaw_depth_stats(self, save_path = None):
        """
        Calculate flaw depth stats
        """
        flaw_depth_stats={'mean absolute error':None,'2 sigma error':None,'mean error':None,'median error':None,'count':None}

        def calculate_error_stats(stats:dict,errors:list):
            for key in stats.keys():
                if stats[key]==None:
                    if key=='mean absolute error':
                        stats[key] = np.mean(np.abs(errors))
                    elif key=='2 sigma error':
                        stats[key] = 2*np.std(errors)
                    elif key == 'mean error':
                        stats[key] = np.mean(errors)
                    elif key=='median error':
                        stats[key]=np.median(errors)
                    elif key=='count':
                        stats[key]=len(errors)
            return stats
                        
        all_errors=[]
        flaw_type_errors={}
            
        for scan in self.scans:
            for predicted_flaw in scan['flaws']:

                try:
                    matched_ind = predicted_flaw['reported_id'] # This can be changed if there is a better way to check for a match
                except:
                    continue

                for reported_flaw in scan['reported_flaws']: # find the reported flaw to calculate error
                    if matched_ind==reported_flaw['ind_num']:
                        try:
                            reported_depth =float(str(reported_flaw['depth']).split('(')[0].replace('<',''))
                            predicted_depth = float(str(predicted_flaw['depth']).split('(')[0].replace('<',''))
                            if np.isnan(reported_depth) or np.isnan(predicted_depth):
                                raise ValueError
                        except: # if depth is see note or not a number
                            break
                        all_errors.append( reported_depth- predicted_depth)
                        if reported_flaw['flaw_type'] not in flaw_type_errors:
                            flaw_type_errors[reported_flaw['flaw_type']] = []
                        flaw_type_errors[reported_flaw['flaw_type']].append(reported_depth- predicted_depth)
                        break

        results = flaw_depth_stats.copy()
        results = calculate_error_stats(results,all_errors)

        for flawtype in flaw_type_errors.keys():
            if 'flaw type stats' not in results.keys():
                results['flaw type stats'] = {}
            results['flaw type stats'][flawtype] = calculate_error_stats(flaw_depth_stats.copy(), flaw_type_errors[flawtype])
        
        if save_path!=None:
            with open(save_path, 'w') as f:
                json.dump(results, f, indent=2)
                
        return results
    def update_query_result(self, scan_id, query_result):
        """
        Update a scan and its related data in the database
        
        Args:
            scan_id (str): The ID of the scan to update
            query_result (dict): The entire query result containing scan and related data
        """
        # Find and update the scan in the database
        for scan in self.scans:
            if scan['scan_id'] == scan_id:
                scan.update(query_result['scan'])
                # If you need to update other related collections (like matching_flaws),
                # you would do that here
                break
                
        # Save the updated database to file
        with open(self.database_path, 'w') as f:
            json.dump({'scans': self.scans}, f, indent=2)

class QueryResult:
    def __init__(self, database: 'ScanDatabase', results: List[Dict[str, Any]]):
        self.database = database
        self.results = results

    def __len__(self):
        return len(self.results)

    def __getitem__(self, index):
        return self.results[index]

    def __iter__(self):
        return iter(self.results)
    
    def update_scan(self, scan_index: int, updated_scan: Scan):
            """
            Update a scan in the database based on the query results, preserving reported flaws,
            depth (for VERSION updates), rotary_start, flaw_type, and existing non-null values.

            Parameters
            ----------
            scan_index : int
                Index of the scan in the query results to update.
            updated_scan : Scan
                The updated Scan object.

            Raises
            ------
            IndexError
                If the scan_index is out of range.
            """
            if scan_index < 0 or scan_index >= len(self.results):
                raise IndexError("scan_index out of range")

            result = self.results[scan_index]
            if 'scan' in result:
                existing_scan = result['scan']
                scan_id = existing_scan['scan_id']
            else:
                existing_scan = result
                scan_id = existing_scan['scan_id']

            # Get the existing scan from the database
            existing_db_scan = self.database.get_scan(scan_id)
            if existing_db_scan is None:
                raise ValueError(f"Scan with id {scan_id} not found in database")

            # Preserve attributes from existing flaws
            existing_flaw_data = {}
            for flaw in existing_db_scan.flaws:
                flaw_data = {}
                if hasattr(flaw, 'reported_id'):
                    flaw_data['reported_id'] = flaw.reported_id
                if hasattr(flaw, 'depth'):
                    flaw_data['depth'] = flaw.depth
                if hasattr(flaw, 'rotary_start'):
                    flaw_data['rotary_start'] = flaw.rotary_start
                if hasattr(flaw, 'flaw_type'):
                    flaw_data['flaw_type'] = flaw.flaw_type
                existing_flaw_data[flaw.ind_num] = flaw_data

            # Update the existing scan with new values, preserving non-null existing values
            for attr, value in vars(updated_scan).items():
                if attr == 'reported_flaws':
                    # Always preserve existing reported flaws
                    continue
                if value is not None:
                    if attr == 'flaws':
                        # Ensure we don't add any new flaws
                        new_flaws = []
                        for flaw in value:
                            if flaw.ind_num in existing_flaw_data:
                                existing_data = existing_flaw_data[flaw.ind_num]
                                # Preserve reported_id
                                if 'reported_id' in existing_data:
                                    flaw.reported_id = existing_data['reported_id']
                                # Preserve depth if VERSION is being updated
                                if ('depth' in existing_data and 
                                    hasattr(flaw, 'version') and 
                                    hasattr(existing_db_scan.flaws[0], 'version') and 
                                    flaw.version != existing_db_scan.flaws[0].version):
                                    flaw.depth = existing_data['depth']
                                # Preserve rotary_start
                                if 'rotary_start' in existing_data:
                                    flaw.rotary_start = existing_data['rotary_start']
                                # Preserve flaw_type
                                if 'flaw_type' in existing_data:
                                    flaw.flaw_type = existing_data['flaw_type']
                                new_flaws.append(flaw)
                        # Only update with flaws that existed in the original scan
                        setattr(existing_db_scan, attr, new_flaws)
                    else:
                        setattr(existing_db_scan, attr, value)

            # Update the scan in the database
            self.database.update_scan(scan_id, existing_db_scan)

            # Update the results to reflect the changes
            updated_dict = self.database._scan_to_dict(existing_db_scan)
            if 'scan' in result:
                result['scan'] = updated_dict
            else:
                self.results[scan_index] = updated_dict