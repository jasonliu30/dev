"""
Auto Analysis Runner Script for UT B-scan processing.
This script executes the auto analysis pipeline on selected scans.

Usage:
    python test_ptfast.py [--test_name TEST_NAME] [--run_name RUN_NAME] 
                                [--target_scan_id TARGET_SCAN_ID] [--start_from_index START_INDEX]
                                
    python test_ptfast.py --target_scan_id "BSCAN Type A  M-07 Pickering B Unit-5 east 09-Mar-2017 183638 [A2455-2496][R150-520]"
    BSCAN Type D  L-16 Darlington Unit-1 west 07-Apr-2017 195630 [A2774-2822][R0-3599]
    python test_ptfast.py --test_name debris
Options:
    --test_name          Type of test to run (all_flaws, D2421, val, debris, FBBPF, CC, etc.)
    --run_name           Name for the analysis run (defaults to test_name if not provided)
    --target_scan_id     ID of a specific scan to analyze
    --start_from_index   Index to start from when processing multiple scans (default: 0)
"""

import os
import pickle
import argparse
from pathlib import Path
import matplotlib.pyplot as plt

from config import Config
from run_ptfast import load_models
from auto_analysis import auto_analysis, get_cscan
from db.db import ScanDatabase
from depth_sizing.morph_detector import FlawMorphologyAnalyzer

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(description="Auto Analysis Runner for UT B-scan processing")
    
    parser.add_argument("--test_name", type=str, default=None,
                        help="Type of test to run (all_flaws, D2421, val, debris, FBBPF, CC)")
    
    parser.add_argument("--run_name", type=str, default=None,
                        help="Name for this analysis run (defaults to test_name if not provided)")
    
    parser.add_argument("--target_scan_id", type=str, default=None,
                        help="ID of a specific scan to analyze")
    
    parser.add_argument("--start_from_index", type=int, default=0,
                        help="Index to start from when processing multiple scans")
    
    return parser.parse_args()

def main():
    # Parse command line arguments
    args = parse_arguments()
    
    # Set up inputs based on arguments
    test_name = args.test_name
    run_name = args.run_name if args.run_name else test_name
    target_scan_id = args.target_scan_id
    if target_scan_id is not None:
        target_scan_id = target_scan_id.replace("Type A", "Type D")
    start_from_index = args.start_from_index
    
    # If target_scan_id is provided but test_name isn't, default to all_flaws
    if target_scan_id and not test_name:
        test_name = "all_flaws"
        if not run_name:
            run_name = test_name
    
    # Load configuration and models
    config = Config.load_all_configs()
    models = load_models(config)
    axial_range = None

    # Connect to scan database
    my_db = ScanDatabase(r'db\UT_db.json')
    scans_db = my_db.scans
    
    # Select query based on test_name
    query_results = None
    
    if test_name == "all_flaws":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            scan_filters={'train_val': ["train", "val"]}
        )
    elif test_name == "D2421":
        query_results = my_db.flexible_query(
            query_type='flaws',
            scan_filters={"outage_number": "P2251"}
        )
    elif test_name == "val":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            scan_filters={'train_val': ["val"]}
        )    
    elif test_name == "debris":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            flaw_filters={'flaw_type': 'Debris'},
            scan_filters={'train_val': ["train", "val"]}
        )  
    elif test_name == "FBBPF":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            flaw_filters={'flaw_type': ["FBBPF", "BM_FBBPF", "FBBPF (M)"]},
            scan_filters={'train_val': ["train", "val"]}
        )        
    elif test_name == "CC":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            flaw_filters={'flaw_type': ["CC", "CC (M)"]},
            scan_filters={'train_val': ["train", "val"]}
        )
    else:
        # Default query if no test_name is specified
        query_results = my_db.flexible_query(
            query_type='flaws',
            scan_filters={"outage_number": "P2251"}
        )
    
    print(f"Processing {len(query_results)} scans")
    
    # Setup output directories
    save_path = os.path.join('auto-analysis-results', run_name if run_name else "default_run")
    os.makedirs(os.path.join(save_path, 'intermediate'), exist_ok=True)
    
    # Initialize the depth analyzer
    processor = FlawMorphologyAnalyzer(output_dir=os.path.join(save_path, "depth"))
    
    # Use provided target_scan_id or set to None
    target_scan_found = False

    for ind, query_result in enumerate(query_results):
        try:
            current_scan_id = query_result['scan']['scan_id']
            
            # If we're looking for a specific scan
            if target_scan_id:
                if current_scan_id != target_scan_id:
                    continue  # Skip if this isn't the target scan
                print(f"Found target scan at index {ind}: {current_scan_id}")
                target_scan_found = True
            else:
                # We're processing all scans starting from an index
                if ind < start_from_index:
                    continue  # Skip iterations before the specified index
            
            print(f"Processing scan {ind} {current_scan_id}")
            scan_reference = my_db.get_scan(current_scan_id)
            cscans_output = get_cscan(query_result=query_result)
            bscan_path = query_result['scan']['local_path']
            if 'WindowsPath' not in str(bscan_path):
                bscan_path = str(bscan_path)
            else:
                start = bscan_path.find("'") + 1  # Find the first single quote and add 1 to exclude it
                end = bscan_path.find("'", start)  # Find the next single quote
                bscan_path = bscan_path[start:end]

            if len(scan_reference.flaws) > 0:
                updated_scan = auto_analysis(bscan_path, models, config, axial_range=axial_range, 
                                            save_path=save_path, processor=processor, cscans_output=cscans_output, scan_reference=scan_reference)
                    
                query_result = processor.add_depth_fields(query_result, updated_scan)
                my_db.update_query_result(current_scan_id, query_result)
                
                print(f"Successfully processed scan {current_scan_id}")
            else:
                print(f"Skipped scan {current_scan_id} - no flaws")
            
            # If we've processed the target scan, we can break the loop
            if target_scan_id and target_scan_found:
                print(f"Completed processing target scan: {target_scan_id}")
                break

        except Exception as e:
            print(f"Error processing scan {query_result['scan']['scan_id']}: {str(e)}")
            import traceback
            traceback.print_exc()  # Print full traceback for debugging
            
            if target_scan_id and current_scan_id == target_scan_id:
                print(f"Error occurred while processing target scan. See details above.")
                break

    # Check if we were looking for a specific scan but didn't find it
    if target_scan_id and not target_scan_found:
        print(f"WARNING: Target scan '{target_scan_id}' was not found in the query results")

if __name__ == "__main__":
    main()
