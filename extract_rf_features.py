#!/usr/bin/env python3
"""
Script to extract Random Forest training features from ultrasonic scan database
and save to Excel format.

Usage:
    python extract_rf_features.py --test_name debris --output features_dataset.xlsx
    python extract_rf_features.py --test_name all_flaws
    python extract_rf_features.py --test_name CC --outage P2251
    python extract_rf_features.py --outage P2251 --output features_dataset.xlsx
"""

import os
import sys
import argparse
import pandas as pd
import numpy as np
from datetime import datetime
import json


def get_query_by_test_name(my_db, test_name, outage_number=None):
    """
    Get database query based on test_name parameter.
    
    Args:
        my_db: Database instance
        test_name: Type of test to run
        outage_number: Optional outage number for filtering
        
    Returns:
        Query results from database
    """
    if test_name == "all_flaws":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            scan_filters={'train_val': ["train", "val"]}
        )
    elif test_name == "val":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            scan_filters={'train_val': ["val"]}
        )    
    elif test_name == "train":
        query_results = my_db.flexible_query(
            query_type='reported_flaws',
            scan_filters={'train_val': ["train"]}
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
    elif test_name == "outage" and outage_number:
        # Query by outage number
        query_results = my_db.flexible_query(
            query_type='flaws',
            scan_filters={"outage_number": outage_number}
        )
    else:
        # Default to outage-based query if outage_number provided
        if outage_number:
            query_results = my_db.flexible_query(
                query_type='flaws',
                scan_filters={"outage_number": outage_number}
            )
        else:
            # Fallback to all reported flaws
            query_results = my_db.flexible_query(
                query_type='reported_flaws',
                scan_filters={'train_val': ["train", "val"]}
            )
    
    return query_results


def extract_features_from_query_results(query_results):
    """
    Extract features from database query results for Random Forest model training.
    
    Args:
        query_results: List of query results from my_db.flexible_query()
        
    Returns:
        List of dictionaries, each containing features for one reported flaw
    """
    all_features = []
    
    for query_result in query_results:
        # Access the scan data
        scan_data = query_result.get('scan', {})
        scan_id = scan_data.get('scan_id')
        
        # Process each reported flaw
        for flaw in scan_data.get('reported_flaws', []):
            # Skip if no prediction was made
            if not flaw.get('is_predicted', False):
                continue
                
            # Skip if no depth_diff (target variable)
            if not flaw.get('metrics', {}).get('depth_diff'):
                continue
            
            # Extract base features
            features = {
                # Scan identifier
                'scan_id': scan_id,
                
                # Basic flaw measurements
                'axial_start': flaw.get('axial_start'),
                'rotary_start': flaw.get('rotary_start'),
                'length': flaw.get('length'),
                'width': flaw.get('width'),
                
                # Depth predictions
                'pred_depth': flaw.get('pred_depth'),
                'depth_nb1': flaw.get('depth_nb1'),
                'depth_nb2': flaw.get('depth_nb2'),
                'depth_apc': flaw.get('depth_apc'),
                'depth_cpc': flaw.get('depth_cpc'),
                
                # Target variable
                'depth_diff': flaw.get('metrics', {}).get('depth_diff'),
                
                # Additional metrics
                'reported_depth': flaw.get('depth'),
                'width_diff': flaw.get('metrics', {}).get('width_diff'),
                'length_diff': flaw.get('metrics', {}).get('length_diff'),
                'position_x_diff': flaw.get('metrics', {}).get('position_x_diff'),
                'position_y_diff': flaw.get('metrics', {}).get('position_y_diff'),
            }
            
            # Extract all bbox_stats features
            bbox_stats = flaw.get('bbox_stats', {})
            if bbox_stats:
                # Add all bbox_stats fields
                for key, value in bbox_stats.items():
                    features[f'bbox_{key}'] = value
            
            # Add metadata for tracking/filtering
            features['ind_num'] = flaw.get('ind_num')
            features['flaw_type'] = flaw.get('flaw_type')
            features['is_classified_correct'] = flaw.get('is_classified_correct')
            features['iou'] = flaw.get('iou')
            features['depth_category'] = flaw.get('depth_category')
            
            # Add scan metadata
            features['outage_number'] = scan_data.get('outage_number')
            features['scan_channel'] = scan_data.get('scan_channel')
            
            all_features.append(features)
    
    return all_features


def prepare_excel_dataset(query_results, output_path):
    """
    Prepare and save dataset to Excel with multiple sheets for analysis.
    
    Args:
        query_results: List of query results from database
        output_path: Path to save Excel file
    """
    # Extract all features
    all_features = extract_features_from_query_results(query_results)
    
    if not all_features:
        print("No valid features found in query results")
        return None
    
    # Convert to DataFrame
    df = pd.DataFrame(all_features)
    
    print(f"\nDataset Summary:")
    print(f"Total samples: {len(df)}")
    print(f"Total features: {len(df.columns)}")
    print(f"Scans processed: {df['scan_id'].nunique()}")
    print(f"Flaw types: {df['flaw_type'].value_counts().to_dict()}")
    
    # Create Excel writer
    with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
        
        # 1. Main dataset sheet
        df.to_excel(writer, sheet_name='Full_Dataset', index=False)
        
        # 2. Feature summary sheet
        feature_summary = pd.DataFrame({
            'Feature': df.columns,
            'Type': [df[col].dtype for col in df.columns],
            'Non_Null_Count': [df[col].notna().sum() for col in df.columns],
            'Null_Count': [df[col].isna().sum() for col in df.columns],
            'Null_Percentage': [(df[col].isna().sum() / len(df) * 100).round(2) for col in df.columns],
            'Unique_Values': [df[col].nunique() for col in df.columns],
            'Sample_Values': [str(df[col].dropna().unique()[:5].tolist()) for col in df.columns]
        })
        feature_summary.to_excel(writer, sheet_name='Feature_Summary', index=False)
        
        # 3. Numeric features statistics
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        numeric_cols = [col for col in numeric_cols if col != 'depth_diff']  # Exclude target from features
        
        if len(numeric_cols) > 0:
            numeric_stats = df[numeric_cols].describe().round(4)
            numeric_stats.to_excel(writer, sheet_name='Numeric_Stats')
        
        # 4. Target variable analysis
        target_analysis = pd.DataFrame({
            'Metric': ['Count', 'Mean', 'Std', 'Min', '25%', '50%', '75%', 'Max', 
                      'Positive_Count', 'Negative_Count', 'Zero_Count'],
            'Value': [
                len(df['depth_diff']),
                df['depth_diff'].mean(),
                df['depth_diff'].std(),
                df['depth_diff'].min(),
                df['depth_diff'].quantile(0.25),
                df['depth_diff'].median(),
                df['depth_diff'].quantile(0.75),
                df['depth_diff'].max(),
                (df['depth_diff'] > 0).sum(),
                (df['depth_diff'] < 0).sum(),
                (df['depth_diff'] == 0).sum()
            ]
        })
        target_analysis.to_excel(writer, sheet_name='Target_Analysis', index=False)
        
        # 5. Correlation matrix for numeric features
        if len(numeric_cols) > 1:
            # Include target in correlation
            corr_cols = list(numeric_cols) + ['depth_diff']
            correlation_matrix = df[corr_cols].corr().round(3)
            correlation_matrix.to_excel(writer, sheet_name='Correlations')
        
        # 6. Missing data analysis
        missing_data = pd.DataFrame({
            'Feature': df.columns,
            'Missing_Count': df.isnull().sum(),
            'Missing_Percentage': (df.isnull().sum() / len(df) * 100).round(2)
        })
        missing_data = missing_data[missing_data['Missing_Count'] > 0].sort_values('Missing_Count', ascending=False)
        if len(missing_data) > 0:
            missing_data.to_excel(writer, sheet_name='Missing_Data', index=False)
        
        # 7. Sample data by flaw type
        for flaw_type in df['flaw_type'].unique():
            if pd.notna(flaw_type):
                flaw_df = df[df['flaw_type'] == flaw_type].head(100)
                sheet_name = f'Sample_{flaw_type}'[:31]  # Excel sheet name limit
                flaw_df.to_excel(writer, sheet_name=sheet_name, index=False)
    
    print(f"\nDataset saved to: {output_path}")
    
    # Print feature groups
    print("\nFeature Groups:")
    print("- Basic measurements: axial_start, rotary_start, length, width")
    print("- Depth features: pred_depth, depth_nb1, depth_nb2, depth_apc, depth_cpc")
    print("- Difference metrics: depth_diff (TARGET), width_diff, length_diff, position_x_diff, position_y_diff")
    print("- BBox stats: All features starting with 'bbox_'")
    print("- Metadata: scan_id, ind_num, flaw_type, is_classified_correct, iou, depth_category")
    
    return df


def main():
    """Main function to run the feature extraction."""
    parser = argparse.ArgumentParser(description='Extract RF features from ultrasonic scan database')
    
    # Test name parameter (similar to test_ptfast.py)
    parser.add_argument('--test_name', type=str, default=None,
                        help='Type of test to run (all_flaws, val, train, debris, FBBPF, CC, outage)')
    
    # Legacy outage parameter for backward compatibility
    parser.add_argument('--outage', type=str, default=None, 
                        help='Outage number to process (for backward compatibility)')
    
    parser.add_argument('--output', type=str, default=None, 
                        help='Output Excel file path')
    
    parser.add_argument('--db_path', type=str, default=None, 
                        help='Database path (if needed)')
    
    parser.add_argument('--limit', type=int, default=None, 
                        help='Limit number of scans to process')
    
    args = parser.parse_args()
    
    # Determine the query type and parameters
    test_name = args.test_name
    outage_number = args.outage
    
    # If no test_name provided but outage is provided, use outage mode
    if not test_name and outage_number:
        test_name = "outage"
    elif not test_name:
        # Default to all_flaws if nothing specified
        test_name = "all_flaws"
    
    # Generate output filename if not provided
    if args.output is None:
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        if outage_number:
            args.output = f'rf_features_{test_name}_{outage_number}_{timestamp}.xlsx'
        else:
            args.output = f'rf_features_{test_name}_{timestamp}.xlsx'
    
    try:
        # Import your database module
        # Adjust this import based on your project structure
        from db.db import ScanDatabase  # UPDATE THIS LINE
        
        # Initialize database connection
        if args.db_path:
            my_db = ScanDatabase(args.db_path)
        else:
            my_db = ScanDatabase(r'db\UT_db.json')
        
        # Query the database based on test_name
        print(f"Querying database for test_name: {test_name}")
        if outage_number:
            print(f"Using outage number: {outage_number}")
            
        query_results = get_query_by_test_name(my_db, test_name, outage_number)
        
        if args.limit:
            query_results = query_results[:args.limit]
            print(f"Limited to {len(query_results)} scans")
        else:
            print(f"Retrieved {len(query_results)} scans")
        
        # Generate Excel dataset
        df = prepare_excel_dataset(query_results, args.output)
        
        # Optional: Save as CSV as well
        csv_path = args.output.replace('.xlsx', '.csv')
        if df is not None:
            df.to_csv(csv_path, index=False)
            print(f"Also saved CSV to: {csv_path}")
        
        # Print some basic statistics
        if df is not None:
            print("\nTarget Variable Statistics:")
            print(f"Mean depth_diff: {df['depth_diff'].mean():.6f}")
            print(f"Std depth_diff: {df['depth_diff'].std():.6f}")
            print(f"Min depth_diff: {df['depth_diff'].min():.6f}")
            print(f"Max depth_diff: {df['depth_diff'].max():.6f}")
            
            # Print breakdown by test type
            print(f"\nTest Summary:")
            print(f"Test name: {test_name}")
            if outage_number:
                print(f"Outage: {outage_number}")
            print(f"Total samples: {len(df)}")
            print(f"Unique scans: {df['scan_id'].nunique()}")
            print(f"Flaw type distribution:")
            for flaw_type, count in df['flaw_type'].value_counts().items():
                print(f"  {flaw_type}: {count}")
        
    except ImportError:
        print("ERROR: Could not import database module.")
        print("Please update the import statement in the main() function.")
        print("Replace 'from db.db import ScanDatabase' with your actual import.")
        
        # For testing without database, create sample data
        print("\nGenerating sample data for demonstration...")
        sample_data = create_sample_data()
        df = prepare_excel_dataset(sample_data, args.output)
        
    except Exception as e:
        print(f"Error: {e}")
        import traceback
        traceback.print_exc()


def create_sample_data():
    """Create sample data for testing without database."""
    # This is just for demonstration - remove in production
    sample_query_results = [
        {
            'scan': {
                'scan_id': 'SCAN_001',
                'outage_number': 'P2251',
                'scan_channel': 'A11',
                'reported_flaws': [
                    {
                        'ind_num': 'Ind 1',
                        'axial_start': 3159.66,
                        'rotary_start': 182.6,
                        'length': 2.376,
                        'width': 1.0,
                        'depth': 0.15,
                        'pred_depth': 0.1136,
                        'depth_nb1': 0.1136,
                        'depth_nb2': None,
                        'depth_apc': 0.0561,
                        'depth_cpc': 0.0484,
                        'flaw_type': 'Debris',
                        'is_predicted': True,
                        'is_classified_correct': True,
                        'iou': 0.551,
                        'depth_category': 'V+d',
                        'metrics': {
                            'width_diff': 0.3,
                            'length_diff': 0.175,
                            'position_x_diff': 0.0,
                            'position_y_diff': 0.339,
                            'depth_diff': 0.0364
                        },
                        'bbox_stats': {
                            'class': 'V',
                            'max_value': 172,
                            'row_position': 'middle',
                            'selection_reason': 'leftmost V box',
                            'area': 170,
                            'width': 17,
                            'height': 10,
                            'length': 17,
                            'avg_value': 153.9,
                            'avg_position': 46.0,
                            'exclusion_ratio': 0.722,
                            'total_bboxes': 3,
                            'total_v_mr_7_boxes': 2,
                            'ignored_v_mr_7_boxes': 1,
                            'v_mr_7_usage_ratio': 0.5,
                            'current_frame_class': 'V+d',
                            'prev_1_frame_class': None,
                            'prev_2_frame_class': None,
                            'next_1_frame_class': 'V+d',
                            'next_2_frame_class': None,
                            'max_depth_frame_probe': '18-NB1',
                            'total_frames_analyzed': 2,
                            'depth_std': 0.0078,
                            'depth_cv': 0.0737,
                            'amplitude_based_selection': False,
                            'amplitude_ratio': None,
                            'frames_apart': None,
                            'avg_max_amp_others': None,
                            'delta_prev_1': None,
                            'delta_prev_2': None,
                            'delta_next_1': -0.0156,
                            'delta_next_2': None
                        }
                    }
                ]
            }
        }
    ]
    return sample_query_results


if __name__ == "__main__":
    main()