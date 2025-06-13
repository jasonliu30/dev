import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, VotingClassifier, ExtraTreesClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, roc_auc_score, roc_curve
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path, threshold=0.03):
    """
    Load Excel data and preprocess it for model training
    """
    print("Loading data from Excel...")
    df = pd.read_excel(file_path)
    print(f"Data loaded. Shape: {df.shape}")
    
    # Feature engineering for depth_nb1 and depth_nb2
    print("\n=== Feature Engineering for NB1/NB2 ===")
    
    # Create boolean feature: True if either nb1 or nb2 is used (not null)
    df['nb_used'] = ((~df['depth_nb1'].isnull()) | (~df['depth_nb2'].isnull())).astype(int)
    
    # Create delta feature: difference between nb1 and nb2
    # Handle cases where one or both are missing
    df['nb_delta'] = np.nan
    
    # Case 1: Both nb1 and nb2 are available
    both_available = (~df['depth_nb1'].isnull()) & (~df['depth_nb2'].isnull())
    df.loc[both_available, 'nb_delta'] = df.loc[both_available, 'depth_nb1'] - df.loc[both_available, 'depth_nb2']
    
    # Case 2: Only nb1 is available - delta is just nb1 (relative to 0)
    only_nb1 = (~df['depth_nb1'].isnull()) & (df['depth_nb2'].isnull())
    df.loc[only_nb1, 'nb_delta'] = df.loc[only_nb1, 'depth_nb1']
    
    # Case 3: Only nb2 is available - delta is negative nb2 (relative to 0)
    only_nb2 = (df['depth_nb1'].isnull()) & (~df['depth_nb2'].isnull())
    df.loc[only_nb2, 'nb_delta'] = -df.loc[only_nb2, 'depth_nb2']
    
    # Case 4: Neither available - delta remains NaN and will be imputed later
    
    print(f"NB Usage Statistics:")
    print(f"  Rows with either NB1 or NB2: {df['nb_used'].sum()} ({df['nb_used'].mean()*100:.1f}%)")
    print(f"  Rows with both NB1 and NB2: {both_available.sum()} ({both_available.mean()*100:.1f}%)")
    print(f"  Rows with only NB1: {only_nb1.sum()} ({only_nb1.mean()*100:.1f}%)")
    print(f"  Rows with only NB2: {only_nb2.sum()} ({only_nb2.mean()*100:.1f}%)")
    print(f"  Rows with neither: {(~df['nb_used'].astype(bool)).sum()}")
    
    # Define features to use (replacing depth_nb1 and depth_nb2 with new features)
    features = ['pred_depth',
        'depth_apc', 'depth_cpc',
        'bbox_class', 'bbox_max_value', 'bbox_area', 'bbox_width', 'bbox_height',
        'bbox_length', 'bbox_avg_value', 'bbox_avg_position', 'bbox_exclusion_ratio',
        'bbox_total_bboxes', 'bbox_total_v_mr_7_boxes', 'bbox_ignored_v_mr_7_boxes',
        'bbox_v_mr_7_usage_ratio', 'bbox_current_frame_class', 'bbox_prev_1_frame_class',
        'bbox_prev_2_frame_class', 'bbox_next_1_frame_class', 'bbox_next_2_frame_class',
        'bbox_max_depth_frame_probe', 'bbox_total_frames_analyzed', 'bbox_depth_std',
        'bbox_depth_cv', 'bbox_amplitude_based_selection', 'bbox_delta_prev_1',
        'bbox_delta_prev_2', 'bbox_delta_next_1', 'bbox_delta_next_2', 'depth_category'
    ]
    
    # Create binary target variable based on depth_diff threshold
    print(f"\n=== Creating Binary Target Variable ===")
    print(f"Threshold for flagging: {threshold}")
    
    # Remove rows where depth_diff is missing first
    mask_valid_target = ~df['depth_diff'].isnull()
    df = df[mask_valid_target].copy()
    
    # Create binary target: 1 if depth_diff > threshold, 0 otherwise
    df['flagged'] = (df['depth_diff'] > threshold).astype(int)
    
    # Print class distribution
    class_counts = df['flagged'].value_counts()
    print(f"Class distribution:")
    print(f"  Not Flagged (0): {class_counts[0]} ({class_counts[0]/len(df)*100:.1f}%)")
    print(f"  Flagged (1): {class_counts[1]} ({class_counts[1]/len(df)*100:.1f}%)")
    
    # Check for class imbalance
    imbalance_ratio = class_counts[0] / class_counts[1] if class_counts[1] > 0 else float('inf')
    print(f"  Imbalance ratio (majority/minority): {imbalance_ratio:.2f}")
    
    target = 'flagged'
    
    # Check if all required columns exist
    missing_cols = [col for col in features + [target] if col not in df.columns]
    if missing_cols:
        print(f"Warning: Missing columns: {missing_cols}")
        features = [col for col in features if col in df.columns]
    
    # Select features and target
    X = df[features].copy()
    y = df[target].copy()
    
    # Print missing value statistics before imputation
    print("\n=== Missing Value Statistics ===")
    missing_stats = X.isnull().sum()
    missing_stats = missing_stats[missing_stats > 0].sort_values(ascending=False)
    if len(missing_stats) > 0:
        print("Columns with missing values:")
        for col, count in missing_stats.items():
            percentage = (count / len(X)) * 100
            print(f"  {col}: {count} ({percentage:.1f}%)")
    else:
        print("No missing values found in features.")
    
    # Advanced imputation for numerical features
    df_for_missing_indicators = df[features].copy()  # Keep original df for missing indicators
    X = advanced_imputation(X, df_for_missing_indicators)
    
    # Handle categorical variables
    categorical_columns = []
    label_encoders = {}
    
    for col in X.columns:
        if X[col].dtype == 'object' or X[col].dtype == 'bool':
            categorical_columns.append(col)
            le = LabelEncoder()
            # Handle missing values before encoding
            X[col] = X[col].fillna('missing')
            X[col] = le.fit_transform(X[col].astype(str))
            label_encoders[col] = le
    
    print(f"\nEncoded categorical columns: {categorical_columns}")
    
    print(f"\nFinal dataset shape: X: {X.shape}, y: {y.shape}")
    
    return X.values, y.values, label_encoders, X.columns.tolist()

def advanced_imputation(X, df_original):
    """
    Advanced imputation strategy for handling missing values

    Args:
        X (pd.DataFrame): Feature matrix with potential missing values
        df_original (pd.DataFrame): Original dataframe for indicator creation

    Returns:
        pd.DataFrame: Imputed feature matrix
    """
    # Identify numerical columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()

    # Fill related depth features (excluding nb_delta)
    depth_features = ['depth_apc', 'depth_cpc', 'pred_depth']
    existing_depth_features = [col for col in depth_features if col in X.columns]

    if len(existing_depth_features) > 1:
        for col in existing_depth_features:
            if X[col].isnull().any():
                other_cols = [c for c in existing_depth_features if c != col]
                row_mean = X[other_cols].mean(axis=1)
                mask = X[col].isnull()
                X.loc[mask, col] = row_mean[mask]

    # Fixed missing values for specific categorical or sequential features
    fixed_missing_value = -999
    fixed_features = [
        'bbox_prev_1_frame_class', 'bbox_prev_2_frame_class',
        'bbox_next_1_frame_class', 'bbox_next_2_frame_class',
        'bbox_delta_prev_1', 'bbox_delta_prev_2',
        'bbox_delta_next_1', 'bbox_delta_next_2'
    ]
    for col in fixed_features:
        if col in X.columns:
            X[col].fillna(fixed_missing_value, inplace=True)

    # Grouped KNN imputation for other numeric groups
    feature_groups = {
        'spatial': ['axial_start', 'rotary_start'],
        'dimensions': ['length', 'width', 'bbox_width', 'bbox_height', 'bbox_length'],
        'depth': existing_depth_features,
        'bbox_metrics': ['bbox_max_value', 'bbox_area', 'bbox_avg_value', 'bbox_avg_position',
                         'bbox_exclusion_ratio', 'bbox_depth_std', 'bbox_depth_cv'],
        'counts': ['bbox_total_bboxes', 'bbox_total_v_mr_7_boxes', 'bbox_ignored_v_mr_7_boxes',
                   'bbox_total_frames_analyzed', 'bbox_frames_apart']
    }

    for group_name, group_features in feature_groups.items():
        existing_features = [f for f in group_features if f in X.columns and not X[f].isnull().all()]
        if existing_features and X[existing_features].isnull().any().any():
            subset = X[existing_features].copy()
            non_null_rows = subset.dropna().shape[0]
            if non_null_rows >= 2:
                try:
                    imputer = KNNImputer(n_neighbors=min(5, non_null_rows), weights='distance')
                    imputed = imputer.fit_transform(subset)
                    X[existing_features] = pd.DataFrame(imputed, columns=existing_features, index=subset.index)
                except Exception as e:
                    print(f"    Warning: KNN imputation failed for {group_name}: {e}")
                    for col in existing_features:
                        if X[col].isnull().any():
                            X[col].fillna(X[col].median(), inplace=True)
            else:
                for col in existing_features:
                    if X[col].isnull().any():
                        X[col].fillna(X[col].median(), inplace=True)

    # Final fallback
    for col in numerical_columns:
        if col in X.columns and X[col].isnull().any():
            median_value = X[col].median()
            X[col].fillna(0 if pd.isna(median_value) else median_value, inplace=True)

    return X


def feature_selection(X, y, label_encoders, feature_names, missing_threshold=0.5):
    """
    Perform feature selection to improve model performance
    """
    print("\n=== Feature Selection ===")
    
    # Convert to numpy array for easier manipulation
    X_array = X.values if hasattr(X, 'values') else X
    
    # Remove features with too many missing values
    features_to_remove = []
    
    # Check original missing percentages (before imputation)
    print("\nRemoving features with >50% missing values...")
    # Updated list - removed depth_nb2 since it's no longer used
    high_missing_features = ['bbox_avg_max_amp_others', 'bbox_frames_apart', 
                            'bbox_amplitude_ratio']
    
    # Remove features created by imputation for high-missing features
    for feat in high_missing_features:
        if feat in feature_names:
            features_to_remove.append(feat)
            if f'{feat}_was_missing' in feature_names:
                features_to_remove.append(f'{feat}_was_missing')
    
    # Keep track of which features we're removing
    print(f"Removing {len(features_to_remove)} features with high missing rates")
    
    # Get indices of features to keep
    feature_indices_to_keep = [i for i, f in enumerate(feature_names) if f not in features_to_remove]
    feature_names_reduced = [f for f in feature_names if f not in features_to_remove]
    
    # Select columns using indices
    X_reduced = X_array[:, feature_indices_to_keep]
    
    # Train a quick RF to get feature importances
    rf_temp = RandomForestClassifier(
        n_estimators=100, max_depth=5, min_samples_split=20,
        min_samples_leaf=10, random_state=42, n_jobs=-1,
        class_weight='balanced'  # Handle class imbalance
    )
    rf_temp.fit(X_reduced, y)
    
    # Select top N features based on importance
    importances = rf_temp.feature_importances_
    n_features_to_keep = min(20, len(feature_names_reduced))  # Keep top 20 or less
    
    # Get indices of top features
    top_indices = np.argsort(importances)[::-1][:n_features_to_keep]
    
    # Create final feature set
    X_final = X_reduced[:, top_indices]
    final_feature_names = [feature_names_reduced[i] for i in top_indices]
    
    print(f"\nSelected top {n_features_to_keep} features:")
    for i, (idx, name) in enumerate(zip(top_indices, final_feature_names)):
        print(f"  {i+1}. {name}: {importances[idx]:.4f}")
    
    # Update label encoders to only include selected features
    final_label_encoders = {k: v for k, v in label_encoders.items() 
                           if k in final_feature_names}
    
    return X_final, final_feature_names, final_label_encoders

def train_random_forest_classifier(X, y, optimize_hyperparameters=True):
    """
    Train Random Forest classifier with optional hyperparameter optimization
    """
    # Split data - stratify to maintain class distribution
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.15, random_state=42, stratify=y
    )
    
    if optimize_hyperparameters:
        print("\nOptimizing hyperparameters with class imbalance handling...")
        # Define parameter grid optimized for maximum training accuracy
        param_grid = {
            'n_estimators': [200, 300, 500],  # More trees for better fitting
            'max_depth': [None, 15, 20, 25],  # Deeper trees (None = no limit)
            'min_samples_split': [2, 3, 5],  # Lower values allow more splits
            'min_samples_leaf': [1, 2, 3],  # Lower values for finer granularity
            'max_features': [None, 'sqrt', 'log2'],  # Include all features option
            'max_samples': [0.8, 0.9, 1.0],  # Use more/all samples
            'class_weight': [None, 'balanced', 'balanced_subsample']  # Include no balancing
        }
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search optimized for training accuracy
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5,  # Fewer folds to reduce validation strictness
            scoring='accuracy',  # Optimize directly for accuracy
            n_jobs=-1, verbose=1,
            return_train_score=True  # Track training scores
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        rf_model = grid_search.best_estimator_
    else:
        print("\nTraining with parameters optimized for maximum training accuracy...")
        rf_model = RandomForestClassifier(
            n_estimators=300,  # More trees
            max_depth=None,  # No depth limit - allow full depth
            min_samples_split=2,  # Minimum possible - allow maximum splits
            min_samples_leaf=1,  # Minimum possible - finest granularity
            max_features=None,  # Use all features
            max_samples=1.0,  # Use all samples for each tree
            bootstrap=True,  # Keep bootstrap for ensemble diversity
            class_weight=None,  # No class balancing to focus on raw accuracy
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    y_pred_proba_test = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    
    # Calculate classification metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Testing Precision: {test_precision:.4f}")
    print(f"Training Recall: {train_recall:.4f}")
    print(f"Testing Recall: {test_recall:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    print(f"Testing F1-Score: {test_f1:.4f}")
    
    # AUC-ROC if we have both classes in test set
    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, y_pred_proba_test)
        print(f"Testing AUC-ROC: {test_auc:.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_test, target_names=['Not Flagged', 'Flagged']))
    
    # Confusion Matrix for Test Set
    print("\n=== Test Set Confusion Matrix ===")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print("Test Confusion Matrix:")
    print(cm_test)
    
    # Confusion Matrix for Training Set
    print("\n=== Training Set Confusion Matrix ===")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print("Training Confusion Matrix:")
    print(cm_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
def analyze_depth_diff_distribution(df_original, threshold):
    """
    Analyze the distribution of depth_diff values around the classification threshold.
    
    Args:
        df_original: Original dataframe with depth_diff column
        threshold: Classification threshold used
    """
    print(f"\n=== DEPTH_DIFF DISTRIBUTION ANALYSIS ===")
    
    if 'depth_diff' not in df_original.columns:
        print("depth_diff column not found in original data")
        return
    
    depth_diff_values = df_original['depth_diff'].dropna()
    
    print(f"Threshold used for classification: {threshold}")
    print(f"Total samples with depth_diff: {len(depth_diff_values)}")
    
    # Classification based on threshold
    flagged_actual = depth_diff_values > threshold
    not_flagged_actual = depth_diff_values <= threshold
    
    print(f"\nActual class distribution:")
    print(f"  Not Flagged (depth_diff ≤ {threshold}): {not_flagged_actual.sum()} samples")
    print(f"  Flagged (depth_diff > {threshold}): {flagged_actual.sum()} samples")
    
    # Statistics for each class
    print(f"\nDepth_diff statistics by class:")
    
    not_flagged_values = depth_diff_values[not_flagged_actual]
    if len(not_flagged_values) > 0:
        print(f"  Not Flagged samples:")
        print(f"    Min: {not_flagged_values.min():.6f}")
        print(f"    Max: {not_flagged_values.max():.6f}")
        print(f"    Mean: {not_flagged_values.mean():.6f}")
        print(f"    Median: {not_flagged_values.median():.6f}")
        print(f"    Std: {not_flagged_values.std():.6f}")
    
    flagged_values = depth_diff_values[flagged_actual]
    if len(flagged_values) > 0:
        print(f"  Flagged samples:")
        print(f"    Min: {flagged_values.min():.6f}")
        print(f"    Max: {flagged_values.max():.6f}")
        print(f"    Mean: {flagged_values.mean():.6f}")
        print(f"    Median: {flagged_values.median():.6f}")
        print(f"    Std: {flagged_values.std():.6f}")
    
    # Show samples near the threshold
    print(f"\nSamples close to threshold ({threshold}):")
    near_threshold = depth_diff_values[
        (depth_diff_values >= threshold - 0.01) & 
        (depth_diff_values <= threshold + 0.01)
    ].sort_values()
    
    if len(near_threshold) > 0:
        print(f"  Found {len(near_threshold)} samples within ±0.01 of threshold:")
        for i, val in enumerate(near_threshold[:20]):  # Show first 20
            class_label = "Flagged" if val > threshold else "Not Flagged"
            print(f"    {val:.6f} ({class_label})")
        if len(near_threshold) > 20:
            print(f"    ... and {len(near_threshold) - 20} more")
    else:
        print("  No samples found close to threshold")
    
    return {
        'total_samples': len(depth_diff_values),
        'not_flagged_count': not_flagged_actual.sum(),
        'flagged_count': flagged_actual.sum(),
        'near_threshold_count': len(near_threshold)
    }

def detailed_misclassification_analysis_with_depth_diff(X_train, X_test, y_train, y_test, 
                                                      y_pred_train, y_pred_test, df_original, 
                                                      threshold, selected_features):
    """
    Detailed analysis of misclassified samples with their actual depth_diff values.
    
    This function attempts to map misclassified samples back to their original depth_diff values
    by recreating the preprocessing steps.
    
    Args:
        X_train, X_test: Training and test feature arrays
        y_train, y_test: True labels for training and test sets
        y_pred_train, y_pred_test: Predicted labels for training and test sets
        df_original: Original dataframe with depth_diff column
        threshold: Classification threshold used (e.g., 0.03)
        selected_features: List of selected feature names
        
    Returns:
        Dictionary with misclassification analysis results
    """
    print(f"\n=== DETAILED MISCLASSIFICATION ANALYSIS WITH DEPTH_DIFF VALUES ===")
    
    try:
        # First, recreate the feature engineering and preprocessing to maintain index mapping
        df_analysis = df_original.copy()
        
        # Remove rows where depth_diff is missing (same as in preprocessing)
        mask_valid_target = ~df_analysis['depth_diff'].isnull()
        df_analysis = df_analysis[mask_valid_target].copy()
        
        # Create binary target
        df_analysis['flagged'] = (df_analysis['depth_diff'] > threshold).astype(int)
        
        # Store the depth_diff values with their indices
        depth_diff_values = df_analysis['depth_diff'].values
        flagged_values = df_analysis['flagged'].values
        
        # The train/test split should give us the same indices if we use the same random_state
        # Let's recreate the split to get the mapping
        from sklearn.model_selection import train_test_split
        
        # Create dummy X for splitting (we just need the indices)
        dummy_X = np.arange(len(df_analysis))
        _, _, train_indices, test_indices = train_test_split(
            dummy_X, flagged_values, test_size=0.02, random_state=42, stratify=flagged_values
        )
        
        # Now we can map back to original depth_diff values
        train_depth_diff = depth_diff_values[train_indices]
        test_depth_diff = depth_diff_values[test_indices]
        
        print(f"Successfully mapped {len(train_indices)} training and {len(test_indices)} test samples")
        
        # Training set analysis
        print(f"\n=== TRAINING SET DETAILED ANALYSIS ===")
        train_misclassified = y_train != y_pred_train
        n_train_misclassified = train_misclassified.sum()
        
        train_results = {}
        
        if n_train_misclassified > 0:
            misclassified_train_depth_diff = train_depth_diff[train_misclassified]
            misclassified_train_true = y_train[train_misclassified]
            misclassified_train_pred = y_pred_train[train_misclassified]
            
            print(f"Training Misclassifications: {n_train_misclassified}")
            print(f"\nActual depth_diff values for misclassified training samples:")
            
            # Separate false positives and false negatives
            fp_mask = (misclassified_train_pred == 1) & (misclassified_train_true == 0)
            fn_mask = (misclassified_train_pred == 0) & (misclassified_train_true == 1)
            
            # False Positives Analysis
            if fp_mask.any():
                fp_depth_values = misclassified_train_depth_diff[fp_mask]
                train_results['fp_depth_values'] = fp_depth_values
                
                print(f"\nFalse Positives ({fp_mask.sum()} samples) - Predicted Flagged, Actually Not Flagged:")
                print(f"  Their actual depth_diff values (should be ≤ {threshold}):")
                for i, depth_val in enumerate(sorted(fp_depth_values)):
                    print(f"    {i+1}: {depth_val:.6f}")
                print(f"  Range: {fp_depth_values.min():.6f} to {fp_depth_values.max():.6f}")
                print(f"  Mean: {fp_depth_values.mean():.6f}")
                print(f"  Std: {fp_depth_values.std():.6f}")
                
                # Check how close they are to threshold
                distances_to_threshold = np.abs(fp_depth_values - threshold)
                closest_to_threshold = fp_depth_values[np.argmin(distances_to_threshold)]
                print(f"  Closest to threshold: {closest_to_threshold:.6f} (distance: {min(distances_to_threshold):.6f})")
                
            else:
                train_results['fp_depth_values'] = np.array([])
                print(f"\nNo False Positives in training set")
            
            # False Negatives Analysis
            if fn_mask.any():
                fn_depth_values = misclassified_train_depth_diff[fn_mask]
                train_results['fn_depth_values'] = fn_depth_values
                
                print(f"\nFalse Negatives ({fn_mask.sum()} samples) - Predicted Not Flagged, Actually Flagged:")
                print(f"  Their actual depth_diff values (should be > {threshold}):")
                for i, depth_val in enumerate(sorted(fn_depth_values)):
                    print(f"    {i+1}: {depth_val:.6f}")
                print(f"  Range: {fn_depth_values.min():.6f} to {fn_depth_values.max():.6f}")
                print(f"  Mean: {fn_depth_values.mean():.6f}")
                print(f"  Std: {fn_depth_values.std():.6f}")
                
                # Check how close they are to threshold
                distances_to_threshold = np.abs(fn_depth_values - threshold)
                closest_to_threshold = fn_depth_values[np.argmin(distances_to_threshold)]
                print(f"  Closest to threshold: {closest_to_threshold:.6f} (distance: {min(distances_to_threshold):.6f})")
                
            else:
                train_results['fn_depth_values'] = np.array([])
                print(f"\nNo False Negatives in training set")
                
        else:
            print("No misclassifications in training set!")
            train_results = {'fp_depth_values': np.array([]), 'fn_depth_values': np.array([])}
        
        # Test set analysis
        print(f"\n=== TEST SET DETAILED ANALYSIS ===")
        test_misclassified = y_test != y_pred_test
        n_test_misclassified = test_misclassified.sum()
        
        test_results = {}
        
        if n_test_misclassified > 0:
            misclassified_test_depth_diff = test_depth_diff[test_misclassified]
            misclassified_test_true = y_test[test_misclassified]
            misclassified_test_pred = y_pred_test[test_misclassified]
            
            print(f"Test Misclassifications: {n_test_misclassified}")
            print(f"\nActual depth_diff values for misclassified test samples:")
            
            # Separate false positives and false negatives
            fp_mask = (misclassified_test_pred == 1) & (misclassified_test_true == 0)
            fn_mask = (misclassified_test_pred == 0) & (misclassified_test_true == 1)
            
            # False Positives Analysis
            if fp_mask.any():
                fp_depth_values = misclassified_test_depth_diff[fp_mask]
                test_results['fp_depth_values'] = fp_depth_values
                
                print(f"\nFalse Positives ({fp_mask.sum()} samples) - Predicted Flagged, Actually Not Flagged:")
                print(f"  Their actual depth_diff values (should be ≤ {threshold}):")
                for i, depth_val in enumerate(sorted(fp_depth_values)):
                    print(f"    {i+1}: {depth_val:.6f}")
                if len(fp_depth_values) > 0:
                    print(f"  Range: {fp_depth_values.min():.6f} to {fp_depth_values.max():.6f}")
                    print(f"  Mean: {fp_depth_values.mean():.6f}")
                    print(f"  Std: {fp_depth_values.std():.6f}")
                    
                    # Check how close they are to threshold
                    distances_to_threshold = np.abs(fp_depth_values - threshold)
                    closest_to_threshold = fp_depth_values[np.argmin(distances_to_threshold)]
                    print(f"  Closest to threshold: {closest_to_threshold:.6f} (distance: {min(distances_to_threshold):.6f})")
                
            else:
                test_results['fp_depth_values'] = np.array([])
                print(f"\nNo False Positives in test set")
            
            # False Negatives Analysis
            if fn_mask.any():
                fn_depth_values = misclassified_test_depth_diff[fn_mask]
                test_results['fn_depth_values'] = fn_depth_values
                
                print(f"\nFalse Negatives ({fn_mask.sum()} samples) - Predicted Not Flagged, Actually Flagged:")
                print(f"  Their actual depth_diff values (should be > {threshold}):")
                for i, depth_val in enumerate(sorted(fn_depth_values)):
                    print(f"    {i+1}: {depth_val:.6f}")
                if len(fn_depth_values) > 0:
                    print(f"  Range: {fn_depth_values.min():.6f} to {fn_depth_values.max():.6f}")
                    print(f"  Mean: {fn_depth_values.mean():.6f}")
                    print(f"  Std: {fn_depth_values.std():.6f}")
                    
                    # Check how close they are to threshold
                    distances_to_threshold = np.abs(fn_depth_values - threshold)
                    closest_to_threshold = fn_depth_values[np.argmin(distances_to_threshold)]
                    print(f"  Closest to threshold: {closest_to_threshold:.6f} (distance: {min(distances_to_threshold):.6f})")
                
            else:
                test_results['fn_depth_values'] = np.array([])
                print(f"\nNo False Negatives in test set")
                
        else:
            print("No misclassifications in test set!")
            test_results = {'fp_depth_values': np.array([]), 'fn_depth_values': np.array([])}
        
        # Summary statistics
        print(f"\n=== SUMMARY STATISTICS ===")
        total_train_fp = len(train_results.get('fp_depth_values', []))
        total_train_fn = len(train_results.get('fn_depth_values', []))
        total_test_fp = len(test_results.get('fp_depth_values', []))
        total_test_fn = len(test_results.get('fn_depth_values', []))
        
        print(f"Training Set:")
        print(f"  False Positives: {total_train_fp}")
        print(f"  False Negatives: {total_train_fn}")
        print(f"  Total Misclassified: {total_train_fp + total_train_fn}")
        print(f"  Training Accuracy: {1 - (total_train_fp + total_train_fn) / len(y_train):.4f}")
        
        print(f"\nTest Set:")
        print(f"  False Positives: {total_test_fp}")
        print(f"  False Negatives: {total_test_fn}")
        print(f"  Total Misclassified: {total_test_fp + total_test_fn}")
        print(f"  Test Accuracy: {1 - (total_test_fp + total_test_fn) / len(y_test):.4f}")
        
        # Analysis of problematic threshold regions
        all_fp_values = np.concatenate([train_results.get('fp_depth_values', []), 
                                       test_results.get('fp_depth_values', [])])
        all_fn_values = np.concatenate([train_results.get('fn_depth_values', []), 
                                       test_results.get('fn_depth_values', [])])
        
        if len(all_fp_values) > 0 or len(all_fn_values) > 0:
            print(f"\n=== THRESHOLD ANALYSIS ===")
            print(f"Current threshold: {threshold}")
            
            if len(all_fp_values) > 0:
                print(f"False Positives (predicted flagged, actually ≤{threshold}):")
                print(f"  Highest FP value: {all_fp_values.max():.6f}")
                print(f"  Gap from threshold: {threshold - all_fp_values.max():.6f}")
                
            if len(all_fn_values) > 0:
                print(f"False Negatives (predicted not flagged, actually >{threshold}):")
                print(f"  Lowest FN value: {all_fn_values.min():.6f}")
                print(f"  Gap from threshold: {all_fn_values.min() - threshold:.6f}")
                
            # Suggest potential threshold adjustments
            if len(all_fp_values) > 0 and len(all_fn_values) > 0:
                # Find the value between max FP and min FN that might work better
                max_fp = all_fp_values.max()
                min_fn = all_fn_values.min()
                if max_fp < min_fn:
                    suggested_threshold = (max_fp + min_fn) / 2
                    print(f"\nSuggested threshold adjustment: {suggested_threshold:.6f}")
                    print(f"  This would eliminate {len(all_fp_values)} FPs and {len(all_fn_values)} FNs")
                else:
                    print(f"\nThreshold adjustment difficult: overlap between FP and FN regions")
            
        return {
            'train_misclassified_depth_diff': train_depth_diff[train_misclassified] if n_train_misclassified > 0 else np.array([]),
            'test_misclassified_depth_diff': test_depth_diff[test_misclassified] if n_test_misclassified > 0 else np.array([]),
            'train_fp_depth_values': train_results.get('fp_depth_values', np.array([])),
            'train_fn_depth_values': train_results.get('fn_depth_values', np.array([])),
            'test_fp_depth_values': test_results.get('fp_depth_values', np.array([])),
            'test_fn_depth_values': test_results.get('fn_depth_values', np.array([])),
            'mapping_successful': True,
            'train_accuracy': 1 - (total_train_fp + total_train_fn) / len(y_train),
            'test_accuracy': 1 - (total_test_fp + total_test_fn) / len(y_test),
            'threshold_used': threshold
        }
        
    except Exception as e:
        print(f"Could not map misclassifications to depth_diff values: {e}")
        print("Error details:", str(e))
        import traceback
        traceback.print_exc()
        
        # Fallback to basic analysis
        print("\nFalling back to basic misclassification analysis...")
        
        train_misclassified = y_train != y_pred_train
        test_misclassified = y_test != y_pred_test
        
        return {
            'train_misclassified_count': train_misclassified.sum(),
            'test_misclassified_count': test_misclassified.sum(),
            'mapping_successful': False,
            'error': str(e)
        }

def analyze_feature_importance(model, feature_names):
    """
    Analyze and visualize feature importance
    """
    # Get feature importance
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    # Print feature ranking
    print("\n=== Feature Importance Ranking ===")
    for i in range(min(20, len(feature_names))):  # Top 20 features
        print(f"{i+1}. {feature_names[indices[i]]}: {importances[indices[i]]:.4f}")
    
    # Plot feature importance
    plt.figure(figsize=(12, 8))
    top_n = min(20, len(feature_names))
    plt.barh(range(top_n), importances[indices[:top_n]][::-1])
    plt.yticks(range(top_n), [feature_names[i] for i in indices[:top_n]][::-1])
    plt.xlabel('Feature Importance')
    plt.title('Top Feature Importances - Classification Model')
    plt.tight_layout()
    plt.savefig('feature_importance_classification.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dict(zip(feature_names, importances))

def create_classification_plots(y_test, y_pred_test, y_pred_proba_test):
    """
    Create visualization plots for classification predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Confusion Matrix Heatmap
    ax1 = axes[0, 0]
    cm = confusion_matrix(y_test, y_pred_test)
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax1,
                xticklabels=['Not Flagged', 'Flagged'],
                yticklabels=['Not Flagged', 'Flagged'])
    ax1.set_xlabel('Predicted')
    ax1.set_ylabel('Actual')
    ax1.set_title('Confusion Matrix')
    
    # ROC Curve (if we have both classes)
    ax2 = axes[0, 1]
    if len(np.unique(y_test)) > 1:
        fpr, tpr, _ = roc_curve(y_test, y_pred_proba_test)
        auc = roc_auc_score(y_test, y_pred_proba_test)
        ax2.plot(fpr, tpr, label=f'ROC Curve (AUC = {auc:.2f})')
        ax2.plot([0, 1], [0, 1], 'k--', label='Random Classifier')
        ax2.set_xlabel('False Positive Rate')
        ax2.set_ylabel('True Positive Rate')
        ax2.set_title('ROC Curve')
        ax2.legend()
        ax2.grid(True, alpha=0.3)
    else:
        ax2.text(0.5, 0.5, 'ROC Curve requires\nboth classes in test set', 
                ha='center', va='center', transform=ax2.transAxes)
        ax2.set_title('ROC Curve (Not Available)')
    
    # Prediction Probability Distribution
    ax3 = axes[1, 0]
    flagged_probs = y_pred_proba_test[y_test == 1]
    not_flagged_probs = y_pred_proba_test[y_test == 0]
    
    ax3.hist(not_flagged_probs, bins=20, alpha=0.7, label='Not Flagged (Actual)', color='blue')
    ax3.hist(flagged_probs, bins=20, alpha=0.7, label='Flagged (Actual)', color='red')
    ax3.set_xlabel('Predicted Probability of Flagged')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Predicted Probabilities')
    ax3.legend()
    ax3.grid(True, alpha=0.3)
    
    # Class Distribution
    ax4 = axes[1, 1]
    class_counts = np.bincount(y_test)
    pred_class_counts = np.bincount(y_pred_test)
    
    x = np.arange(2)
    width = 0.35
    
    ax4.bar(x - width/2, class_counts, width, label='Actual', alpha=0.7)
    ax4.bar(x + width/2, pred_class_counts, width, label='Predicted', alpha=0.7)
    ax4.set_xlabel('Class')
    ax4.set_ylabel('Count')
    ax4.set_title('Class Distribution: Actual vs Predicted')
    ax4.set_xticks(x)
    ax4.set_xticklabels(['Not Flagged', 'Flagged'])
    ax4.legend()
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('classification_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model_and_encoders(model, label_encoders, feature_names, threshold):
    """
    Save the trained model and label encoders
    """
    # Save the model
    joblib.dump(model, 'random_forest_classifier_depth_diff.pkl')
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders_classifier.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names_classifier.pkl')
    
    # Save threshold for reference
    joblib.dump(threshold, 'classification_threshold.pkl')
    
    print("\nModel and encoders saved successfully!")
    print("Files created:")
    print("- random_forest_classifier_depth_diff.pkl")
    print("- label_encoders_classifier.pkl")
    print("- feature_names_classifier.pkl")
    print("- classification_threshold.pkl")

def predict_new_data(model_path, encoders_path, features_path, threshold_path, new_data):
    """
    Function to make predictions on new data
    Note: new_data should already have the engineered features (nb_used, nb_delta)
    or the original depth_nb1 and depth_nb2 columns for feature engineering
    """
    # Load model and encoders
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    feature_names = joblib.load(features_path)
    threshold = joblib.load(threshold_path)
    
    print(f"Using classification threshold: {threshold}")
    
    # Check if we need to engineer the NB features
    if 'nb_used' not in new_data.columns and ('depth_nb1' in new_data.columns or 'depth_nb2' in new_data.columns):
        print("Engineering NB features for new data...")
        
        # Create boolean feature: True if either nb1 or nb2 is used (not null)
        new_data['nb_used'] = ((~new_data['depth_nb1'].isnull()) | (~new_data['depth_nb2'].isnull())).astype(int)
        
        # Create delta feature: difference between nb1 and nb2
        new_data['nb_delta'] = np.nan
        
        # Case 1: Both nb1 and nb2 are available
        both_available = (~new_data['depth_nb1'].isnull()) & (~new_data['depth_nb2'].isnull())
        new_data.loc[both_available, 'nb_delta'] = new_data.loc[both_available, 'depth_nb1'] - new_data.loc[both_available, 'depth_nb2']
        
        # Case 2: Only nb1 is available
        only_nb1 = (~new_data['depth_nb1'].isnull()) & (new_data['depth_nb2'].isnull())
        new_data.loc[only_nb1, 'nb_delta'] = new_data.loc[only_nb1, 'depth_nb1']
        
        # Case 3: Only nb2 is available
        only_nb2 = (new_data['depth_nb1'].isnull()) & (~new_data['depth_nb2'].isnull())
        new_data.loc[only_nb2, 'nb_delta'] = -new_data.loc[only_nb2, 'depth_nb2']
        
        # Fill any remaining NaN values in nb_delta with 0 (neither available)
        new_data['nb_delta'].fillna(0, inplace=True)
    
    # Preprocess new data
    X_new = new_data[feature_names].copy()
    
    # Apply label encoding to categorical columns
    for col, encoder in label_encoders.items():
        if col in X_new.columns:
            X_new[col] = X_new[col].fillna('missing')
            # Handle unseen categories
            try:
                X_new[col] = encoder.transform(X_new[col].astype(str))
            except ValueError as e:
                print(f"Warning: Unseen categories in {col}. Using 'missing' category.")
                # Replace unseen categories with 'missing'
                known_categories = set(encoder.classes_)
                X_new[col] = X_new[col].astype(str).apply(
                    lambda x: x if x in known_categories else 'missing'
                )
                X_new[col] = encoder.transform(X_new[col])
    
    # Fill any remaining missing values
    for col in X_new.select_dtypes(include=[np.number]).columns:
        X_new[col].fillna(X_new[col].median(), inplace=True)
    
    # Make predictions
    predictions = model.predict(X_new)
    prediction_probabilities = model.predict_proba(X_new)[:, 1]  # Probability of being flagged
    
    # Create results dataframe
    results = pd.DataFrame({
        'flagged_prediction': predictions,
        'flagged_probability': prediction_probabilities,
        'flagged_classification': ['Flagged' if pred == 1 else 'Not Flagged' for pred in predictions]
    })
    
    return results


def train_balanced_random_forest_with_indices(X, y):
    """
    Train Random Forest for high training accuracy while avoiding severe overfitting
    Returns model and train/test indices
    """
    print("\n=== TRAINING BALANCED RANDOM FOREST ===")
    print("Targeting high training accuracy while maintaining reasonable generalization")
    
    # Create indices array
    indices = np.arange(len(X))
    
    # Split data with indices
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.02, random_state=42, stratify=y  
    )
    
    # Test different Random Forest configurations
    rf_configs = {
        'Conservative RF': {
            'n_estimators': 150,
            'max_depth': 10,
            'min_samples_split': 10,
            'min_samples_leaf': 5,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        },
        'Moderate RF': {
            'n_estimators': 200,
            'max_depth': 15,
            'min_samples_split': 5,
            'min_samples_leaf': 2,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        },
        'Aggressive RF': {
            'n_estimators': 300,
            'max_depth': 20,
            'min_samples_split': 3,
            'min_samples_leaf': 1,
            'max_features': 'sqrt',
            'class_weight': 'balanced'
        },
        'High Capacity RF': {
            'n_estimators': 250,
            'max_depth': 25,
            'min_samples_split': 2,
            'min_samples_leaf': 1,
            'max_features': None,  # Use all features
            'class_weight': 'balanced'
        }
    }
    
    best_model = None
    best_score = 0
    best_name = ""
    best_config = None
    
    print("\n=== Random Forest Configuration Comparison ===")
    
    for name, config in rf_configs.items():
        print(f"\nTesting {name}...")
        
        # Train model with current configuration
        rf_model = RandomForestClassifier(
            random_state=42,
            n_jobs=-1,
            **config
        )
        rf_model.fit(X_train, y_train)
        
        # Evaluate performance
        train_pred = rf_model.predict(X_train)
        train_acc = accuracy_score(y_train, train_pred)
        test_pred = rf_model.predict(X_test)
        test_acc = accuracy_score(y_test, test_pred)
        
        # Calculate overfitting gap
        overfitting_gap = train_acc - test_acc
        
        # Composite score: prioritize training accuracy but penalize excessive overfitting
        overfitting_penalty = 1.5  # Moderate penalty
        composite_score = train_acc - (overfitting_penalty * overfitting_gap**2)
        
        print(f"  Training Accuracy: {train_acc:.6f}")
        print(f"  Testing Accuracy: {test_acc:.6f}")
        print(f"  Overfitting Gap: {overfitting_gap:.6f}")
        print(f"  Composite Score: {composite_score:.6f}")
        
        if composite_score > best_score:
            best_score = composite_score
            best_model = rf_model
            best_name = name
            best_config = config
    
    print(f"\n=== BEST RANDOM FOREST CONFIGURATION ===")
    print(f"Configuration: {best_name}")
    print(f"Composite Score: {best_score:.6f}")
    print(f"Parameters: {best_config}")
    
    # Get final predictions from best model
    y_pred_train = best_model.predict(X_train)
    y_pred_test = best_model.predict(X_test)
    y_pred_proba_test = best_model.predict_proba(X_test)[:, 1]
    y_pred_proba_train = best_model.predict_proba(X_train)[:, 1]
    
    # Calculate final metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print(f"\n=== FINAL MODEL PERFORMANCE ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Testing Precision: {test_precision:.4f}")
    print(f"Training Recall: {train_recall:.4f}")
    print(f"Testing Recall: {test_recall:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    print(f"Testing F1-Score: {test_f1:.4f}")
    
    # AUC-ROC if we have both classes in test set
    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, y_pred_proba_test)
        print(f"Testing AUC-ROC: {test_auc:.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_test, target_names=['Not Flagged', 'Flagged']))
    
    # Confusion matrices
    print("\n=== Test Set Confusion Matrix ===")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print("Test Confusion Matrix:")
    print(cm_test)
    
    print("\n=== Training Set Confusion Matrix ===")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print("Training Confusion Matrix:")
    print(cm_train)
    
    # Cross-validation
    cv_scores = cross_val_score(best_model, X, y, cv=10, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return (best_model, X_train, X_test, y_train, y_test, y_pred_test, y_pred_proba_test, 
            y_pred_train, y_pred_proba_train, train_indices, test_indices)


def train_random_forest_classifier_with_indices(X, y, optimize_hyperparameters=True):
    """
    Train Random Forest classifier with optional hyperparameter optimization
    Returns model and train/test indices
    """
    # Create indices array
    indices = np.arange(len(X))
    
    # Split data with indices - stratify to maintain class distribution
    X_train, X_test, y_train, y_test, train_indices, test_indices = train_test_split(
        X, y, indices, test_size=0.05, random_state=42, stratify=y
    )
    
    if optimize_hyperparameters:
        print("\nOptimizing hyperparameters with class imbalance handling...")
        # Define parameter grid optimized for maximum training accuracy
        param_grid = {
            'n_estimators': [200, 300, 500],  # More trees for better fitting
            'max_depth': [None, 15, 20, 25],  # Deeper trees (None = no limit)
            'min_samples_split': [2, 3, 5],  # Lower values allow more splits
            'min_samples_leaf': [1, 2, 3],  # Lower values for finer granularity
            'max_features': [None, 'sqrt', 'log2'],  # Include all features option
            'max_samples': [0.8, 0.9, 1.0],  # Use more/all samples
            'class_weight': [None, 'balanced', 'balanced_subsample']  # Include no balancing
        }
        
        # Create base model
        rf_base = RandomForestClassifier(random_state=42, n_jobs=-1)
        
        # Grid search optimized for training accuracy
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=5,  # Fewer folds to reduce validation strictness
            scoring='accuracy',  # Optimize directly for accuracy
            n_jobs=-1, verbose=1,
            return_train_score=True  # Track training scores
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        rf_model = grid_search.best_estimator_
    else:
        print("\nTraining with parameters optimized for maximum training accuracy...")
        rf_model = RandomForestClassifier(
            n_estimators=300,  # More trees
            max_depth=None,  # No depth limit - allow full depth
            min_samples_split=2,  # Minimum possible - allow maximum splits
            min_samples_leaf=1,  # Minimum possible - finest granularity
            max_features=None,  # Use all features
            max_samples=1.0,  # Use all samples for each tree
            bootstrap=True,  # Keep bootstrap for ensemble diversity
            class_weight=None,  # No class balancing to focus on raw accuracy
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    y_pred_proba_test = rf_model.predict_proba(X_test)[:, 1]  # Probabilities for positive class
    y_pred_proba_train = rf_model.predict_proba(X_train)[:, 1]
    
    # Calculate classification metrics
    train_accuracy = accuracy_score(y_train, y_pred_train)
    test_accuracy = accuracy_score(y_test, y_pred_test)
    train_precision = precision_score(y_train, y_pred_train, zero_division=0)
    test_precision = precision_score(y_test, y_pred_test, zero_division=0)
    train_recall = recall_score(y_train, y_pred_train, zero_division=0)
    test_recall = recall_score(y_test, y_pred_test, zero_division=0)
    train_f1 = f1_score(y_train, y_pred_train, zero_division=0)
    test_f1 = f1_score(y_test, y_pred_test, zero_division=0)
    
    print("\n=== Model Performance ===")
    print(f"Training Accuracy: {train_accuracy:.4f}")
    print(f"Testing Accuracy: {test_accuracy:.4f}")
    print(f"Training Precision: {train_precision:.4f}")
    print(f"Testing Precision: {test_precision:.4f}")
    print(f"Training Recall: {train_recall:.4f}")
    print(f"Testing Recall: {test_recall:.4f}")
    print(f"Training F1-Score: {train_f1:.4f}")
    print(f"Testing F1-Score: {test_f1:.4f}")
    
    # AUC-ROC if we have both classes in test set
    if len(np.unique(y_test)) > 1:
        test_auc = roc_auc_score(y_test, y_pred_proba_test)
        print(f"Testing AUC-ROC: {test_auc:.4f}")
    
    # Classification report
    print("\n=== Classification Report ===")
    print(classification_report(y_test, y_pred_test, target_names=['Not Flagged', 'Flagged']))
    
    # Confusion Matrix for Test Set
    print("\n=== Test Set Confusion Matrix ===")
    cm_test = confusion_matrix(y_test, y_pred_test)
    print("Test Confusion Matrix:")
    print(cm_test)
    
    # Confusion Matrix for Training Set
    print("\n=== Training Set Confusion Matrix ===")
    cm_train = confusion_matrix(y_train, y_pred_train)
    print("Training Confusion Matrix:")
    print(cm_train)
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='f1')
    print(f"\nCross-validation F1 scores: {cv_scores}")
    print(f"Mean CV F1: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return (rf_model, X_train, X_test, y_train, y_test, y_pred_test, y_pred_proba_test, 
            y_pred_train, y_pred_proba_train, train_indices, test_indices)
    
def save_predictions_to_excel(X_train, X_test, y_train, y_test, y_pred_train, y_pred_test, 
                              y_pred_proba_train, y_pred_proba_test, feature_names, 
                              df_original, threshold, train_indices, test_indices,
                              output_path='predictions_output.xlsx'):
    """
    Save train and test predictions to an Excel file with multiple sheets.
    
    Args:
        X_train, X_test: Training and test feature arrays
        y_train, y_test: True labels
        y_pred_train, y_pred_test: Predicted labels
        y_pred_proba_train, y_pred_proba_test: Prediction probabilities
        feature_names: List of feature names
        df_original: Original dataframe with depth_diff values
        threshold: Classification threshold
        output_path: Path to save the Excel file
    """
    print(f"\n=== SAVING PREDICTIONS TO EXCEL ===")
    
    try:
        # Create Excel writer
        with pd.ExcelWriter(output_path, engine='openpyxl') as writer:
            
            # 1. Training Set Predictions
            print("Creating training set predictions sheet...")
            
            # Convert X_train to DataFrame
            train_df = pd.DataFrame(X_train, columns=feature_names)
            
            # Add predictions and actual values
            train_df['actual_label'] = y_train
            train_df['predicted_label'] = y_pred_train
            train_df['predicted_probability'] = y_pred_proba_train
            train_df['actual_class'] = ['Not Flagged' if y == 0 else 'Flagged' for y in y_train]
            train_df['predicted_class'] = ['Not Flagged' if y == 0 else 'Flagged' for y in y_pred_train]
            train_df['correct_prediction'] = (y_train == y_pred_train).astype(int)
            
            # Add misclassification type
            train_df['error_type'] = 'Correct'
            train_df.loc[(y_train == 0) & (y_pred_train == 1), 'error_type'] = 'False Positive'
            train_df.loc[(y_train == 1) & (y_pred_train == 0), 'error_type'] = 'False Negative'
            
            # Map back to original depth_diff values using provided indices
            try:
                # Get the filtered dataframe (after removing nulls)
                df_temp = df_original.copy()
                mask_valid = ~df_temp['depth_diff'].isnull()
                df_temp = df_temp[mask_valid].reset_index(drop=False)  # Keep original index
                
                # Map depth_diff values using the provided train indices
                train_df['depth_diff'] = df_temp.iloc[train_indices]['depth_diff'].values
                train_df['depth_diff_above_threshold'] = (train_df['depth_diff'] > threshold).astype(int)
                
                # Get the actual original row indices from the dataframe
                train_df['original_row_index'] = df_temp.iloc[train_indices]['index'].values
                
            except Exception as e:
                print(f"Warning: Could not map depth_diff values for training set: {e}")
            
            # Reorder columns for better readability
            cols_order = ['original_row_index', 'depth_diff', 'depth_diff_above_threshold', 
                         'actual_label', 'actual_class', 'predicted_label', 'predicted_class', 
                         'predicted_probability', 'correct_prediction', 'error_type'] + feature_names
            cols_order = [col for col in cols_order if col in train_df.columns]
            train_df = train_df[cols_order]
            
            # Save to Excel
            train_df.to_excel(writer, sheet_name='Training_Predictions', index=False)
            
            # 2. Test Set Predictions
            print("Creating test set predictions sheet...")
            
            # Convert X_test to DataFrame
            test_df = pd.DataFrame(X_test, columns=feature_names)
            
            # Add predictions and actual values
            test_df['actual_label'] = y_test
            test_df['predicted_label'] = y_pred_test
            test_df['predicted_probability'] = y_pred_proba_test
            test_df['actual_class'] = ['Not Flagged' if y == 0 else 'Flagged' for y in y_test]
            test_df['predicted_class'] = ['Not Flagged' if y == 0 else 'Flagged' for y in y_pred_test]
            test_df['correct_prediction'] = (y_test == y_pred_test).astype(int)
            
            # Add misclassification type
            test_df['error_type'] = 'Correct'
            test_df.loc[(y_test == 0) & (y_pred_test == 1), 'error_type'] = 'False Positive'
            test_df.loc[(y_test == 1) & (y_pred_test == 0), 'error_type'] = 'False Negative'
            
            # Map back to original depth_diff values using provided indices
            try:
                # Get the filtered dataframe (after removing nulls)
                df_temp = df_original.copy()
                mask_valid = ~df_temp['depth_diff'].isnull()
                df_temp = df_temp[mask_valid].reset_index(drop=False)  # Keep original index
                
                # Map depth_diff values using the provided test indices
                test_df['depth_diff'] = df_temp.iloc[test_indices]['depth_diff'].values
                test_df['depth_diff_above_threshold'] = (test_df['depth_diff'] > threshold).astype(int)
                
                # Get the actual original row indices from the dataframe
                test_df['original_row_index'] = df_temp.iloc[test_indices]['index'].values
                
            except Exception as e:
                print(f"Warning: Could not map depth_diff values for test set: {e}")
            
            # Reorder columns for better readability
            cols_order = ['original_row_index', 'depth_diff', 'depth_diff_above_threshold', 
                         'actual_label', 'actual_class', 'predicted_label', 'predicted_class', 
                         'predicted_probability', 'correct_prediction', 'error_type'] + feature_names
            cols_order = [col for col in cols_order if col in test_df.columns]
            test_df = test_df[cols_order]
            
            # Save to Excel
            test_df.to_excel(writer, sheet_name='Test_Predictions', index=False)
            
            # 3. Summary Statistics Sheet
            print("Creating summary statistics sheet...")
            
            # Calculate metrics for both sets
            train_metrics = {
                'Dataset': 'Training',
                'Total_Samples': len(y_train),
                'Correct_Predictions': (y_train == y_pred_train).sum(),
                'Incorrect_Predictions': (y_train != y_pred_train).sum(),
                'Accuracy': accuracy_score(y_train, y_pred_train),
                'Precision': precision_score(y_train, y_pred_train, zero_division=0),
                'Recall': recall_score(y_train, y_pred_train, zero_division=0),
                'F1_Score': f1_score(y_train, y_pred_train, zero_division=0),
                'True_Positives': ((y_train == 1) & (y_pred_train == 1)).sum(),
                'True_Negatives': ((y_train == 0) & (y_pred_train == 0)).sum(),
                'False_Positives': ((y_train == 0) & (y_pred_train == 1)).sum(),
                'False_Negatives': ((y_train == 1) & (y_pred_train == 0)).sum(),
                'Actual_Flagged': (y_train == 1).sum(),
                'Actual_Not_Flagged': (y_train == 0).sum(),
                'Predicted_Flagged': (y_pred_train == 1).sum(),
                'Predicted_Not_Flagged': (y_pred_train == 0).sum()
            }
            
            test_metrics = {
                'Dataset': 'Test',
                'Total_Samples': len(y_test),
                'Correct_Predictions': (y_test == y_pred_test).sum(),
                'Incorrect_Predictions': (y_test != y_pred_test).sum(),
                'Accuracy': accuracy_score(y_test, y_pred_test),
                'Precision': precision_score(y_test, y_pred_test, zero_division=0),
                'Recall': recall_score(y_test, y_pred_test, zero_division=0),
                'F1_Score': f1_score(y_test, y_pred_test, zero_division=0),
                'True_Positives': ((y_test == 1) & (y_pred_test == 1)).sum(),
                'True_Negatives': ((y_test == 0) & (y_pred_test == 0)).sum(),
                'False_Positives': ((y_test == 0) & (y_pred_test == 1)).sum(),
                'False_Negatives': ((y_test == 1) & (y_pred_test == 0)).sum(),
                'Actual_Flagged': (y_test == 1).sum(),
                'Actual_Not_Flagged': (y_test == 0).sum(),
                'Predicted_Flagged': (y_pred_test == 1).sum(),
                'Predicted_Not_Flagged': (y_pred_test == 0).sum()
            }
            
            # Add AUC if available
            try:
                if len(np.unique(y_test)) > 1:
                    test_metrics['AUC_ROC'] = roc_auc_score(y_test, y_pred_proba_test)
                if len(np.unique(y_train)) > 1:
                    train_metrics['AUC_ROC'] = roc_auc_score(y_train, y_pred_proba_train)
            except:
                pass
            
            summary_df = pd.DataFrame([train_metrics, test_metrics])
            summary_df.to_excel(writer, sheet_name='Summary_Statistics', index=False)
            
            # 4. Misclassified Samples Sheet (sorted by error magnitude)
            print("Creating misclassified samples sheet...")
            
            misclassified_data = []
            
            # Training misclassifications
            train_misclassified_mask = y_train != y_pred_train
            if train_misclassified_mask.any() and 'depth_diff' in train_df.columns:
                train_misclassified = train_df[train_misclassified_mask].copy()
                train_misclassified['dataset'] = 'Training'
                train_misclassified['distance_from_threshold'] = np.abs(train_misclassified['depth_diff'] - threshold)
                misclassified_data.append(train_misclassified)
            
            # Test misclassifications
            test_misclassified_mask = y_test != y_pred_test
            if test_misclassified_mask.any() and 'depth_diff' in test_df.columns:
                test_misclassified = test_df[test_misclassified_mask].copy()
                test_misclassified['dataset'] = 'Test'
                test_misclassified['distance_from_threshold'] = np.abs(test_misclassified['depth_diff'] - threshold)
                misclassified_data.append(test_misclassified)
            
            if misclassified_data:
                misclassified_df = pd.concat(misclassified_data, ignore_index=True)
                # Sort by distance from threshold (closest first)
                misclassified_df = misclassified_df.sort_values('distance_from_threshold')
                
                # Select important columns
                cols_to_show = ['dataset', 'original_row_index', 'depth_diff', 'distance_from_threshold',
                               'actual_class', 'predicted_class', 'predicted_probability', 'error_type']
                cols_to_show = [col for col in cols_to_show if col in misclassified_df.columns]
                misclassified_df[cols_to_show].to_excel(writer, sheet_name='Misclassified_Samples', index=False)
            
            # 5. Model Configuration Sheet
            print("Creating model configuration sheet...")
            
            config_data = {
                'Parameter': ['Classification_Threshold', 'Test_Set_Size', 'Random_State', 
                             'Model_Type', 'Creation_Date'],
                'Value': [threshold, '5%', '33', 'Random Forest (Balanced)', 
                         pd.Timestamp.now().strftime('%Y-%m-%d %H:%M:%S')]
            }
            config_df = pd.DataFrame(config_data)
            config_df.to_excel(writer, sheet_name='Model_Configuration', index=False)
            
        print(f"\nPredictions successfully saved to: {output_path}")
        print(f"Excel file contains {5} sheets:")
        print("  1. Training_Predictions - All training samples with predictions")
        print("  2. Test_Predictions - All test samples with predictions")
        print("  3. Summary_Statistics - Performance metrics for both sets")
        print("  4. Misclassified_Samples - Samples where prediction != actual")
        print("  5. Model_Configuration - Model parameters and settings")
        
        return output_path
        
    except Exception as e:
        print(f"Error saving predictions to Excel: {e}")
        import traceback
        traceback.print_exc()
        return None
# Main execution
if __name__ == "__main__":
    # Set your Excel file path here
    excel_file_path = r"train_flagging_debris\features_dataset.xlsx"  
    
    # Set the threshold for flagging (depth_diff > threshold = flagged)
    FLAGGING_THRESHOLD = 0.03
    
    # Set training strategy
    MAXIMIZE_TRAINING_ACCURACY = True  # Set to True for balanced high training accuracy
    
    try:
        # Load and preprocess data
        X, y, label_encoders, feature_names = load_and_preprocess_data(
            excel_file_path, threshold=FLAGGING_THRESHOLD
        )
        
        # Store original dataframe for depth_diff analysis
        df = pd.read_excel(excel_file_path)
        
        # For balanced high training accuracy, use ALL features (skip feature selection)
        if MAXIMIZE_TRAINING_ACCURACY:
            print("\n=== USING ALL FEATURES FOR BALANCED RANDOM FOREST ===")
            X_selected = X
            selected_features = feature_names
            selected_encoders = label_encoders
        else:
            # Perform feature selection to reduce overfitting
            X_selected, selected_features, selected_encoders = feature_selection(
                X, y, label_encoders, feature_names
            )
        
        print(f"\nDataset shape: {X_selected.shape}")
        
        # Train model based on strategy
        if MAXIMIZE_TRAINING_ACCURACY:
            # Use the balanced Random Forest approach (RF only) - now returns indices
            result = train_balanced_random_forest_with_indices(X_selected, y)
            model, X_train, X_test, y_train, y_test, y_pred_test, y_pred_proba_test, y_pred_train, y_pred_proba_train, train_indices, test_indices = result
        else:
            # Use the standard balanced approach - now returns indices
            result = train_random_forest_classifier_with_indices(X_selected, y, optimize_hyperparameters=True)
            model, X_train, X_test, y_train, y_test, y_pred_test, y_pred_proba_test, y_pred_train, y_pred_proba_train, train_indices, test_indices = result
        
        # Save predictions to Excel with proper indices
        excel_output_path = 'model_predictions_output.xlsx'
        saved_path = save_predictions_to_excel(
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            y_pred_train=y_pred_train,
            y_pred_test=y_pred_test,
            y_pred_proba_train=y_pred_proba_train,
            y_pred_proba_test=y_pred_proba_test,
            feature_names=selected_features,
            df_original=df,
            threshold=FLAGGING_THRESHOLD,
            train_indices=train_indices,  # Now passing the actual indices
            test_indices=test_indices,    # Now passing the actual indices
            output_path=excel_output_path
        )
        
        if saved_path:
            print(f"\n✓ Predictions saved successfully to: {saved_path}")
        else:
            print("\n✗ Failed to save predictions to Excel")
        # Analyze feature importance
        feature_importance_dict = analyze_feature_importance(model, selected_features)
        
        # Analyze misclassified samples and depth_diff distribution
        print("\n" + "="*60)
        misclassification_stats = detailed_misclassification_analysis_with_depth_diff(
            X_train, X_test, y_train, y_test, y_pred_train, y_pred_test,
            df, FLAGGING_THRESHOLD, selected_features
        )
        
        depth_stats = analyze_depth_diff_distribution(df, FLAGGING_THRESHOLD)
        
        # Create classification plots
        create_classification_plots(y_test, y_pred_test, y_pred_proba_test)
        
        # Save model and encoders
        save_model_and_encoders(model, selected_encoders, selected_features, FLAGGING_THRESHOLD)
        
        # Example of how to use the model for new predictions
        print("\n=== Example: Making predictions on new data ===")
        print("To make predictions on new data, use:")
        print("results = predict_new_data('random_forest_classifier_depth_diff.pkl',")
        print("                           'label_encoders_classifier.pkl',")
        print("                           'feature_names_classifier.pkl',")
        print("                           'classification_threshold.pkl',")
        print("                           new_dataframe)")
        print("\nThe results will contain:")
        print("- 'flagged_prediction': Binary prediction (0=Not Flagged, 1=Flagged)")
        print("- 'flagged_probability': Probability of being flagged (0.0 to 1.0)")
        print("- 'flagged_classification': Text classification ('Flagged' or 'Not Flagged')")
        print(f"\nCurrent flagging threshold: depth_diff > {FLAGGING_THRESHOLD}")
        print("\nNote: New data should contain either:")
        print("1. The engineered features 'nb_used' and 'nb_delta', OR")
        print("2. The original 'depth_nb1' and 'depth_nb2' columns for automatic feature engineering")
        
    except FileNotFoundError:
        print(f"Error: Could not find file '{excel_file_path}'")
        print("Please update the excel_file_path variable with your actual file path.")
    except Exception as e:
        print(f"An error occurred: {str(e)}")
        import traceback
        traceback.print_exc()