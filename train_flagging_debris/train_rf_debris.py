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

def load_and_preprocess_data(file_path, positive_threshold=0.038, negative_threshold=-0.030):
    """
    Load Excel data and preprocess it for model training
    Now uses separate positive and negative thresholds
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
    
    # Create binary target variable based on dual thresholds
    print(f"\n=== Creating Binary Target Variable with Dual Thresholds ===")
    print(f"Positive threshold for flagging: depth_diff > {positive_threshold}")
    print(f"Negative threshold for flagging: depth_diff < {negative_threshold}")
    
    # Remove rows where depth_diff is missing first
    mask_valid_target = ~df['depth_diff'].isnull()
    df = df[mask_valid_target].copy()
    
    # Calculate statistics before creating target variable
    print(f"\n=== Depth Diff Statistics (Before Flagging) ===")
    depth_diff_values = df['depth_diff'].values
    mean_depth_diff = np.mean(depth_diff_values)
    std_depth_diff = np.std(depth_diff_values)
    
    print(f"Total samples with valid depth_diff: {len(depth_diff_values)}")
    print(f"Mean of depth_diff: {mean_depth_diff:.6f}")
    print(f"Standard deviation of depth_diff: {std_depth_diff:.6f}")
    print(f"2-sigma range: [{mean_depth_diff - 2*std_depth_diff:.6f}, {mean_depth_diff + 2*std_depth_diff:.6f}]")
    print(f"Min depth_diff: {np.min(depth_diff_values):.6f}")
    print(f"Max depth_diff: {np.max(depth_diff_values):.6f}")
    
    # Distribution of values relative to thresholds
    above_positive = np.sum(depth_diff_values > positive_threshold)
    below_negative = np.sum(depth_diff_values < negative_threshold)
    in_between = np.sum((depth_diff_values >= negative_threshold) & (depth_diff_values <= positive_threshold))
    
    print(f"\nDistribution relative to thresholds:")
    print(f"  Above positive threshold ({positive_threshold}): {above_positive} ({above_positive/len(depth_diff_values)*100:.1f}%)")
    print(f"  Below negative threshold ({negative_threshold}): {below_negative} ({below_negative/len(depth_diff_values)*100:.1f}%)")
    print(f"  Between thresholds: {in_between} ({in_between/len(depth_diff_values)*100:.1f}%)")
    
    # Create binary target: 1 if outside thresholds (flagged), 0 if within thresholds (not flagged)
    df['flagged'] = ((df['depth_diff'] > positive_threshold) | (df['depth_diff'] < negative_threshold)).astype(int)
    
    # Print class distribution
    class_counts = df['flagged'].value_counts()
    print(f"\n=== Class Distribution ===")
    print(f"Not Flagged (0) - within [{negative_threshold}, {positive_threshold}]: {class_counts.get(0, 0)} ({class_counts.get(0, 0)/len(df)*100:.1f}%)")
    print(f"Flagged (1) - outside thresholds: {class_counts.get(1, 0)} ({class_counts.get(1, 0)/len(df)*100:.1f}%)")
    
    # Further breakdown of flagged samples
    flagged_positive = np.sum((df['flagged'] == 1) & (df['depth_diff'] > positive_threshold))
    flagged_negative = np.sum((df['flagged'] == 1) & (df['depth_diff'] < negative_threshold))
    print(f"\nFlagged samples breakdown:")
    print(f"  Flagged due to positive threshold: {flagged_positive} ({flagged_positive/class_counts.get(1, 1)*100:.1f}% of flagged)")
    print(f"  Flagged due to negative threshold: {flagged_negative} ({flagged_negative/class_counts.get(1, 1)*100:.1f}% of flagged)")
    
    # Check for class imbalance
    if class_counts.get(1, 0) > 0 and class_counts.get(0, 0) > 0:
        imbalance_ratio = max(class_counts[0], class_counts[1]) / min(class_counts[0], class_counts[1])
        print(f"\nImbalance ratio (majority/minority): {imbalance_ratio:.2f}")
    else:
        print(f"\nWarning: One class has no samples!")
    
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
    
    # Store thresholds in the dataframe for later reference
    df.attrs['positive_threshold'] = positive_threshold
    df.attrs['negative_threshold'] = negative_threshold
    
    return X.values, y.values, label_encoders, X.columns.tolist(), df

def analyze_depth_diff_distribution(df_original, positive_threshold, negative_threshold):
    """
    Analyze the distribution of depth_diff values around the dual classification thresholds.
    
    Args:
        df_original: Original dataframe with depth_diff column
        positive_threshold: Positive classification threshold
        negative_threshold: Negative classification threshold
    """
    print(f"\n=== DEPTH_DIFF DISTRIBUTION ANALYSIS ===")
    
    if 'depth_diff' not in df_original.columns:
        print("depth_diff column not found in original data")
        return
    
    depth_diff_values = df_original['depth_diff'].dropna()
    
    print(f"Positive threshold: {positive_threshold}")
    print(f"Negative threshold: {negative_threshold}")
    print(f"Total samples with depth_diff: {len(depth_diff_values)}")
    
    # Classification based on dual thresholds
    flagged_positive = depth_diff_values > positive_threshold
    flagged_negative = depth_diff_values < negative_threshold
    flagged_actual = flagged_positive | flagged_negative
    not_flagged_actual = ~flagged_actual
    
    print(f"\nActual class distribution:")
    print(f"  Not Flagged (within [{negative_threshold}, {positive_threshold}]): {not_flagged_actual.sum()} samples")
    print(f"  Flagged (outside thresholds): {flagged_actual.sum()} samples")
    print(f"    - Above positive threshold: {flagged_positive.sum()} samples")
    print(f"    - Below negative threshold: {flagged_negative.sum()} samples")
    
    # Statistics for each class
    print(f"\nDepth_diff statistics by class:")
    
    not_flagged_values = depth_diff_values[not_flagged_actual]
    if len(not_flagged_values) > 0:
        print(f"  Not Flagged samples (within thresholds):")
        print(f"    Min: {not_flagged_values.min():.6f}")
        print(f"    Max: {not_flagged_values.max():.6f}")
        print(f"    Mean: {not_flagged_values.mean():.6f}")
        print(f"    Median: {not_flagged_values.median():.6f}")
        print(f"    Std: {not_flagged_values.std():.6f}")
    
    flagged_values = depth_diff_values[flagged_actual]
    if len(flagged_values) > 0:
        print(f"  Flagged samples (outside thresholds):")
        print(f"    Min: {flagged_values.min():.6f}")
        print(f"    Max: {flagged_values.max():.6f}")
        print(f"    Mean: {flagged_values.mean():.6f}")
        print(f"    Median: {flagged_values.median():.6f}")
        print(f"    Std: {flagged_values.std():.6f}")
        
        # Separate statistics for positive and negative flagged
        if flagged_positive.sum() > 0:
            pos_flagged_values = depth_diff_values[flagged_positive]
            print(f"    Positive flagged (>{positive_threshold}):")
            print(f"      Count: {len(pos_flagged_values)}")
            print(f"      Mean: {pos_flagged_values.mean():.6f}")
            print(f"      Min: {pos_flagged_values.min():.6f}")
        
        if flagged_negative.sum() > 0:
            neg_flagged_values = depth_diff_values[flagged_negative]
            print(f"    Negative flagged (<{negative_threshold}):")
            print(f"      Count: {len(neg_flagged_values)}")
            print(f"      Mean: {neg_flagged_values.mean():.6f}")
            print(f"      Max: {neg_flagged_values.max():.6f}")
    
    # Show samples near the thresholds
    print(f"\nSamples close to thresholds:")
    
    # Near positive threshold
    near_positive = depth_diff_values[
        (depth_diff_values >= positive_threshold - 0.01) & 
        (depth_diff_values <= positive_threshold + 0.01)
    ].sort_values()
    
    if len(near_positive) > 0:
        print(f"  Near positive threshold ({positive_threshold} ± 0.01):")
        print(f"    Found {len(near_positive)} samples")
        for i, val in enumerate(near_positive[:10]):  # Show first 10
            class_label = "Flagged" if val > positive_threshold else "Not Flagged"
            print(f"      {val:.6f} ({class_label})")
        if len(near_positive) > 10:
            print(f"      ... and {len(near_positive) - 10} more")
    
    # Near negative threshold
    near_negative = depth_diff_values[
        (depth_diff_values >= negative_threshold - 0.01) & 
        (depth_diff_values <= negative_threshold + 0.01)
    ].sort_values()
    
    if len(near_negative) > 0:
        print(f"  Near negative threshold ({negative_threshold} ± 0.01):")
        print(f"    Found {len(near_negative)} samples")
        for i, val in enumerate(near_negative[:10]):  # Show first 10
            class_label = "Flagged" if val < negative_threshold else "Not Flagged"
            print(f"      {val:.6f} ({class_label})")
        if len(near_negative) > 10:
            print(f"      ... and {len(near_negative) - 10} more")
    
    return {
        'total_samples': len(depth_diff_values),
        'not_flagged_count': not_flagged_actual.sum(),
        'flagged_count': flagged_actual.sum(),
        'flagged_positive_count': flagged_positive.sum(),
        'flagged_negative_count': flagged_negative.sum(),
        'near_positive_threshold_count': len(near_positive),
        'near_negative_threshold_count': len(near_negative)
    }

def save_model_and_encoders(model, label_encoders, feature_names, positive_threshold, negative_threshold):
    """
    Save the trained model and label encoders with dual thresholds
    """
    # Save the model
    joblib.dump(model, 'random_forest_classifier_depth_diff.pkl')
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders_classifier.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names_classifier.pkl')
    
    # Save thresholds for reference
    thresholds = {
        'positive_threshold': positive_threshold,
        'negative_threshold': negative_threshold
    }
    joblib.dump(thresholds, 'classification_thresholds.pkl')
    
    print("\nModel and encoders saved successfully!")
    print("Files created:")
    print("- random_forest_classifier_depth_diff.pkl")
    print("- label_encoders_classifier.pkl")
    print("- feature_names_classifier.pkl")
    print("- classification_thresholds.pkl")

# Keep all other functions the same (advanced_imputation, feature_selection, etc.)
# Just need to update the main execution block

# Main execution
if __name__ == "__main__":
    # Set your Excel file path here
    excel_file_path = r"features_dataset.xlsx"  
    
    # Set the dual thresholds for flagging
    POSITIVE_THRESHOLD = 0.038  # Flag if depth_diff > this value
    NEGATIVE_THRESHOLD = -0.030  # Flag if depth_diff < this value
    
    # Set training strategy
    MAXIMIZE_TRAINING_ACCURACY = True  # Set to True for balanced high training accuracy
    
    try:
        # Load and preprocess data with dual thresholds
        X, y, label_encoders, feature_names, df = load_and_preprocess_data(
            excel_file_path, 
            positive_threshold=POSITIVE_THRESHOLD,
            negative_threshold=NEGATIVE_THRESHOLD
        )
        
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
            threshold=None,  # We'll modify this function to handle dual thresholds
            train_indices=train_indices,
            test_indices=test_indices,
            output_path=excel_output_path
        )
        
        if saved_path:
            print(f"\n✓ Predictions saved successfully to: {saved_path}")
        else:
            print("\n✗ Failed to save predictions to Excel")
        
        # Analyze feature importance
        feature_importance_dict = analyze_feature_importance(model, selected_features)
        
        # Analyze depth_diff distribution with dual thresholds
        depth_stats = analyze_depth_diff_distribution(df, POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD)
        
        # Create classification plots
        create_classification_plots(y_test, y_pred_test, y_pred_proba_test)
        
        # Save model and encoders with dual thresholds
        save_model_and_encoders(model, selected_encoders, selected_features, 
                               POSITIVE_THRESHOLD, NEGATIVE_THRESHOLD)
        
        # Example of how to use the model for new predictions
        print("\n=== Example: Making predictions on new data ===")
        print("To make predictions on new data, use:")
        print("results = predict_new_data('random_forest_classifier_depth_diff.pkl',")
        print("                           'label_encoders_classifier.pkl',")
        print("                           'feature_names_classifier.pkl',")
        print("                           'classification_thresholds.pkl',")
        print("                           new_dataframe)")
        print("\nThe results will contain:")
        print("- 'flagged_prediction': Binary prediction (0=Not Flagged, 1=Flagged)")
        print("- 'flagged_probability': Probability of being flagged (0.0 to 1.0)")
        print("- 'flagged_classification': Text classification ('Flagged' or 'Not Flagged')")
        print(f"\nCurrent flagging thresholds:")
        print(f"  Positive: depth_diff > {POSITIVE_THRESHOLD}")
        print(f"  Negative: depth_diff < {NEGATIVE_THRESHOLD}")
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