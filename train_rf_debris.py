import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
from sklearn.impute import KNNImputer
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import warnings
warnings.filterwarnings('ignore')

def load_and_preprocess_data(file_path):
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
    features = [
        'axial_start', 'rotary_start', 'length', 'width', 'pred_depth',
        'nb_used', 'nb_delta',  # New engineered features
        'depth_apc', 'depth_cpc',
        'bbox_class', 'bbox_max_value', 'bbox_area', 'bbox_width', 'bbox_height',
        'bbox_length', 'bbox_avg_value', 'bbox_avg_position', 'bbox_exclusion_ratio',
        'bbox_total_bboxes', 'bbox_total_v_mr_7_boxes', 'bbox_ignored_v_mr_7_boxes',
        'bbox_v_mr_7_usage_ratio', 'bbox_current_frame_class', 'bbox_prev_1_frame_class',
        'bbox_prev_2_frame_class', 'bbox_next_1_frame_class', 'bbox_next_2_frame_class',
        'bbox_max_depth_frame_probe', 'bbox_total_frames_analyzed', 'bbox_depth_std',
        'bbox_depth_cv', 'bbox_amplitude_based_selection', 'bbox_amplitude_ratio',
        'bbox_frames_apart', 'bbox_avg_max_amp_others', 'bbox_delta_prev_1',
        'bbox_delta_prev_2', 'bbox_delta_next_1', 'bbox_delta_next_2', 'depth_category'
    ]
    
    target = 'depth_diff'
    
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
    
    # Remove rows where target is missing
    mask = ~y.isnull()
    X = X[mask]
    y = y[mask]
    
    print(f"\nFinal dataset shape: X: {X.shape}, y: {y.shape}")
    
    return X.values, y.values, label_encoders, X.columns.tolist()

def advanced_imputation(X, df_original):
    """
    Advanced imputation strategy for handling missing values
    """
    # Identify numerical columns
    numerical_columns = X.select_dtypes(include=[np.number]).columns.tolist()
    
    # Strategy 1: Handle paired features (updated to exclude nb1/nb2 since they're replaced)
    paired_features = [
        ('bbox_delta_prev_1', 'bbox_delta_prev_2'),
        ('bbox_delta_next_1', 'bbox_delta_next_2')
    ]
    
    for feat1, feat2 in paired_features:
        if feat1 in X.columns and feat2 in X.columns:
            # If one is missing but not the other, use the available one
            mask1_missing = X[feat1].isnull()
            mask2_missing = X[feat2].isnull()
            
            # Fill feat1 with feat2 where feat1 is missing but feat2 is not
            X.loc[mask1_missing & ~mask2_missing, feat1] = X.loc[mask1_missing & ~mask2_missing, feat2]
            
            # Fill feat2 with feat1 where feat2 is missing but feat1 is not
            X.loc[mask2_missing & ~mask1_missing, feat2] = X.loc[mask2_missing & ~mask1_missing, feat1]
    
    # Strategy 2: Handle sequential features (prev/next pattern)
    sequential_groups = [
        ['bbox_delta_prev_2', 'bbox_delta_prev_1', 'bbox_delta_next_1', 'bbox_delta_next_2']
    ]
    
    for group in sequential_groups:
        existing_cols = [col for col in group if col in X.columns]
        if len(existing_cols) > 1:
            # For each missing value, use interpolation from neighboring values
            for i, col in enumerate(existing_cols):
                if X[col].isnull().any():
                    # Get neighboring columns
                    neighbors = []
                    if i > 0:
                        neighbors.append(existing_cols[i-1])
                    if i < len(existing_cols) - 1:
                        neighbors.append(existing_cols[i+1])
                    
                    if neighbors:
                        # Fill with mean of available neighbors
                        neighbor_mean = X[neighbors].mean(axis=1)
                        mask = X[col].isnull()
                        X.loc[mask, col] = neighbor_mean[mask]
    
    # Strategy 3: Fill related depth features (updated to include nb_delta)
    depth_features = ['nb_delta', 'depth_apc', 'depth_cpc', 'pred_depth']
    existing_depth_features = [col for col in depth_features if col in X.columns]
    
    if len(existing_depth_features) > 1:
        for col in existing_depth_features:
            if X[col].isnull().any():
                # Fill with mean of other depth features for that row
                other_depth_cols = [c for c in existing_depth_features if c != col]
                if other_depth_cols:
                    row_mean = X[other_depth_cols].mean(axis=1)
                    mask = X[col].isnull()
                    X.loc[mask, col] = row_mean[mask]
    
    # Strategy 4: KNN imputation for remaining missing values
    # Group features by type for better KNN performance
    feature_groups = {
        'spatial': ['axial_start', 'rotary_start'],
        'dimensions': ['length', 'width', 'bbox_width', 'bbox_height', 'bbox_length'],
        'depth': existing_depth_features,
        'bbox_metrics': ['bbox_max_value', 'bbox_area', 'bbox_avg_value', 'bbox_avg_position',
                        'bbox_exclusion_ratio', 'bbox_depth_std', 'bbox_depth_cv'],
        'deltas': ['bbox_delta_prev_1', 'bbox_delta_prev_2', 'bbox_delta_next_1', 'bbox_delta_next_2'],
        'counts': ['bbox_total_bboxes', 'bbox_total_v_mr_7_boxes', 'bbox_ignored_v_mr_7_boxes',
                  'bbox_total_frames_analyzed', 'bbox_frames_apart']
    }
    
    # Apply KNN imputation to each group
    for group_name, group_features in feature_groups.items():
        # Get features that exist in the dataframe
        existing_features = [f for f in group_features if f in numerical_columns and f in X.columns]
        
        if existing_features and X[existing_features].isnull().any().any():
            print(f"  Applying KNN imputation to {group_name} features...")
            knn_imputer = KNNImputer(n_neighbors=5, weights='distance')
            X[existing_features] = knn_imputer.fit_transform(X[existing_features])
    
    # Strategy 5: Final fallback - median imputation for any remaining NaNs
    for col in numerical_columns:
        if col in X.columns and X[col].isnull().any():
            X[col].fillna(X[col].median(), inplace=True)
    
    # Add indicator features for important missing patterns
    important_missing_cols = ['bbox_delta_prev_1', 'bbox_delta_prev_2', 
                             'bbox_delta_next_1', 'bbox_delta_next_2', 'nb_delta']
    
    for col in important_missing_cols:
        if col in X.columns and col in df_original.columns:
            # Create a binary indicator for whether the value was missing
            X[f'{col}_was_missing'] = df_original[col].isnull().astype(int)
    
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
    rf_temp = RandomForestRegressor(
        n_estimators=100, max_depth=5, min_samples_split=20,
        min_samples_leaf=10, random_state=42, n_jobs=-1
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

def train_random_forest(X, y, optimize_hyperparameters=True):
    """
    Train Random Forest model with optional hyperparameter optimization
    """
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    
    if optimize_hyperparameters:
        print("\nOptimizing hyperparameters with regularization focus...")
        # Define parameter grid with more conservative parameters for small dataset
        param_grid = {
            'n_estimators': [50, 100, 150],
            'max_depth': [3, 5, 7, 10],  # Shallower trees to prevent overfitting
            'min_samples_split': [10, 20, 30],  # Higher values for regularization
            'min_samples_leaf': [5, 10, 15],  # Higher values for regularization
            'max_features': ['sqrt', 'log2'],  # Fewer features per split
            'max_samples': [0.5, 0.7, 0.9]  # Bootstrap sample size
        }
        
        # Create base model
        rf_base = RandomForestRegressor(random_state=42, n_jobs=-1)
        
        # Grid search with more folds for small dataset
        grid_search = GridSearchCV(
            rf_base, param_grid, cv=10,  # More folds for better validation
            scoring='neg_mean_squared_error',
            n_jobs=-1, verbose=1
        )
        
        grid_search.fit(X_train, y_train)
        
        print(f"Best parameters: {grid_search.best_params_}")
        rf_model = grid_search.best_estimator_
    else:
        print("\nTraining with regularized parameters for small dataset...")
        rf_model = RandomForestRegressor(
            n_estimators=100,
            max_depth=5,  # Shallow trees
            min_samples_split=20,  # High minimum samples to split
            min_samples_leaf=10,  # High minimum samples in leaf
            max_features='sqrt',  # Use only sqrt of features
            max_samples=0.8,  # Use 80% of samples for each tree
            random_state=42,
            n_jobs=-1
        )
        rf_model.fit(X_train, y_train)
    
    # Make predictions
    y_pred_train = rf_model.predict(X_train)
    y_pred_test = rf_model.predict(X_test)
    
    # Calculate metrics
    train_mse = mean_squared_error(y_train, y_pred_train)
    test_mse = mean_squared_error(y_test, y_pred_test)
    train_mae = mean_absolute_error(y_train, y_pred_train)
    test_mae = mean_absolute_error(y_test, y_pred_test)
    train_r2 = r2_score(y_train, y_pred_train)
    test_r2 = r2_score(y_test, y_pred_test)
    
    print("\n=== Model Performance ===")
    print(f"Training MSE: {train_mse:.6f}")
    print(f"Testing MSE: {test_mse:.6f}")
    print(f"Training MAE: {train_mae:.6f}")
    print(f"Testing MAE: {test_mae:.6f}")
    print(f"Training R²: {train_r2:.4f}")
    print(f"Testing R²: {test_r2:.4f}")
    
    # Cross-validation
    cv_scores = cross_val_score(rf_model, X, y, cv=10, scoring='r2')
    print(f"\nCross-validation R² scores: {cv_scores}")
    print(f"Mean CV R²: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")
    
    return rf_model, X_train, X_test, y_train, y_test, y_pred_test

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
    plt.title('Top Feature Importances')
    plt.tight_layout()
    plt.savefig('feature_importance.png', dpi=300, bbox_inches='tight')
    plt.show()
    
    return dict(zip(feature_names, importances))

def create_prediction_plots(y_test, y_pred_test):
    """
    Create visualization plots for predictions
    """
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    
    # Scatter plot of predictions vs actual
    ax1 = axes[0, 0]
    ax1.scatter(y_test, y_pred_test, alpha=0.5)
    ax1.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'r--', lw=2)
    ax1.set_xlabel('Actual depth_diff')
    ax1.set_ylabel('Predicted depth_diff')
    ax1.set_title('Predictions vs Actual Values')
    ax1.grid(True, alpha=0.3)
    
    # Residual plot
    ax2 = axes[0, 1]
    residuals = y_test - y_pred_test
    ax2.scatter(y_pred_test, residuals, alpha=0.5)
    ax2.axhline(y=0, color='r', linestyle='--')
    ax2.set_xlabel('Predicted depth_diff')
    ax2.set_ylabel('Residuals')
    ax2.set_title('Residual Plot')
    ax2.grid(True, alpha=0.3)
    
    # Distribution of residuals
    ax3 = axes[1, 0]
    ax3.hist(residuals, bins=50, edgecolor='black', alpha=0.7)
    ax3.set_xlabel('Residuals')
    ax3.set_ylabel('Frequency')
    ax3.set_title('Distribution of Residuals')
    ax3.grid(True, alpha=0.3)
    
    # Error distribution
    ax4 = axes[1, 1]
    abs_errors = np.abs(residuals)
    ax4.hist(abs_errors, bins=50, edgecolor='black', alpha=0.7, color='orange')
    ax4.set_xlabel('Absolute Error')
    ax4.set_ylabel('Frequency')
    ax4.set_title('Distribution of Absolute Errors')
    ax4.grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('prediction_analysis.png', dpi=300, bbox_inches='tight')
    plt.show()

def save_model_and_encoders(model, label_encoders, feature_names):
    """
    Save the trained model and label encoders
    """
    # Save the model
    joblib.dump(model, 'random_forest_depth_diff_model.pkl')
    
    # Save label encoders
    joblib.dump(label_encoders, 'label_encoders.pkl')
    
    # Save feature names
    joblib.dump(feature_names, 'feature_names.pkl')
    
    print("\nModel and encoders saved successfully!")
    print("Files created:")
    print("- random_forest_depth_diff_model.pkl")
    print("- label_encoders.pkl")
    print("- feature_names.pkl")

def predict_new_data(model_path, encoders_path, features_path, new_data):
    """
    Function to make predictions on new data
    Note: new_data should already have the engineered features (nb_used, nb_delta)
    or the original depth_nb1 and depth_nb2 columns for feature engineering
    """
    # Load model and encoders
    model = joblib.load(model_path)
    label_encoders = joblib.load(encoders_path)
    feature_names = joblib.load(features_path)
    
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
    
    return predictions

# Main execution
if __name__ == "__main__":
    # Set your Excel file path here
    excel_file_path = "features_dataset.xlsx"  # Replace with your actual file path
    
    try:
        # Load and preprocess data
        X, y, label_encoders, feature_names = load_and_preprocess_data(excel_file_path)
        
        # Perform feature selection to reduce overfitting
        X_selected, selected_features, selected_encoders = feature_selection(
            X, y, label_encoders, feature_names
        )
        
        print(f"\nDataset after feature selection: {X_selected.shape}")
        
        # Train Random Forest model with selected features
        # Use optimize_hyperparameters=True for better regularization
        model, X_train, X_test, y_train, y_test, y_pred_test = train_random_forest(
            X_selected, y, optimize_hyperparameters=True
        )
        
        # Analyze feature importance
        feature_importance_dict = analyze_feature_importance(model, selected_features)
        
        # Create prediction plots
        create_prediction_plots(y_test, y_pred_test)
        
        # Save model and encoders
        save_model_and_encoders(model, selected_encoders, selected_features)
        
        # Example of how to use the model for new predictions
        print("\n=== Example: Making predictions on new data ===")
        print("To make predictions on new data, use:")
        print("predictions = predict_new_data('random_forest_depth_diff_model.pkl',")
        print("                               'label_encoders.pkl',")
        print("                               'feature_names.pkl',")
        print("                               new_dataframe)")
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