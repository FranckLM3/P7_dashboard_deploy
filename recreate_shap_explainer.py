"""
Script to recreate SHAP explainer compatible with current libraries
Based on the original notebook methodology
"""

import numpy as np
import pandas as pd
import joblib
import pickle
import shap
from sklearn.model_selection import train_test_split

print("Loading model artifacts...")



pipeline = joblib.load('ressource/pipeline.joblib')
print(f"New unified pipeline loaded (pipeline.joblib)")
use_new_pipeline = True

# Extract the classifier from the pipeline
model = pipeline.named_steps['classifier']
print(f"Classifier extracted: {type(model)}")

# Get feature names - we'll extract them after transformation
# Since we don't know the pipeline structure, we'll extract feature names after transformation
print("Will extract feature names after transformation...")

# Load the sample dataset
print("\nLoading dataset...")
df = pd.read_csv('data/dataset_sample.csv', low_memory=False)
print(f"Dataset loaded: {df.shape}")

# Prepare data
print("\nPreparing data...")
df = df.replace([np.inf, -np.inf], np.nan)
df_train, df_test = train_test_split(df, test_size=0.2, random_state=5)

X_train = df_train.drop(['TARGET', 'SK_ID_CURR'], axis=1)
y_train = df_train['TARGET']

print(f"X_train shape: {X_train.shape}")

# Transform using pipeline (without the classifier step)
print("\nTransforming data with pipeline...")
X_train_transformed = pipeline[:-1].transform(X_train)  # Exclude classifier
print(f"X_train_transformed shape: {X_train_transformed.shape}")

# Extract feature names from the pipeline - manually build them
print("\nExtracting feature names...")
try:
    # Create a dummy DataFrame to get feature names
    import pandas as pd
    X_train_df = pd.DataFrame(X_train, columns=X_train.columns)
    # Transform and check sklearn pipeline for feature names
    # For sklearn 1.6+, we can try to get feature names from the pipeline
    try:
        feats_list = []
        for name, transform in pipeline[:-1].named_steps.items():
            if hasattr(transform, 'get_feature_names_out'):
                try:
                    feats_list = list(transform.get_feature_names_out())
                    print(f"  ✓ Got {len(feats_list)} features from {name}")
                except Exception as e:
                    print(f"Could not get features from {name}: {e}")
        
        if not feats_list:
            # Fallback: generate generic feature names
            n_features = X_train_transformed.shape[1]
            feats_list = [f"feature_{i}" for i in range(n_features)]
            print(f"Using generic names for {n_features} features")
        
        feats = feats_list
        print(f"✓ Feature names: {len(feats)} features")
        
        # Save feats
        with open('ressource/feats', 'wb') as f:
            pickle.dump(feats, f)
        print(f"✓ Features saved to ressource/feats")
        
    except Exception as e:
        print(f"Error extracting feature names: {e}")
        # Use generic names
        n_features = X_train_transformed.shape[1]
        feats = [f"feature_{i}" for i in range(n_features)]
        print(f"  Using {len(feats)} generic feature names")
        # Save feats
        with open('ressource/feats', 'wb') as f:
            pickle.dump(feats, f)
        print(f"✓ Features saved to ressource/feats")
except Exception as e:
    print(f"Fatal error extracting features: {e}")
    exit(1)

# Sample data for SHAP (as in original notebook)
print("\nCreating SHAP explainer...")
X_train_sample = shap.sample(X_train_transformed, 100)
print(f"X_train_sample shape: {X_train_sample.shape}")

# Use TreeExplainer for LightGBM (faster and more accurate than KernelExplainer)
# NOTE: TreeExplainer works directly with tree-based models
print("Using TreeExplainer (optimized for LightGBM)...")
try:
    SHAP_explainer = shap.TreeExplainer(model)
    print(f"SHAP TreeExplainer created: {type(SHAP_explainer)}")
except Exception as e:
    print(f"TreeExplainer failed: {e}")
    print("Falling back to KernelExplainer...")
    # Wrap the model to avoid attribute error
    def model_wrapper(X):
        return model.predict_proba(X)[:, 1]
    SHAP_explainer = shap.KernelExplainer(model_wrapper, X_train_sample)
    print(f"SHAP KernelExplainer created: {type(SHAP_explainer)}")

# Export SHAP explainer
print("\nSaving SHAP explainer...")
with open('ressource/shap_explainer_new', 'wb') as file:
    pickle.dump(SHAP_explainer, file)

print("✓ SHAP explainer saved to ressource/shap_explainer_new")

# Test loading it back
print("\nTesting reload...")
with open('ressource/shap_explainer_new', 'rb') as file:
    test_explainer = pickle.load(file)
print(f"Successfully reloaded: {type(test_explainer)}")

print("\nSHAP explainer recreation complete!")
