import os
import pickle
import dill
import pandas as pd
import numpy as np

def read_df(path):
    """Read a CSV into a DataFrame with consistent encoding and replacements."""
    return pd.read_csv(path, verbose=False, encoding='ISO-8859-1')


def read_pickle(path):
    """Try to load an object with pickle or dill. Accept `path` with or without extension."""
    # allow passing directory-like names saved without extension
    if not os.path.exists(path) and os.path.exists(path + '.pkl'):
        path = path + '.pkl'
    if not os.path.exists(path):
        raise FileNotFoundError(f"Pickle path not found: {path}")

    # Try dill first (more robust), else try pickle
    with open(path, 'rb') as f:
        try:
            return dill.load(f)
        except Exception:
            f.seek(0)
            try:
                return pickle.load(f)
            except Exception as exc:
                raise RuntimeError(f"Failed to load pickle/dill from {path}: {exc}")


def predict_with_api_or_local(client_id, X_df, api_url=None, classifier=None, preprocessor=None, timeout=5):
    """
    Try to get prediction from API. If it fails, and classifier+preprocessor are provided,
    compute local probability using classifier.predict_proba.

    Returns probability (float between 0 and 1).
    """
    # Try API if provided
    if api_url:
        import requests
        try:
            data_json = {"id": int(client_id)}
            headers = {"Content-Type": "application/json"}
            response = requests.post(f"{api_url}/predict", json=data_json, headers=headers, timeout=timeout)
            response.raise_for_status()
            content = response.json()
            # New API format returns {"credit_score": float, "advice": str}
            if isinstance(content, dict) and "credit_score" in content:
                return float(content["credit_score"])
        except Exception:
            # swallow and fallback to local if available
            pass

    # Local fallback
    if classifier is None or preprocessor is None:
        raise RuntimeError("No API response and no local model available for prediction")

    # Prepare X: remove SK_ID_CURR or TARGET if present
    X = X_df.copy()
    for col in ['SK_ID_CURR', 'TARGET']:
        if col in X.columns:
            X = X.drop(columns=[col])

    # Preprocess then predict
    try:
        X_trans = preprocessor.transform(X)
    except Exception as exc:
        raise RuntimeError(f"Preprocessor failed: {exc}")

    try:
        proba = None
        # scikit-learn like
        if hasattr(classifier, 'predict_proba'):
            proba = classifier.predict_proba(X_trans)[0][1]
        elif hasattr(classifier, 'predict'):
            # if only predict exists, assume returns probability-like score
            proba = float(classifier.predict(X_trans)[0])
        else:
            raise RuntimeError('Classifier has no predict_proba nor predict')
        return float(proba)
    except Exception as exc:
        raise RuntimeError(f"Classifier prediction failed: {exc}")
