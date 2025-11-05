import os
import pickle
import dill
import joblib
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
shap.initjs()


def read_df(path):
    """Read a CSV into a DataFrame with consistent encoding and replacements."""
    return pd.read_csv(path, encoding='ISO-8859-1')


def read_pickle(path):
    """
    Load a serialized Python object from disk.

    - Accepts a path with or without extension
    - Tries common extensions: .pkl, .pickle, .joblib
    - Tries loaders in order: dill -> pickle -> joblib

    This makes it robust to artifacts saved with either pickle/dill or joblib.
    """
    # Resolve actual path: if given path doesn't exist, try common extensions.
    resolved_path = path
    if not os.path.exists(resolved_path):
        for ext in ('.pkl', '.pickle', '.joblib'):
            candidate = f"{path}{ext}"
            if os.path.exists(candidate):
                resolved_path = candidate
                break
    if not os.path.exists(resolved_path):
        raise FileNotFoundError(f"Pickle path not found: {path}")

    # Attempt to load with dill, then pickle, then joblib
    # Note: joblib can load many pickle files, but we keep it last to allow dill-specific objects first
    errors = []
    try:
        with open(resolved_path, 'rb') as f:
            return dill.load(f)
    except Exception as e:
        errors.append(f"dill: {e}")
    try:
        with open(resolved_path, 'rb') as f:
            return pickle.load(f)
    except Exception as e:
        errors.append(f"pickle: {e}")
    try:
        return joblib.load(resolved_path)
    except Exception as e:
        errors.append(f"joblib: {e}")

    raise RuntimeError(
        f"Failed to load object from {resolved_path}. Tried dill, pickle, joblib. Errors: {errors}"
    )


def load_shap_explainer(path, classifier, save_rebuilt: bool = False):
    """
    Load a SHAP TreeExplainer from a serialized artifact.
    If loading fails due to environment/class version mismatch, rebuild the explainer
    from the provided classifier. Optionally persist the rebuilt explainer.

    Args:
        path: Base path (with or without extension) to the explainer artifact.
        classifier: Fitted model used to rebuild the TreeExplainer if needed.
        save_rebuilt: If True, save the rebuilt explainer back to disk using dill.

    Returns:
        A SHAP TreeExplainer instance.
    """
    try:
        explainer = read_pickle(path)
        return explainer
    except Exception as load_err:
        # Rebuild from classifier to avoid pickling issues across SHAP versions
        try:
            import shap  # local import to ensure availability
            explainer = shap.TreeExplainer(classifier)
            if save_rebuilt:
                # Save using dill to a .pkl alongside original base path
                out_path = path
                if not os.path.exists(out_path) and not out_path.endswith(('.pkl', '.pickle', '.joblib')):
                    out_path = f"{path}.pkl"
                try:
                    with open(out_path, 'wb') as f:
                        dill.dump(explainer, f)
                except Exception:
                    # Saving is best-effort; ignore persistence errors
                    pass
            return explainer
        except Exception as rebuild_err:
            raise RuntimeError(
                f"Failed to load or rebuild SHAP explainer from {path}. Load error: {load_err} | Rebuild error: {rebuild_err}"
            )


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


def plot_gauge(prediction_default):
    # Determine color based on risk level
    if prediction_default < 30:
        bar_color = '#2ecc71'  # Green
    elif prediction_default < 50:
        bar_color = '#f39c12'  # Orange
    else:
        bar_color = '#e74c3c'  # Red
    
    fig_gauge = go.Figure(go.Indicator(
        domain = {'x': [0, 1], 'y': [0, 1]},
        value = prediction_default,
        mode = "gauge+number",
        title = {'text': "Risque de défaut (%)", 'font': {'size': 20, 'color': '#2c3e50'}},
        number = {'font': {'size': 40, 'color': '#2c3e50'}},
        gauge = {
            'axis': {
                'range': [None, 100],
                'tick0': 0,
                'dtick': 10,
                'tickwidth': 2,
                'tickcolor': '#2c3e50',
                'tickfont': {'color': '#2c3e50', 'size': 14},
                'showticklabels': True
            },
            'bar': {
                'color': bar_color,
                'thickness': 0.35,
                'line': {'color': 'white', 'width': 2}
            },
            'bgcolor': 'white',
            'borderwidth': 2,
            'bordercolor': '#ecf0f1',
            'steps': [
                {'range': [0, 30], 'color': "rgba(46, 204, 113, 0.2)"},   # Light green
                {'range': [30, 50], 'color': "rgba(243, 156, 18, 0.2)"},  # Light orange
                {'range': [50, 100], 'color': "rgba(231, 76, 60, 0.2)"}   # Light red
            ],
            'threshold': {
                'line': {'color': bar_color, 'width': 4},
                'thickness': 0.75,
                'value': prediction_default
            }
        }
    ))

    fig_gauge.update_layout(
        height=400,
        margin={'l': 30, 'r': 40, 'b': 30, 't': 50},
        paper_bgcolor='white',
        font={'family': 'Arial, sans-serif'},
        autosize=True
    )
    
    # Add smooth CSS-based animation using Plotly's built-in transitions
    fig_gauge.update_traces(
        selector=dict(type='indicator'),
        # Smooth transition for gauge needle
    )

    return fig_gauge

def format_shap_values(shap_values, feature_names):
    """
    Format shap values into a dataframe to be plotted with Plotly.
    Returns the 15 most important shap values with colors and signs.
    """

    # Si shap_values est 2D (par exemple shap_values.shape = (n_samples, n_features))
    # on prend la moyenne absolue par feature
    if len(shap_values.shape) > 1:
        shap_values_mean = np.abs(shap_values).mean(axis=0)
        shap_values_single = shap_values.mean(axis=0)
    else:
        shap_values_mean = np.abs(shap_values)
        shap_values_single = shap_values

    # Crée le DataFrame avec les vrais noms de features
    df = pd.DataFrame({
        "features": feature_names,
        "shap_values": shap_values_single,
        "absolute_values": shap_values_mean
    })

    # Trie par importance
    df.sort_values(by="absolute_values", ascending=False, inplace=True)
    df.reset_index(drop=True, inplace=True)

    # Ajout des colonnes de signe et couleur
    df["left"] = df["shap_values"].where(df["shap_values"] < 0, 0)
    df["right"] = df["shap_values"].where(df["shap_values"] > 0, 0)
    df["color"] = np.where(df["shap_values"] > 0, "#D73027", "#1A9851")

    # Sélectionne les 15 plus importants
    shap_explained = df.head(15).copy()

    # Liste des features dans l'ordre inverse pour affichage
    most_important_features = shap_explained["features"].iloc[::-1].tolist()

    return shap_explained, most_important_features

def plot_important_features(shap_explained, most_important_features):
    """
    Create a Plotly horizontal bar chart for SHAP feature importance.
    Red bars = increase risk, Green bars = decrease risk
    """
    # Reverse order for plotting (most important at top)
    shap_plot = shap_explained.iloc[::-1].copy()
    
    # Create Plotly figure with custom colors for each bar
    fig = go.Figure()
    
    # Add bars with individual colors
    for idx, row in shap_plot.iterrows():
        fig.add_trace(go.Bar(
            x=[row['shap_values']],
            y=[row['features']],
            orientation='h',
            marker=dict(color=row['color']),
            hovertemplate='<b>%{y}</b><br>Impact: %{x:.4f}<extra></extra>',
            showlegend=False
        ))
    
    # Update layout
    fig.update_layout(
        title={
            'text': "Facteurs les plus importants dans la décision de l'algorithme",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 16, 'color': '#2c3e50'}
        },
        xaxis_title="Impact sur la sortie du modèle",
        yaxis_title="Informations du client",
        height=500,
        margin=dict(l=200, r=50, t=80, b=50),
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(
            gridcolor='lightgray',
            zeroline=True,
            zerolinecolor='black',
            zerolinewidth=2,
            tickfont=dict(color='#333', size=12),
            titlefont=dict(color='#2c3e50', size=14)
        ),
        yaxis=dict(
            gridcolor='lightgray',
            tickfont=dict(color='#333', size=12),
            titlefont=dict(color='#2c3e50', size=14)
        )
    )
    
    return fig


def plot_feature_distrib(feature_distrib, client_line, hist_source, data_client_value, max_histogram):
    """
    Create a Plotly histogram showing feature distribution with client's position.
    """
    # Extract histogram data
    hist_df = hist_source.data if hasattr(hist_source, 'data') else hist_source
    
    # Create the bar chart for histogram
    fig = go.Figure()
    
    # Add histogram bars
    fig.add_trace(go.Bar(
        x=[(hist_df['edges_left'][i] + hist_df['edges_right'][i]) / 2 
           for i in range(len(hist_df['edges_left']))],
        y=hist_df['hist'],
        width=[hist_df['edges_right'][i] - hist_df['edges_left'][i] 
               for i in range(len(hist_df['edges_left']))],
        marker=dict(
            color='steelblue',
            opacity=0.7,
            line=dict(color='white', width=1)
        ),
        hovertemplate='<b>Plage :</b> %{customdata[0]:.2f} à %{customdata[1]:.2f}<br><b>Nombre :</b> %{y}<extra></extra>',
        customdata=[[hist_df['edges_left'][i], hist_df['edges_right'][i]] 
                    for i in range(len(hist_df['edges_left']))],
        name='Distribution'
    ))
    
    # Add client's value line
    fig.add_trace(go.Scatter(
        x=[data_client_value[0], data_client_value[0]],
        y=[0, max_histogram],
        mode='lines',
        line=dict(color='orange', width=3, dash='dash'),
        name="Valeur du client",
        hovertemplate='<b>Valeur du client :</b> %{x:.2f}<extra></extra>'
    ))
    
    # Add annotation for client's value
    fig.add_annotation(
        x=data_client_value[0],
        y=max_histogram * 1.05,
        text="Valeur du client",
        showarrow=True,
        arrowhead=2,
        arrowcolor='orange',
        font=dict(color='orange', size=12),
        bgcolor='white',
        bordercolor='orange',
        borderwidth=2
    )
    
    # Update layout
    fig.update_layout(
        title={
            'text': f"Valeur du client pour {feature_distrib} comparée aux autres clients",
            'x': 0.5,
            'xanchor': 'center',
            'font': {'size': 14, 'color': '#2c3e50'}
        },
        xaxis_title=feature_distrib,
        yaxis_title="Nombre de clients",
        height=580,
        showlegend=True,
        hovermode='closest',
        plot_bgcolor='white',
        paper_bgcolor='white',
        xaxis=dict(gridcolor='lightgray'),
        yaxis=dict(gridcolor='lightgray', rangemode='tozero'),
        legend=dict(
            bgcolor='rgba(255,255,255,0.9)',
            borderwidth=1,
            font=dict(color='#333', size=12),
            yanchor='top'
        )
    )
    
    return fig

