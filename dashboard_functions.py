import pandas as pd
import numpy as np
import plotly.graph_objects as go
import shap
shap.initjs()

from bokeh.models import HoverTool, ColumnDataSource
from bokeh.plotting import figure, show
from bokeh.models.annotations import Label


def plot_gauge(prediction_default):
    fig_gauge = go.Figure(go.Indicator(
    domain = {'x': [0, 1], 'y': [0, 1]},
    value = prediction_default,
    mode = "gauge+number",
    title = {'text': "Risk of default (%)"},
    gauge = {'axis': {'range': [None, 100],
                    'tick0': 0,
                    'dtick':10},
            'bar': {'color': 'blue',
                    'thickness': 0.3,
                    'line': {'color': 'black'}},
            'steps' : [{'range': [0, 29.8], 'color': "green"},
                        {'range': [30.2, 49.8], 'color': "orange"},
                        {'range': [50.2, 100], 'color': "red"}]}        
                        ))

    fig_gauge.update_layout(width=600, 
                            height=400,
                            margin= {'l': 30, 'r': 30, 'b': 30, 't':30})

    return fig_gauge

def format_shap_values(shap_values, feature_names):
    """
    Format shap values into a dataframe to be plotted with Bokeh.
    Return dataframe with first 15 most important shap values, left and right values and color for Bokeh plot.
    """
    # Formatting df
    df = pd.DataFrame(shap_values, index=feature_names).reset_index()
    df.rename(columns={"index": "features", 0:"shap_values"}, inplace=True)
    df["absolute_values"]=abs(df["shap_values"])
    df.sort_values(by="absolute_values", ascending=False, inplace=True)
    df.reset_index(inplace=True)
    df.drop("index", axis=1, inplace=True)

    # Getting left and right from shap
    df["left"] = df["shap_values"].where(df["shap_values"]<0, 0)
    df["right"] = df["shap_values"].where(df["shap_values"]>0, 0)

    # Color depending on sign
    df["color"] = np.where(df["shap_values"]>0, "#D73027", "#1A9851")

    # Select first 15
    shap_explained = df.loc[0:14, ["features", "shap_values", "left", "right", "color"]]
    shap_explained.reset_index(inplace=True)

    # Make list of most important features (inversed for Bokeh)
    most_important_features = shap_explained["features"].tolist()
    most_important_features = most_important_features[::-1]

    return shap_explained, most_important_features

def plot_important_features(shap_explained, most_important_features):

    explained_plot = figure(y_range=most_important_features, title="Most important data in the algorithm decision")

    source = ColumnDataSource(data=shap_explained)
    bars = explained_plot.hbar(y="features", left="left", right="right", height=0.5, color="color", 
                                hover_line_color="black", hover_line_width=2, source=source)

    explained_plot.xaxis.axis_label = "Impact on model output"
    explained_plot.yaxis.axis_label = "Client's informations"
    explained_plot.add_tools(HoverTool(tooltips=[("Importance", "@shap_values")], 
                                renderers = [bars]))
    return explained_plot


def plot_feature_distrib(feature_distrib, client_line, hist_source, data_client_value, max_histogram):
    distrib = figure(title=f"Client value for {feature_distrib} compared to other clients", 
                    plot_width=1000, plot_height=500)
    qr = distrib.quad(top="hist", bottom=0, line_color="white", left="edges_left", right="edges_right",
        fill_color="steelblue", hover_fill_color="orange", alpha=0.5, hover_alpha=1, source=hist_source)

    distrib.line(x=client_line["x"], y=client_line["y"], line_color="orange", line_width=2, line_dash="dashed")
    label_client = Label(text="Client's value", x=data_client_value[0], y=max_histogram, text_color="orange",
                        x_offset=-50, y_offset=10)



    hover_tools = HoverTool(tooltips=[("Between:", "@edges_left"), ("and:", "@edges_right"), ("Count:", "@hist")], 
                        renderers = [qr])

    distrib.xaxis.axis_label = feature_distrib
    distrib.y_range.start = 0
    distrib.y_range.range_padding = 0.2
    distrib.yaxis.axis_label = "Number of clients"
    distrib.grid.grid_line_color="grey"
    distrib.xgrid.grid_line_color=None
    distrib.ygrid.grid_line_alpha=0.5

    distrib.add_tools(hover_tools)
    distrib.add_layout(label_client)

    return distrib
