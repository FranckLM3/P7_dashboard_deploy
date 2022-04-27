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


def b_boxplot(df, target:str, feature:str):
    df = df.loc[:, [feature, target]]
    # find the quartiles and IQR for each category
    groups = df.groupby(target)
    cats = df[target].unique()
    q1 = groups.quantile(q=0.25)
    q2 = groups.quantile(q=0.5)
    q3 = groups.quantile(q=0.75)
    iqr = q3 - q1
    upper = q3 + 1.5*iqr
    lower = q1 - 1.5*iqr

    # find the outliers for each category
    '''def outliers(group):
        cat = group.name
        return group[(group.loc[feature] > upper.loc[cat][feature]) | (group[feature] < lower.loc[cat][feature])][feature]
    out = groups.apply(outliers).dropna()
    # prepare outlier data for plotting, we need coordinates for every outlier.
    if not out.empty:
        outx = list(out.index.get_level_values(0))
        outy = list(out.values)'''

    p = figure(tools="", background_fill_color="#efefef", x_range=cats, toolbar_location=None)

    # if no outliers, shrink lengths of stems to be no longer than the minimums or maximums
    qmin = groups.quantile(q=0.00)
    qmax = groups.quantile(q=1.00)
    upper.loc[feature] = [min([x,y]) for (x,y) in zip(list(qmax.loc[:,feature]),upper.loc[feature])]
    lower.loc[feature] = [max([x,y]) for (x,y) in zip(list(qmin.loc[:,feature]),lower.loc[feature])]

    # stems
    p.segment(cats, upper.loc[feature], cats, q3.loc[feature], line_color="black")
    p.segment(cats, lower.loc[feature], cats, q1.loc[feature], line_color="black")

    # boxes
    p.vbar(cats, 0.7, q2.loc[feature], q3.loc[feature], fill_color="#E08E79", line_color="black")
    p.vbar(cats, 0.7, q1.loc[feature], q2.loc[feature], fill_color="#3B8686", line_color="black")

    # whiskers (almost-0 height rects simpler than segments)
    p.rect(cats, lower.loc[feature], 0.2, 0.01, line_color="black")
    p.rect(cats, upper.loc[feature], 0.2, 0.01, line_color="black")

    # outliers
    if not out.empty:
        p.circle(outx, outy, size=6, color="#F38630", fill_alpha=0.6)

    p.xgrid.grid_line_color = None
    p.ygrid.grid_line_color = "white"
    p.grid.grid_line_width = 2
    p.xaxis.major_label_text_font_size="16px"

    return p
