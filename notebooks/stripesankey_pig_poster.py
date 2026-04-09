import marimo

__generated_with = "0.22.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import os

    from lda4microbiome_backup import (
        StripeSankeyInline,
        StripeSankeyV2,
        StripeSankeyV3,
    )

    return StripeSankeyInline, StripeSankeyV2, StripeSankeyV3, json, mo, os, pd


@app.cell
def _(json, os):
    # Load pre-computed sankey data (produced by the training notebook)
    sankey_path = os.path.join(
        os.getcwd(), 'lda_output', 'lda_results',
        'fully_integrated_sankey_data_improved.json'
    )
    with open(sankey_path, 'r') as _f:
        sankey_data = json.load(_f)
    return (sankey_data,)


@app.cell
def _(StripeSankeyInline, mo, sankey_data):
    # Display original StripeSankey (v1)
    widget_v1 = StripeSankeyInline(
        sankey_data=sankey_data,
        width=1400,
        height=700,
        mode="metrics",
        min_flow_samples=5,
    )
    mo.ui.anywidget(widget_v1)
    return


@app.cell
def _(StripeSankeyV2, mo, sankey_data):
    # Display improved StripeSankey (v2 - stacked ribbon flows)
    widget_v2 = StripeSankeyV2(
        sankey_data=sankey_data,
        width=1400,
        height=700,
        mode="metrics",
        min_flow_samples=5,
    )
    mo.ui.anywidget(widget_v2)
    return


@app.cell
def _(pd):
    metadata = pd.read_csv('data/metadata2comparing.csv', index_col=0)
    return (metadata,)


@app.cell
def _(StripeSankeyV3, metadata, mo, sankey_data):
    # Display V3: flows colored by metadata majority (optional)
    widget_v3 = StripeSankeyV3(
        sankey_data=sankey_data,
        width=1400,
        height=700,
        mode="metrics",
        min_flow_samples=5,
    )
    # Color flows by Country (majority vote per flow)
    _color_map = widget_v3.set_metadata_coloring(sankey_data, metadata, "Breed_type")
    mo.ui.anywidget(widget_v3)
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
