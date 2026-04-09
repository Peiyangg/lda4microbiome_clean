# StripeSankey

Interactive Sankey diagrams for LDA microbiome analysis with sample tracing, metric overlays, and metadata coloring.

## Structure

```
StripeSankey/
├── lda4microbiome_backup/     # Core package
│   ├── __init__.py
│   ├── preprocessing.py       # TaxonomyProcessor
│   ├── training.py            # LDATrainer (Gensim/MALLET)
│   ├── selection.py           # SankeyDataProcessor
│   ├── metrics.py             # Perplexity, coherence, reconstruction
│   ├── visualization.py       # Plotly-based heatmaps
│   ├── stripesankey.py        # V1: Original StripeSankeyInline widget
│   ├── stripesankey_v2.py     # V2: Stacked ribbon flows, dimming, aligned tracing
│   ├── stripesankey_v3.py     # V3: V2 + proportional metadata sub-ribbon coloring
│   └── stripesankey_with_clusters.py
├── notebooks/                 # Marimo notebooks & utility scripts
│   ├── stripesankey_pig_poster.py   # Main visualization notebook (V1/V2/V3)
│   ├── interactive_sankey.py        # Interactive cluster adjustment notebook
│   ├── filter_sankey_by_k.py        # Filter sankey JSON to specific K values
│   ├── weighted_soft_clustering_v2.py  # Consensus clustering (JSD + HDBSCAN)
│   ├── add_cluster_trajectories.py  # Add cluster data to sankey JSON
│   └── create_sankey_with_cluster_trajectories.py  # Full clustering workflow
├── data/                      # Input data
│   ├── ASV_count.csv
│   ├── new_taxa(only7).csv
│   └── metadata2comparing.csv
├── lda_output/                # Pre-computed results
│   └── lda_results/
│       └── fully_integrated_sankey_data_improved.json
└── pyproject.toml
```

## Widget Versions

| Version | Class | Key Features |
|---------|-------|--------------|
| V1 | `StripeSankeyInline` | Original: stroked-line flows, center-point connections |
| V2 | `StripeSankeyV2` | Filled ribbon flows stacked along segment edges, aligned sample tracing, dimming on click, ribbon cluster trajectories |
| V3 | `StripeSankeyV3` | V2 + proportional metadata sub-ribbon coloring with legend |

## Quick Start

```python
import json
from lda4microbiome_backup import StripeSankeyV3

with open('lda_output/lda_results/fully_integrated_sankey_data_improved.json') as f:
    sankey_data = json.load(f)

widget = StripeSankeyV3(sankey_data=sankey_data, width=1400, height=700, mode="metrics")

# Optional: color flows by metadata
import pandas as pd
metadata = pd.read_csv('data/metadata2comparing.csv', index_col=0)
widget.set_metadata_coloring(sankey_data, metadata, "Country")
```

## Author

Peiyang Huo (peiyang.huo@kuleuven.be)
