# lda4microbiome

A Python library for exploratory data analysis of microbiome data using Latent Dirichlet Allocation (LDA). The library provides a complete pipeline — from raw sequence count tables to interactive visualisations — designed for microbiome researchers.

## Overview

LDA is recognised as a "topic modelling" technique in natural language processing, but it serves as both dimensionality reduction and clustering when applied to microbial datasets. Its underlying Dirichlet distribution naturally handles the compositional nature of microbiome data. In this library, LDA topics are called **Microbial Components (MCs)** — each MC is a distribution over microbial features, and each sample is modelled as a mixture of MCs.

Two practical challenges motivated this library:

1. **Choosing *k***: LDA requires the number of MCs as input. Metrics like perplexity and coherence can guide the choice, but manual inspection of how samples behave across candidate models is still essential.
2. **Interpreting results**: Microbiome researchers need visualisations that connect LDA outputs to domain knowledge — metadata, taxonomy, and sample groupings.

`lda4microbiome` addresses both through a novel interactive visualisation called the **StripeSankey diagram** and a set of annotated heatmaps.

## Installation

```bash
pip install lda4microbiome
```

Source code and example notebooks are available at:
[https://gitlab.kuleuven.be/aida-lab/projects/LDA4Microbiome_Workflow](https://gitlab.kuleuven.be/aida-lab/projects/LDA4Microbiome_Workflow)

## Workflow

```
ASV_count.csv + taxonomy.csv + metadata.csv
        ↓  Step 1: TaxonomyProcessor
    Preprocessed intermediate files
        ↓  Step 2: LDATrainer (Gensim VB or MALLET MCMC)
    Per-k model outputs (MC–sample probability matrices, metrics)
        ↓  Step 3: SankeyDataProcessor → StripeSankeyInline
    Interactive StripeSankey diagram (model selection)
        ↓  Step 4: LDAModelVisualizerInteractive
    Annotated heatmaps, stacked bar charts, MC–feature heatmap
```

## Quick Start

```python
from lda4microbiome import (
    TaxonomyProcessor,
    LDATrainer,
    SankeyDataProcessor,
    StripeSankeyInline,
    LDAModelVisualizerInteractive,
    MCComparison,
)
```

## Step-by-Step Usage

### Step 1 — Data Transformation

The `TaxonomyProcessor` converts raw ASV count data and taxonomy tables into LDA-compatible format. Inputs must be CSV files:

- **ASV table**: rows = samples, columns = ASV features, values = raw counts (not relative abundances)
- **Taxonomy table**: rows = ASV IDs, columns = taxonomic levels
- **Output directory**: an existing folder for storing all generated files

```python
processor = TaxonomyProcessor(
    asvtable_path="data/ASV_count.csv",
    taxonomy_path="data/new_taxa.csv",
    base_directory="lda_output/"
)
results = processor.process_all()
```

### Step 2 — Training LDA Models

`LDATrainer` supports two implementations:

- **VB (variational Bayesian)** via `gensim` — faster, good for exploration
- **MCMC** via `MALLET` — slower but often more accurate; requires downloading MALLET separately

Models are trained across a range of *k* values (default: 2–20). Results are saved to the output folder.

```python
trainer = LDATrainer(
    base_directory="lda_output/",
    implementation="MCMC",           # or "VB"
    path_to_mallet="path/to/mallet"  # required for MCMC only
)
lda_results = trainer.train_models(MC_range=range(2, 11))
```

**Training time note**: On a dataset with 289 samples and 6,899 features, MCMC takes ~4.5 minutes per model; VB takes ~1 minute.

### Step 3 — Model Selection with StripeSankey

Process training results into the StripeSankey format, then launch the interactive widget to explore all candidate models at once.

```python
processor_sankey = SankeyDataProcessor.from_lda_trainer(trainer)
sankey_data = processor_sankey.process_all_data()

widget = StripeSankeyInline(
    sankey_data=sankey_data,
    width=1400,
    height=700,
    mode="metrics"
)
widget  # display in a marimo notebook
```

#### What the StripeSankey shows

The diagram has two parts:

- **Perplexity bar chart** (above): one bar per *k* value. Shorter / brighter red bars indicate better model fit.
- **Sankey diagram** (below): columns correspond to *k* values. Each node (stacked bar) represents one MC.
  - Node height = number of samples assigned to that MC.
  - Node colour = MC coherence score (blue colourmap; brighter = higher quality co-occurrence patterns).
  - Each node is split into two segments: **upper** = high-probability samples (≥ 0.67), **lower** = medium-probability samples (≥ 0.33). Low-probability samples are omitted.
  - Flows between adjacent columns show how samples transition as *k* increases; flow width = sample count.

**Interactivity**:
- Hover over a segment → tooltip with sample count, perplexity, and coherence score.
- Click a flow → that flow is highlighted in orange and sample counts appear next to the corresponding nodes, tracing exactly which samples moved where.

#### Two flow patterns to watch for

As *k* increases, flows typically exhibit one of two behaviours:

1. **Stable flow** — a group of samples is consistently represented by the same MC across candidate models. This indicates a robust, well-separated cluster.
2. **Splitting / joining flow** — a group splits into two MCs (or two merge into one). This can represent hierarchical topic decomposition and warrants further investigation.

#### Investigating splitting flows with MCComparison

```python
mc_comp = MCComparison(
    base_directory="lda_output/",
    metadata_path="data/metadata.csv"
)
mc_comp.compare_two_mcs("K4_MC0", "K4_MC1", metadata=["Country", "Breed_type"])
```

This compares the top features of two MCs and produces grouped bar charts for the specified metadata columns, helping you decide whether a split reflects genuine biological heterogeneity or is an artefact.

### Step 4 — Result Interpretation

Once a *k* value is chosen, `LDAModelVisualizerInteractive` generates a full set of visualisations saved as SVG, PNG, and HTML.

```python
viz = LDAModelVisualizerInteractive(
    base_directory="lda_output/",
    k_value=4,
    metadata_path="data/metadata.csv",
    universal_headers=["Country", "Breed_type"],   # categorical metadata columns
    continuous_headers=["Age"]                      # numerical metadata columns
)
viz_results = viz.create_all_visualizations_interactive()
```

**Visualisations produced**:

| Plot | Description |
|------|-------------|
| **Sample–MC heatmap** | Annotated heatmap with metadata bars and dendrogram. Use hover to inspect individual samples and their metadata. Use the dendrogram to select precise sample groups. |
| **Stacked bar charts** | Samples grouped by metadata; bars show MC mixture proportions. Useful for comparing enterotype-like groups. |
| **MC–feature heatmap** | Interactive heatmap of the MC × ASV matrix. Hover tooltips include taxonomy annotations at all levels and per-MC probabilities. |

## Package Structure

```
lda4microbiome/
├── preprocessing.py      # TaxonomyProcessor: ASV table + taxonomy → LDA-ready format
├── training.py           # LDATrainer: Gensim VB and MALLET MCMC training
├── selection.py          # SankeyDataProcessor: model outputs → StripeSankey JSON
├── metrics.py            # Perplexity, coherence (Gensim c_v, MALLET XML), reconstruction
├── visualization.py      # LDAModelVisualizerInteractive, MCComparison, TopicFeatureProcessor
└── stripesankey.py       # StripeSankeyInline: interactive anywidget for marimo notebooks
```

## Dependencies

**Widget only** (for displaying pre-computed results):
- `anywidget`, `traitlets` (d3.js loaded via CDN)

**Full pipeline**:
- `gensim`, `pandas`, `numpy`, `scipy`, `scikit-learn`
- `little_mallet_wrapper` (for MCMC training)
- `plotly`, `kaleido`, `matplotlib`, `seaborn`

**Notebooks**:
- `marimo`

## Citation

If you use `lda4microbiome` in your research, please cite:

> Peiyang Huo, Luke Comer, Pablo Vargas, Hans Rediers, Nadia Everaert, Jan Aerts.
> *lda4microbiome: a Python library for exploratory data analysis of microbiome data using latent Dirichlet allocation.*
> Bioinformatics (Application Note).

## Authors

- **Peiyang Huo** (peiyang.huo@kuleuven.be) — AIDA Lab, KU Leuven
- Luke Comer — NAMES Lab, KU Leuven
- Pablo Vargas — IRTA Cabrils
- Hans Rediers — KU Leuven
- Nadia Everaert — KU Leuven
- Jan Aerts — AIDA Lab / Leuven.AI, KU Leuven (corresponding author)
