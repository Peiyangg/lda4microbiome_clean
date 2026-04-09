"""Interactive Sankey with adjustable cluster size"""

import marimo

__generated_with = "0.17.0"
app = marimo.App(width="full")


@app.cell
def _():
    import marimo as mo
    import pandas as pd
    import json
    import sys
    sys.path.append('.')
    from weighted_soft_clustering_v2 import load_lda_topic_probabilities, WeightedSoftClustering
    from add_cluster_trajectories import add_cluster_trajectories_to_sankey
    from lda4microbiome_backup.stripesankey import StripeSankeyInline
    return (
        StripeSankeyInline,
        WeightedSoftClustering,
        add_cluster_trajectories_to_sankey,
        json,
        load_lda_topic_probabilities,
        mo,
        pd,
    )


@app.cell
def _(mo):
    mo.md(
        """
    # Interactive Sankey: k=[3,4,5] with Adjustable Clusters

    This notebook lets you **interactively adjust** the number of clusters using the slider below.
    """
    )
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 🎛️ Adjust Parameters

    Use the slider to control the **minimum cluster size**:
    - **Higher values** (40-50) → Fewer, larger clusters (~3-4)
    - **Lower values** (15-25) → More, smaller clusters (~7-9)
    - **Recommended**: 30 for ~5 balanced clusters
    """
    )
    return


@app.cell
def _(mo):
    # Create slider for min_cluster_size
    min_cluster_size_slider = mo.ui.slider(
        start=10,
        stop=60,
        step=5,
        value=30,
        label="Minimum Cluster Size:",
        show_value=True
    )
    min_cluster_size_slider
    return (min_cluster_size_slider,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 📊 Load Data & Build Clusters

    Loading topic probabilities and building consensus tree...
    """
    )
    return


@app.cell
def _(load_lda_topic_probabilities):
    # Load topic probabilities for k=[3,4,5]
    selected_k = [3, 4, 5]
    topic_probs = load_lda_topic_probabilities('GENSIM', k_list=selected_k)
    print(f"✓ Loaded k={selected_k}")
    return selected_k, topic_probs


@app.cell
def _(WeightedSoftClustering, topic_probs):
    # Build consensus tree (this is cached, only runs once)
    tree = WeightedSoftClustering(topic_probs, verbose=False)
    print("✓ Consensus tree built")
    return (tree,)


@app.cell
def _(min_cluster_size_slider, mo, pd, tree):
    # Extract clusters with current slider value
    min_size = min_cluster_size_slider.value

    clusters = tree.get_clusters_from_tree(min_cluster_size=min_size)
    cluster_labels = pd.Series(clusters['sample_to_label'])

    # Display cluster statistics
    mo.md(f"""
    ### Clustering Results (min_size={min_size})

    - **Clusters found**: {clusters['n_clusters']}
    - **Total samples**: {len(cluster_labels)}
    - **Noise points**: {(cluster_labels == -1).sum()} ({clusters['noise_ratio']:.1%})
    """)
    return cluster_labels, clusters


@app.cell
def _(clusters, mo, pd):
    # Show cluster sizes table
    cluster_sizes_data = []
    for i, size in enumerate(clusters['cluster_sizes']):
        pct = 100 * size / clusters['cluster_sizes'].sum()
        cluster_sizes_data.append({
            'Cluster': f"C{i}",
            'Size': int(size),
            'Percentage': f"{pct:.1f}%"
        })

    cluster_sizes_df = pd.DataFrame(cluster_sizes_data)

    mo.md("""
    #### Cluster Size Distribution:
    """)
    mo.ui.table(cluster_sizes_df)
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 🔄 Generate Sankey Data

    Creating filtered Sankey with cluster trajectories...
    """
    )
    return


@app.cell
def _(add_cluster_trajectories_to_sankey, cluster_labels, json, selected_k):
    # Add cluster trajectories to full Sankey
    sankey_with_traj = add_cluster_trajectories_to_sankey(
        sankey_json_path='GENSIM/lda_results/fully_integrated_sankey_data_improved.json',
        cluster_labels=cluster_labels,
        output_path='GENSIM/lda_results/temp_sankey_trajectories.json',
        alpha=0.7
    )

    # Filter to selected k values
    sankey_with_traj['k_range'] = sorted(selected_k)

    # Filter nodes
    filtered_nodes = {}
    for node_id, node_data in sankey_with_traj['nodes'].items():
        if node_id.startswith('K'):
            k_value = int(node_id.split('_')[0][1:])
            if k_value in selected_k:
                filtered_nodes[node_id] = node_data

    sankey_with_traj['nodes'] = filtered_nodes

    # Filter flows
    filtered_flows = []
    for flow in sankey_with_traj['flows']:
        if flow['source_k'] in selected_k and flow['target_k'] in selected_k:
            filtered_flows.append(flow)

    sankey_with_traj['flows'] = filtered_flows

    # Save
    with open('GENSIM/lda_results/temp_sankey_k345.json', 'w') as f:
        json.dump(sankey_with_traj, f, indent=2)

    print(f"✓ Filtered Sankey created: {len(filtered_nodes)} nodes, {len(filtered_flows)} flows")
    return (sankey_with_traj,)


@app.cell
def _(mo):
    mo.md(
        """
    ## 🎨 Sankey Visualization

    **Interactive Sankey with:**
    - **Nodes**: Colored by coherence score (blue intensity)
    - **Flows**: Colored by cluster membership
    - **Bars**: Perplexity scores (red bars above)

    Adjust the slider above to see different cluster groupings!
    """
    )
    return


@app.cell
def _(StripeSankeyInline, sankey_with_traj):
    # Create visualization
    sankey_viz = StripeSankeyInline(
        sankey_data=sankey_with_traj,
        mode='metrics',         # Coherence nodes + perplexity bars
        width=1400,
        height=900,
        show_clusters=True,     # Cluster-colored flows
        cluster_alpha=0.7
    )
    sankey_viz
    return


@app.cell
def _(mo):
    mo.md(
        """
    ## 💡 Tips

    - **Adjust the slider** at the top to change cluster count
    - **Hover** over nodes and flows for details
    - **Click flows** to highlight sample trajectories
    - **Higher min_cluster_size** → Fewer, more robust clusters
    - **Lower min_cluster_size** → More, smaller clusters

    ### Recommended Values:
    - `min_cluster_size = 50` → ~3 large clusters
    - `min_cluster_size = 30` → ~5 balanced clusters (default)
    - `min_cluster_size = 15` → ~9 small clusters
    """
    )
    return


@app.cell
def _():
    return


if __name__ == "__main__":
    app.run()
