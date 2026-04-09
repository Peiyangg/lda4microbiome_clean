"""
Add cluster membership as colored sample trajectories to Sankey visualization.

This keeps the existing topic-level Sankey structure and overlays cluster membership
as colored trajectories (like the click-to-trace feature), with each cluster having
a unique color with transparency.
"""

import json
import pandas as pd
from typing import Dict, List


def add_cluster_trajectories_to_sankey(
    sankey_json_path: str,
    cluster_labels: pd.Series,  # Series with index=sample names, values=cluster IDs
    output_path: str,
    alpha: float = 0.6
) -> Dict:
    """
    Add cluster trajectory information to Sankey data for visualization.
    
    Unlike the previous approach, this creates data for drawing all cluster
    trajectories simultaneously as colored overlays (similar to the orange
    sample tracing when clicking a flow).
    
    Parameters
    ----------
    sankey_json_path : str
        Path to fully_integrated_sankey_data_improved.json
    cluster_labels : pd.Series
        Cluster labels for each sample (from HDBSCAN)
        Index = sample names, Values = cluster IDs (-1 for noise)
    output_path : str
        Where to save enhanced JSON
    alpha : float
        Transparency for cluster trajectories (0-1)
    
    Returns
    -------
    dict : Enhanced Sankey data with cluster trajectory information
    """
    # Load original Sankey data
    with open(sankey_json_path, 'r') as f:
        sankey_data = json.load(f)
    
    # Generate cluster color palette
    n_clusters = int(cluster_labels[cluster_labels >= 0].nunique())
    cluster_colors = generate_cluster_palette(n_clusters, alpha=alpha)
    
    # Group samples by cluster
    samples_by_cluster = {}
    for sample, cluster_id in cluster_labels.items():
        cluster_id = int(cluster_id)
        if cluster_id not in samples_by_cluster:
            samples_by_cluster[cluster_id] = []
        samples_by_cluster[cluster_id].append(sample)
    
    # For each cluster, trace its samples through the Sankey
    cluster_trajectories = []
    
    for cluster_id, sample_list in samples_by_cluster.items():
        if cluster_id == -1:
            continue  # Skip noise points
        
        # Trace these samples through all flows to build trajectory data
        trajectory = {
            'cluster_id': cluster_id,
            'color': cluster_colors[cluster_id],
            'n_samples': len(sample_list),
            'samples': sample_list,
            # This will be used by the visualization to draw trajectories
            # just like the click-to-trace feature
        }
        
        cluster_trajectories.append(trajectory)
    
    # Add cluster visualization data to Sankey
    sankey_data['cluster_trajectories'] = cluster_trajectories
    sankey_data['cluster_colors'] = cluster_colors
    sankey_data['cluster_metadata'] = {
        'n_clusters': n_clusters,
        'n_noise': int((cluster_labels == -1).sum()),
        'cluster_sizes': {int(k): int(v) for k, v in 
                         cluster_labels[cluster_labels >= 0].value_counts().to_dict().items()},
        'alpha': alpha
    }
    
    # Save enhanced data
    with open(output_path, 'w') as f:
        json.dump(sankey_data, f, indent=2)
    
    print(f"✓ Enhanced Sankey data saved to: {output_path}")
    print(f"  - {n_clusters} clusters to visualize")
    print(f"  - {len(cluster_labels)} total samples")
    print(f"  - {sankey_data['cluster_metadata']['n_noise']} noise points (excluded)")
    print(f"  - Transparency: {alpha}")
    
    return sankey_data


def generate_cluster_palette(n_clusters: int, alpha: float = 0.6) -> Dict[int, str]:
    """
    Generate a color palette for clusters with transparency.
    
    Returns dict mapping cluster_id -> rgba color string
    """
    # Use a colorblind-friendly palette (distinct, high-contrast colors)
    base_colors_rgb = [
        (31, 119, 180),   # blue
        (255, 127, 14),   # orange
        (44, 160, 44),    # green
        (214, 39, 40),    # red
        (148, 103, 189),  # purple
        (140, 86, 75),    # brown
        (227, 119, 194),  # pink
        (127, 127, 127),  # gray
        (188, 189, 34),   # yellow-green
        (23, 190, 207),   # cyan
        (174, 199, 232),  # light blue
        (255, 187, 120),  # light orange
        (152, 223, 138),  # light green
        (255, 152, 150),  # light red
        (197, 176, 213),  # light purple
        (196, 156, 148),  # light brown
        (247, 182, 210),  # light pink
        (199, 199, 199),  # light gray
        (219, 219, 141),  # light yellow
        (158, 218, 229),  # light cyan
    ]
    
    palette = {}
    for i in range(n_clusters):
        rgb = base_colors_rgb[i % len(base_colors_rgb)]
        palette[i] = f"rgba({rgb[0]}, {rgb[1]}, {rgb[2]}, {alpha})"
    
    # Noise gets light gray with lower alpha
    palette[-1] = f"rgba(200, 200, 200, {alpha * 0.5})"
    
    return palette


# Example usage
if __name__ == '__main__':
    from weighted_soft_clustering_v2 import WeightedSoftClustering, load_lda_topic_probabilities
    
    print("="*70)
    print("Add Cluster Trajectories to Sankey")
    print("="*70)
    
    # 1. Build tree and get clusters
    print("\n1. Building consensus tree and extracting clusters...")
    topic_probs = load_lda_topic_probabilities('GENSIM', k_min=2, k_max=9)
    tree = WeightedSoftClustering(topic_probs, verbose=False)
    clusters = tree.get_clusters_from_tree(min_cluster_size=15)
    
    # Create Series of cluster labels
    cluster_labels = pd.Series(clusters['sample_to_label'])
    print(f"   ✓ Found {clusters['n_clusters']} clusters for {len(cluster_labels)} samples")
    
    # 2. Add cluster trajectories to Sankey data
    print("\n2. Adding cluster trajectories to Sankey data...")
    enhanced_data = add_cluster_trajectories_to_sankey(
        sankey_json_path='GENSIM/lda_results/fully_integrated_sankey_data_improved.json',
        cluster_labels=cluster_labels,
        output_path='GENSIM/lda_results/sankey_with_cluster_trajectories.json',
        alpha=0.6
    )
    
    # 3. Show what was created
    print("\n3. Cluster trajectory colors:")
    for i, traj in enumerate(enhanced_data['cluster_trajectories'][:5]):
        print(f"   Cluster {traj['cluster_id']}: {traj['color']} "
              f"({traj['n_samples']} samples)")
    if len(enhanced_data['cluster_trajectories']) > 5:
        print(f"   ... and {len(enhanced_data['cluster_trajectories']) - 5} more clusters")
    
    print("\n" + "="*70)
    print("✓ Done! Ready to visualize in StripeSankey")
    print("="*70)
