"""
Complete workflow: Create filtered Sankey with cluster trajectory visualization
"""
import pandas as pd
import json
from weighted_soft_clustering_v2 import load_lda_topic_probabilities, WeightedSoftClustering
from add_cluster_trajectories import add_cluster_trajectories_to_sankey
from filter_sankey_by_k import filter_sankey_by_k_with_clusters

print("="*80)
print("COMPLETE WORKFLOW: Sankey k=[3,4,5] with Cluster Trajectories")
print("="*80)

# Selected k values
selected_k = [3, 4, 5]

# Step 1: Load topic probabilities
print(f"\n{'='*80}")
print(f"STEP 1: Load topic probabilities for k={selected_k}")
print("="*80)
topic_probs = load_lda_topic_probabilities('GENSIM', k_list=selected_k)
print(f"✓ Loaded k values: {sorted(topic_probs.keys())}")

# Step 2: Build consensus tree
print(f"\n{'='*80}")
print("STEP 2: Build consensus tree")
print("="*80)
tree = WeightedSoftClustering(topic_probs, verbose=True)

# Step 3: Extract clusters
print(f"\n{'='*80}")
print("STEP 3: Extract flat clusters")
print("="*80)
clusters = tree.get_clusters_from_tree(min_cluster_size=15)
cluster_labels = pd.Series(clusters['sample_to_label'])

print(f"✓ Clustering results:")
print(f"  - Clusters: {clusters['n_clusters']}")
print(f"  - Total samples: {len(cluster_labels)}")
print(f"  - Noise: {(cluster_labels == -1).sum()} ({clusters['noise_ratio']:.1%})")

# Step 4: Add cluster trajectories to FULL Sankey data
print(f"\n{'='*80}")
print("STEP 4: Add cluster trajectories to full Sankey data")
print("="*80)
sankey_with_trajectories = add_cluster_trajectories_to_sankey(
    sankey_json_path='GENSIM/lda_results/fully_integrated_sankey_data_improved.json',
    cluster_labels=cluster_labels,
    output_path='GENSIM/lda_results/sankey_with_trajectories_full.json',
    alpha=0.7
)

# Step 5: Filter to selected k values
print(f"\n{'='*80}")
print(f"STEP 5: Filter to k={selected_k}")
print("="*80)

# Load the full sankey with trajectories
with open('GENSIM/lda_results/sankey_with_trajectories_full.json', 'r') as f:
    full_data = json.load(f)

# Filter nodes and flows
print(f"Original: {len(full_data['nodes'])} nodes, {len(full_data['flows'])} flows")

# Update k_range
full_data['k_range'] = sorted(selected_k)

# Filter nodes
filtered_nodes = {}
for node_id, node_data in full_data['nodes'].items():
    if node_id.startswith('K'):
        k_value = int(node_id.split('_')[0][1:])
        if k_value in selected_k:
            filtered_nodes[node_id] = node_data

full_data['nodes'] = filtered_nodes

# Filter flows
filtered_flows = []
for flow in full_data['flows']:
    if flow['source_k'] in selected_k and flow['target_k'] in selected_k:
        filtered_flows.append(flow)

full_data['flows'] = filtered_flows

print(f"Filtered: {len(filtered_nodes)} nodes, {len(filtered_flows)} flows")

# Save filtered data with cluster trajectories
with open('GENSIM/lda_results/sankey_k345_with_trajectories.json', 'w') as f:
    json.dump(full_data, f, indent=2)

print(f"✓ Saved to: sankey_k345_with_trajectories.json")

# Step 6: Summary
print(f"\n{'='*80}")
print("STEP 6: Summary")
print("="*80)
print(f"✓ Filtered Sankey with cluster trajectories created!")
print(f"\nData statistics:")
print(f"  - K values: {full_data['k_range']}")
print(f"  - Nodes: {len(full_data['nodes'])}")
print(f"  - Flows: {len(full_data['flows'])}")
print(f"  - Clusters: {len(full_data['cluster_trajectories'])}")

print(f"\nCluster distribution:")
for traj in full_data['cluster_trajectories']:
    cid = traj['cluster_id']
    size = traj['n_samples']
    color = traj['color']
    print(f"  - Cluster {cid}: {size} samples (color: {color})")

print(f"\n{'='*80}")
print("✓ READY TO VISUALIZE!")
print("="*80)
print(f"\nThe file 'sankey_k345_with_trajectories.json' contains:")
print(f"  ✓ Only k=3, 4, 5 (filtered)")
print(f"  ✓ Cluster trajectory data for visualization")
print(f"  ✓ {len(full_data['cluster_trajectories'])} clusters with distinct colors")
print(f"\nNext: Run the visualization notebook to see cluster-colored flows!")
print("="*80)
