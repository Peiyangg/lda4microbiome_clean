"""
Filter Sankey JSON data to show only selected k values.

This allows you to display only specific k values (e.g., [3, 4, 5]) 
in the StripeSankey diagram instead of all k values.
"""

import json
from typing import List, Dict


def filter_sankey_by_k(
    sankey_json_path: str,
    k_list: List[int],
    output_path: str = None
) -> Dict:
    """
    Filter Sankey data to include only specified k values.
    
    Parameters
    ----------
    sankey_json_path : str
        Path to the original Sankey JSON file
    k_list : List[int]
        List of k values to keep (e.g., [3, 4, 5])
    output_path : str, optional
        Path to save filtered JSON. If None, returns data without saving.
    
    Returns
    -------
    dict : Filtered Sankey data with only selected k values
    
    Examples
    --------
    # Filter to show only k=3, 4, 5
    filtered_data = filter_sankey_by_k(
        'GENSIM/lda_results/fully_integrated_sankey_data_improved.json',
        k_list=[3, 4, 5],
        output_path='GENSIM/lda_results/sankey_k345.json'
    )
    """
    # Load original Sankey data
    with open(sankey_json_path, 'r') as f:
        sankey_data = json.load(f)
    
    print(f"Original data: {len(sankey_data.get('nodes', {}))} nodes, "
          f"{len(sankey_data.get('flows', []))} flows, "
          f"k_range: {sankey_data.get('k_range', [])}")
    
    # Update k_range
    sankey_data['k_range'] = sorted(k_list)
    
    # Filter nodes - keep only nodes from selected k values
    filtered_nodes = {}
    for node_id, node_data in sankey_data.get('nodes', {}).items():
        # Extract k value from node ID (format: K3_MC0, K4_MC1, etc.)
        if node_id.startswith('K'):
            k_value = int(node_id.split('_')[0][1:])  # Extract number after 'K'
            if k_value in k_list:
                filtered_nodes[node_id] = node_data
    
    sankey_data['nodes'] = filtered_nodes
    
    # Filter flows - keep only flows between nodes in selected k values
    filtered_flows = []
    for flow in sankey_data.get('flows', []):
        source_k = flow.get('source_k')
        target_k = flow.get('target_k')
        
        # Keep flow if both source and target are in selected k values
        if source_k in k_list and target_k in k_list:
            filtered_flows.append(flow)
    
    sankey_data['flows'] = filtered_flows
    
    print(f"Filtered data: {len(filtered_nodes)} nodes, "
          f"{len(filtered_flows)} flows, "
          f"k_range: {sankey_data['k_range']}")
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(sankey_data, f, indent=2)
        print(f"✓ Filtered Sankey data saved to: {output_path}")
    
    return sankey_data


def filter_sankey_by_k_with_clusters(
    sankey_json_path: str,
    k_list: List[int],
    output_path: str = None
) -> Dict:
    """
    Filter Sankey data with cluster colors to include only specified k values.
    
    This is a wrapper around filter_sankey_by_k that preserves cluster metadata.
    
    Parameters
    ----------
    sankey_json_path : str
        Path to Sankey JSON file with cluster colors
    k_list : List[int]
        List of k values to keep (e.g., [3, 4, 5])
    output_path : str, optional
        Path to save filtered JSON
    
    Returns
    -------
    dict : Filtered Sankey data with cluster information preserved
    """
    filtered_data = filter_sankey_by_k(sankey_json_path, k_list, output_path=None)
    
    # Preserve cluster metadata if it exists
    with open(sankey_json_path, 'r') as f:
        original_data = json.load(f)
    
    if 'cluster_metadata' in original_data:
        filtered_data['cluster_metadata'] = original_data['cluster_metadata']
    if 'sample_clusters' in original_data:
        filtered_data['sample_clusters'] = original_data['sample_clusters']
    if 'cluster_colors' in original_data:
        filtered_data['cluster_colors'] = original_data['cluster_colors']
    if 'cluster_trajectories' in original_data:
        filtered_data['cluster_trajectories'] = original_data['cluster_trajectories']
    
    # Save if output path provided
    if output_path:
        with open(output_path, 'w') as f:
            json.dump(filtered_data, f, indent=2)
        print(f"✓ Filtered Sankey data (with clusters) saved to: {output_path}")
    
    return filtered_data


# Example usage
if __name__ == '__main__':
    print("="*70)
    print("Filter Sankey Data by K Values")
    print("="*70)
    
    # Example 1: Filter basic Sankey data
    print("\n1. Filtering basic Sankey to k=[3, 4, 5]...")
    filtered_data = filter_sankey_by_k(
        sankey_json_path='GENSIM/lda_results/fully_integrated_sankey_data_improved.json',
        k_list=[3, 4, 5],
        output_path='GENSIM/lda_results/sankey_k345_filtered.json'
    )
    
    # Example 2: Filter Sankey with cluster colors
    print("\n2. Filtering Sankey with clusters to k=[3, 4, 5]...")
    filtered_with_clusters = filter_sankey_by_k_with_clusters(
        sankey_json_path='GENSIM/lda_results/sankey_with_cluster_colors.json',
        k_list=[3, 4, 5],
        output_path='GENSIM/lda_results/sankey_with_clusters_k345.json'
    )
    
    print("\n" + "="*70)
    print("✓ Done!")
    print("="*70)
    print("\nNext steps:")
    print("1. Load the filtered JSON in StripeSankey")
    print("2. The diagram will only show k=3, 4, 5")
    print("\nExample code:")
    print("  from lda4microbiome_backup.stripesankey import StripeSankeyInline")
    print("  sankey = StripeSankeyInline(")
    print("      sankey_data='GENSIM/lda_results/sankey_k345_filtered.json',")
    print("      mode='default'")
    print("  )")
    print("  sankey")
