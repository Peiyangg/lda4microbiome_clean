"""
Simplified Weighted Soft Clustering - Version 2
Only includes distance consensus approach with direct Jensen-Shannon distances.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import scipy.cluster.hierarchy as sch
from scipy.spatial.distance import squareform
from typing import Dict, List


def load_lda_topic_probabilities(
    folder: str, 
    k_min: int = 2, 
    k_max: int = 10,
    k_list: List[int] = None
) -> Dict[int, pd.DataFrame]:
    """
    Load LDA topic probability matrices for multiple k values.
    
    Parameters
    ----------
    folder : str
        Path to folder containing MC_Sample_probabilitiesK.csv files
    k_min : int
        Minimum k value (used if k_list is None)
    k_max : int
        Maximum k value (used if k_list is None)
    k_list : List[int], optional
        Specific k values to load (e.g., [3, 4, 5]). 
        If provided, k_min and k_max are ignored.
    
    Returns
    -------
    dict mapping k -> DataFrame (topics x samples)
    
    Examples
    --------
    # Load all k from 2 to 9
    topic_probs = load_lda_topic_probabilities('GENSIM', k_min=2, k_max=9)
    
    # Load only specific k values based on perplexity/coherence scores
    topic_probs = load_lda_topic_probabilities('GENSIM', k_list=[3, 4, 5])
    """
    import os
    topic_probs = {}
    
    # Determine which k values to load
    if k_list is not None:
        k_values = k_list
    else:
        k_values = range(k_min, k_max + 1)
    
    for k in k_values:
        file_path = os.path.join(folder, 'lda_results', 'MC_Sample', f'MC_Sample_probabilities{k}.csv')
        if os.path.exists(file_path):
            df = pd.read_csv(file_path, index_col=0)
            topic_probs[k] = df
        else:
            print(f"Warning: File not found for k={k}: {file_path}")
    
    return topic_probs


class WeightedSoftClustering:
    """
    Weighted soft clustering using distance consensus approach.
    
    Builds a consensus hierarchical tree by averaging Jensen-Shannon distances
    across multiple k values.
    """
    
    def __init__(self, topic_probs: Dict[int, pd.DataFrame], verbose: bool = True):
        """
        Initialize weighted soft clustering.
        
        Parameters
        ----------
        topic_probs : dict
            Dictionary mapping k -> DataFrame of topic probabilities (topics x samples)
        verbose : bool
            Print progress information
        """
        self.topic_probs = topic_probs
        self.verbose = verbose
        self.k_values = sorted(topic_probs.keys())
        
        # Get sample names from first k
        first_k = self.k_values[0]
        self.samples = topic_probs[first_k].columns.tolist()
        self.n_samples = len(self.samples)
        
        # Build consensus
        self.consensus_distance = None
        self.linkage_matrix = None
        self.dendrogram_info = None
        
        # Build tree automatically
        self.build_tree()
    
    def _jensen_shannon_distance(self, p: np.ndarray, q: np.ndarray) -> float:
        """Compute Jensen-Shannon distance between two probability distributions."""
        from scipy.spatial.distance import jensenshannon
        return jensenshannon(p, q)
    
    def _compute_js_distance_matrix(self, k: int) -> np.ndarray:
        """Compute Jensen-Shannon distance matrix for a given k."""
        probs = self.topic_probs[k].T.values  # Shape: (n_samples, n_topics)
        n = len(probs)
        dist_matrix = np.zeros((n, n))
        
        for i in range(n):
            for j in range(i + 1, n):
                dist = self._jensen_shannon_distance(probs[i], probs[j])
                dist_matrix[i, j] = dist
                dist_matrix[j, i] = dist
        
        return dist_matrix
    
    def build_tree(self):
        """
        Build consensus tree by averaging Jensen-Shannon distances across k values.
        """
        if self.verbose:
            print(f'Computing consensus distances across k={self.k_values}...')
        
        # Compute distance matrix for each k
        distance_matrices = []
        for k in self.k_values:
            dist_matrix = self._compute_js_distance_matrix(k)
            distance_matrices.append(dist_matrix)
            if self.verbose:
                print(f'  k={k}: distance range [{dist_matrix.min():.4f}, {dist_matrix.max():.4f}]')
        
        # Average across all k values (equal weights)
        self.consensus_distance = np.mean(distance_matrices, axis=0)
        
        if self.verbose:
            print(f'Consensus distance range: [{self.consensus_distance.min():.4f}, {self.consensus_distance.max():.4f}]')
        
        # Build hierarchical tree using average linkage
        condensed_dist = squareform(self.consensus_distance)
        self.linkage_matrix = sch.linkage(condensed_dist, method='average')
        
        if self.verbose:
            print('✓ Consensus tree built')
    
    def get_distance_matrix(self) -> pd.DataFrame:
        """
        Get the consensus distance matrix as a DataFrame.
        
        This can be used directly with HDBSCAN:
        
        Example
        -------
        >>> tree = WeightedSoftClustering(topic_probs)
        >>> dist_matrix = tree.get_distance_matrix()
        >>> 
        >>> import hdbscan
        >>> clusterer = hdbscan.HDBSCAN(min_cluster_size=15, metric='precomputed')
        >>> labels = clusterer.fit_predict(dist_matrix.values)
        
        Returns
        -------
        pd.DataFrame with sample names as index and columns
        """
        return pd.DataFrame(
            self.consensus_distance,
            index=self.samples,
            columns=self.samples
        )
    
    def viz(self, figsize=(12, 8), save_path=None, **kwargs):
        """
        Visualize the consensus dendrogram.
        
        Parameters
        ----------
        figsize : tuple
            Figure size
        save_path : str, optional
            Path to save the figure
        **kwargs : dict
            Additional arguments passed to scipy.dendrogram
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Plot dendrogram
        self.dendrogram_info = sch.dendrogram(
            self.linkage_matrix,
            labels=self.samples,
            ax=ax,
            **kwargs
        )
        
        ax.set_xlabel('Sample', fontsize=12)
        ax.set_ylabel('Distance', fontsize=12)
        ax.set_title('Weighted Soft Clustering Dendrogram (Consensus)', 
                     fontsize=14, fontweight='bold')
        
        # Rotate labels for readability
        plt.xticks(rotation=90, ha='right')
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"Figure saved to {save_path}")
        
        return fig, ax
    
    def get_clusters_from_tree(self, min_cluster_size: int = 15) -> Dict:
        """
        Extract flat clusters from the linkage tree using HDBSCAN's EOM method.
        
        Parameters
        ----------
        min_cluster_size : int
            Minimum number of samples required to form a cluster
        
        Returns
        -------
        dict containing:
            'labels': cluster labels (-1 for noise)
            'probabilities': membership probabilities
            'sample_to_label': dict mapping sample names to labels
            'n_clusters': number of clusters
            'n_noise': number of noise points
            'cluster_sizes': array of cluster sizes
            'noise_ratio': fraction of noise points
            'condensed_tree': condensed tree object
            'selected_clusters': selected cluster IDs
        """
        try:
            from from_fast_hdbscan.extract_flat_clusters import extract_flat_clustering_from_linkage, get_cluster_info
        except ImportError as e:
            raise ImportError(
                f"Cannot import from from_fast_hdbscan: {e}\n"
                "Make sure fast-hdbscan is installed: pip install fast-hdbscan"
            )
        
        # Extract flat clustering using HDBSCAN EOM method
        labels, probabilities, condensed_tree, selected_clusters = extract_flat_clustering_from_linkage(
            self.linkage_matrix,
            min_cluster_size=min_cluster_size,
            allow_single_cluster=False,
        )
        
        # Get cluster statistics
        info = get_cluster_info(labels)
        
        # Create sample name to label mapping
        sample_to_label = dict(zip(self.samples, labels))
        
        return {
            'labels': labels,
            'probabilities': probabilities,
            'sample_to_label': sample_to_label,
            'n_clusters': info['n_clusters'],
            'n_noise': info['n_noise'],
            'cluster_sizes': info['cluster_sizes'],
            'noise_ratio': info['noise_ratio'],
            'condensed_tree': condensed_tree,
            'selected_clusters': selected_clusters,
        }
    
    def plot_condensed_tree(self, condensed_tree, selected_clusters, figsize=(14, 8)):
        """
        Plot the HDBSCAN condensed tree showing cluster selection.
        
        Parameters
        ----------
        condensed_tree : CondensedTree
            Condensed tree from get_clusters_from_tree()
        selected_clusters : ndarray
            Selected cluster IDs from get_clusters_from_tree()
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        # Get condensed tree data
        parents = condensed_tree.parent
        children = condensed_tree.child
        lambdas = condensed_tree.lambda_val
        
        # Create position mapping
        unique_nodes = np.unique(np.concatenate([parents, children]))
        node_positions = {node: i for i, node in enumerate(sorted(unique_nodes))}
        
        # Convert selected clusters to set
        selected_set = set(selected_clusters) if selected_clusters is not None else set()
        
        # Plot each edge
        for parent, child, lambda_val in zip(parents, children, lambdas):
            x_parent = node_positions[parent]
            x_child = node_positions[child]
            
            # Determine color
            if child in selected_set:
                color = 'red'
                linewidth = 2
                alpha = 0.8
            else:
                color = 'blue'
                linewidth = 1
                alpha = 0.3
            
            # Draw vertical line
            ax.plot([x_child, x_child], [0, lambda_val], 
                   color=color, linewidth=linewidth, alpha=alpha)
            
            # Draw horizontal connection
            if x_parent != x_child:
                ax.plot([x_child, x_parent], [lambda_val, lambda_val],
                       color=color, linewidth=linewidth, alpha=alpha, linestyle='--')
        
        ax.set_xlabel('Cluster/Sample', fontsize=12)
        ax.set_ylabel('λ (1/distance)', fontsize=12)
        ax.set_title('HDBSCAN Condensed Tree\\n(Red = Selected Clusters, Blue = Not Selected)', 
                     fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3)
        ax.set_xticks([])
        
        plt.tight_layout()
        return fig, ax
    
    def build_cluster_hierarchy(self, labels: np.ndarray, method: str = 'average') -> Dict:
        """
        Build hierarchical tree showing relationships between flat clusters.
        
        Takes HDBSCAN flat clustering results and builds a hierarchy showing
        how the clusters would merge (ignoring noise points).
        
        Parameters
        ----------
        labels : np.ndarray
            Flat cluster labels (e.g., from get_clusters_from_tree() or HDBSCAN)
            Shape: (n_samples,), values: cluster IDs or -1 for noise
        method : str
            Linkage method: 'average', 'single', 'complete', 'ward'
        
        Returns
        -------
        dict containing:
            'cluster_linkage': linkage matrix for clusters
            'cluster_labels': array of cluster IDs (excluding noise)
            'n_clusters': number of clusters
            'cluster_distances': distance matrix between clusters
            'cluster_sizes': size of each cluster
        """
        from scipy.spatial.distance import squareform
        
        # Remove noise points
        mask = labels != -1
        labels_no_noise = labels[mask]
        
        # Get distance matrix for non-noise samples
        dist_matrix_no_noise = self.consensus_distance[np.ix_(mask, mask)]
        
        # Get unique clusters
        unique_clusters = np.unique(labels_no_noise)
        n_clusters = len(unique_clusters)
        
        # Compute average distance between each pair of clusters
        cluster_distances = np.zeros((n_clusters, n_clusters))
        cluster_sizes = []
        
        for i, c1 in enumerate(unique_clusters):
            idx1 = np.where(labels_no_noise == c1)[0]
            cluster_sizes.append(len(idx1))
            
            for j, c2 in enumerate(unique_clusters):
                if i < j:
                    idx2 = np.where(labels_no_noise == c2)[0]
                    # Average distance between all pairs of samples in the two clusters
                    dists = dist_matrix_no_noise[np.ix_(idx1, idx2)]
                    cluster_distances[i, j] = dists.mean()
                    cluster_distances[j, i] = cluster_distances[i, j]
        
        # Build hierarchical clustering of clusters
        condensed_dist = squareform(cluster_distances)
        cluster_linkage = sch.linkage(condensed_dist, method=method)
        
        return {
            'cluster_linkage': cluster_linkage,
            'cluster_labels': unique_clusters,
            'n_clusters': n_clusters,
            'cluster_distances': cluster_distances,
            'cluster_sizes': np.array(cluster_sizes),
        }
    
    def plot_cluster_hierarchy(self, cluster_hierarchy: Dict, figsize=(10, 6)):
        """
        Plot dendrogram showing how flat clusters merge.
        
        Parameters
        ----------
        cluster_hierarchy : dict
            Output from build_cluster_hierarchy()
        figsize : tuple
            Figure size
        
        Returns
        -------
        fig, ax : matplotlib figure and axes
        """
        fig, ax = plt.subplots(figsize=figsize)
        
        cluster_labels = cluster_hierarchy['cluster_labels']
        cluster_sizes = cluster_hierarchy['cluster_sizes']
        n_clusters = cluster_hierarchy['n_clusters']
        
        # Create labels with cluster ID and size
        labels = [f"C{c}\n(n={s})" for c, s in zip(cluster_labels, cluster_sizes)]
        
        # Plot dendrogram
        sch.dendrogram(
            cluster_hierarchy['cluster_linkage'],
            labels=labels,
            ax=ax
        )
        
        ax.set_xlabel('Cluster ID (size)', fontsize=12)
        ax.set_ylabel('Average Distance', fontsize=12)
        ax.set_title(f'Cluster-Level Hierarchy ({n_clusters} clusters)', 
                     fontsize=14, fontweight='bold')
        
        plt.xticks(rotation=45, ha='right')
        plt.tight_layout()
        
        return fig, ax
    
    def stats(self) -> pd.DataFrame:
        """
        Get statistics about the consensus distances.
        
        Returns
        -------
        pd.DataFrame with distance statistics for each k
        """
        stats_list = []
        
        for k in self.k_values:
            dist_matrix = self._compute_js_distance_matrix(k)
            stats_list.append({
                'k': k,
                'mean_distance': dist_matrix[np.triu_indices_from(dist_matrix, k=1)].mean(),
                'std_distance': dist_matrix[np.triu_indices_from(dist_matrix, k=1)].std(),
                'min_distance': dist_matrix[np.triu_indices_from(dist_matrix, k=1)].min(),
                'max_distance': dist_matrix[np.triu_indices_from(dist_matrix, k=1)].max(),
            })
        
        return pd.DataFrame(stats_list)
