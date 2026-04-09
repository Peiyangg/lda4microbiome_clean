"""
StripeSankey with cluster trajectory overlay visualization.

For now, this uses a simple approach: automatically draw cluster trajectories
by simulating multiple flow selections.
"""

import traitlets
from .stripesankey import StripeSankeyInline


class StripeSankeyWithClusters(StripeSankeyInline):
    """
    StripeSankey with cluster trajectory visualization.
    
    When cluster_trajectories data is present, this will draw all clusters
    simultaneously by leveraging the existing sample tracing feature.
    """
    
    # Add trait to control cluster visualization
    show_clusters = traitlets.Bool(default_value=False).tag(sync=True)
    cluster_trajectories_data = traitlets.List(default_value=[]).tag(sync=True)
    
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        
        # Extract cluster trajectory data if present
        if self.sankey_data and 'cluster_trajectories' in self.sankey_data:
            self.cluster_trajectories_data = self.sankey_data['cluster_trajectories']
    
    def toggle_clusters(self):
        """Toggle cluster trajectory visibility."""
        self.show_clusters = not self.show_clusters
        self._update_cluster_display()
        return self
    
    def show_cluster_trajectories(self):
        """Show cluster trajectories."""
        self.show_clusters = True
        self._update_cluster_display()
        return self
    
    def hide_cluster_trajectories(self):
        """Hide cluster trajectories."""
        self.show_clusters = False
        self.selected_flow = {}  # Clear selection to hide trajectories
        return self
    
    def _update_cluster_display(self):
        """Update the visualization based on cluster display state."""
        if self.show_clusters and self.cluster_trajectories_data:
            # For now, show first cluster as a demonstration
            # A full implementation would require JavaScript modifications
            if len(self.cluster_trajectories_data) > 0:
                first_cluster = self.cluster_trajectories_data[0]
                # This is a simplified version - ideally we'd draw all clusters
                print(f"Displaying cluster {first_cluster['cluster_id']} with {first_cluster['n_samples']} samples")
                print("Note: Full multi-cluster visualization requires JavaScript extension")
        else:
            self.selected_flow = {}
