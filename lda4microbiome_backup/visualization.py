"""
Interactive Visualization tools for LDA results using Plotly.

This module contains classes for:
- Creating interactive clustered heatmaps with metadata annotations
- Topic-taxon distribution visualizations with hover information
- Customizable color schemes and interactive layouts
- Web-ready visualizations for data exploration
"""

from typing import List, Dict, Optional, Tuple, Any, Union
import os
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
import plotly.io as pio
from plotly.subplots import make_subplots
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial import distance
from scipy.cluster.hierarchy import fcluster, cut_tree
import plotly.colors as pcolors
# Kaleido imports for SVG saving
import kaleido
from kaleido import write_fig_sync
# Additional imports for MCComparison class
import matplotlib.pyplot as plt
import seaborn as sns
from matplotlib.colors import LinearSegmentedColormap, ListedColormap
import matplotlib.patches as mpatches


class TopicFeatureProcessor:
    """
    A class for processing topic-feature matrices by mapping ASVs to taxonomic levels.

    This class handles:
    - Loading ASV probabilities and taxonomic data
    - Mapping ASVs to specified taxonomic levels
    - Grouping and summing probabilities by taxonomic level
    - Creating visualizations of topic-feature matrices
    - Integration with LDAModelVisualizer workflow
    """

    def __init__(self, base_directory: str, k_value: int):
        """
        Initialize the Topic Feature Processor.

        Args:
            base_directory: Base directory where LDA results are stored
            k_value: Number of topics (K) to process
        """
        self.base_directory = base_directory
        self.k_value = k_value

        # Set up paths
        self._setup_paths()

        # Initialize data containers
        self.new_taxa = None
        self.asv_probabilities = None
        self.processed_data = {}

        print(f"TopicFeatureProcessor initialized for K={k_value}")
        print(f"  Base directory: {self.base_directory}")

    def _setup_paths(self):
        """Set up all necessary paths."""
        self.inter_directory = os.path.join(self.base_directory, 'intermediate')
        self.lda_directory = os.path.join(self.base_directory, 'lda_results')
        self.MC_feature_directory = os.path.join(self.lda_directory, 'MC_Feature')
        self.viz_directory = os.path.join(self.base_directory, 'lda_visualization')

        # Create visualization directory
        os.makedirs(self.viz_directory, exist_ok=True)

        # Set up file paths
        self.path_to_new_taxa = os.path.join(self.inter_directory, "intermediate_taxa.csv")
        self.path_to_ASVProbabilities = os.path.join(
            self.MC_feature_directory, f"MC_Feature_Probabilities_{self.k_value}.csv"
        )

    def load_data(self):
        """Load taxonomic and ASV probability data."""
        try:
            # Load taxonomic data
            print(f"Reading taxonomic data from: {self.path_to_new_taxa}")
            self.new_taxa = pd.read_csv(self.path_to_new_taxa, index_col=0)
            print(f"✓ Taxonomic data loaded: {self.new_taxa.shape}")

            # Load ASV probabilities
            print(f"Reading ASV probabilities from: {self.path_to_ASVProbabilities}")
            self.asv_probabilities = pd.read_csv(self.path_to_ASVProbabilities, index_col=0)
            print(f"✓ ASV probabilities loaded: {self.asv_probabilities.shape}")

            # Display available taxonomic levels
            available_levels = [col for col in self.new_taxa.columns if col not in ['randomID']]
            print(f"✓ Available taxonomic levels: {available_levels}")

        except FileNotFoundError as e:
            raise FileNotFoundError(f"Required data file not found: {e}")
        except Exception as e:
            raise Exception(f"Error loading data: {e}")

    def process_feature_level(self, 
                            feature_level: str = 'genus_ID', 
                            top_n: int = 10) -> Tuple[pd.DataFrame, Optional[pd.DataFrame]]:
        """
        Process ASV probabilities at a specified taxonomic level.

        Args:
            feature_level: Taxonomic level to group by ('Genus', 'Family', 'Order', etc.)
            top_n: Number of top features to extract per topic

        Returns:
            Tuple of (processed_probabilities_df, top_tokens_df)
        """
        # Load data if not already loaded
        if self.new_taxa is None or self.asv_probabilities is None:
            self.load_data()

        # Validate feature level
        if feature_level not in self.new_taxa.columns:
            available_cols = [col for col in self.new_taxa.columns if col not in ['randomID']]
            raise ValueError(f"Feature level '{feature_level}' not found. Available: {available_cols}")

        print(f"\nProcessing feature level: {feature_level}")
        print("=" * 50)

        # Create mapping dictionary
        print(f"Creating mapping dictionary using column: {feature_level}")
        mapping_dict = dict(zip(self.new_taxa['randomID'], self.new_taxa[feature_level]))

        # Create MC labels
        MC_list = [f"MC{i}" for i in range(self.k_value)]
        print(f"Created {len(MC_list)} MC labels: {MC_list}")

        # Prepare ASV probabilities
        AP = self.asv_probabilities.copy()
        AP = AP.reset_index(drop=True)
        AP.index = MC_list

        print(f"Original ASV probabilities shape: {AP.shape}")

        # Map column names using the dictionary
        print("Mapping ASV IDs to taxonomic names...")
        original_columns = AP.columns.tolist()
        AP.columns = [mapping_dict.get(col, col) for col in AP.columns]

        # Count mapped columns
        mapped_count = sum(1 for orig_col in original_columns if orig_col in mapping_dict)
        print(f"Mapped {mapped_count}/{len(original_columns)} columns to taxonomic names")

        # Group by taxonomic level and sum probabilities
        print(f"Grouping by {feature_level} and summing probabilities...")
        grouped_AP = AP.groupby(level=0, axis=1).sum()

        print(f"✓ Grouped probabilities shape: {grouped_AP.shape}")
        print(f"✓ Unique {feature_level} features: {len(grouped_AP.columns)}")

        # Calculate top tokens
        top_tokens_df = None
        try:
            print(f"Calculating top {top_n} tokens for each MC...")
            top_tokens_df = grouped_AP.apply(lambda row: row.nlargest(top_n), axis=1)
            print(f"✓ Top tokens calculated: {top_tokens_df.shape}")
        except Exception as e:
            print(f"Warning: Could not calculate top tokens - {str(e)}")
            top_tokens_df = None

        # Store processed data
        self.processed_data[feature_level] = {
            'grouped_probabilities': grouped_AP,
            'top_tokens': top_tokens_df,
            'mapping_count': mapped_count,
            'total_features': len(grouped_AP.columns)
        }

        # Print summary
        print("\n=== PROCESSING SUMMARY ===")
        print(f"Feature level: {feature_level}")
        print(f"Number of MCs: {len(MC_list)}")
        print(f"Original ASVs: {len(original_columns)}")
        print(f"Mapped ASVs: {mapped_count}")
        print(f"Unique {feature_level} features: {len(grouped_AP.columns)}")
        print(f"Top {top_n} features calculated: {'Yes' if top_tokens_df is not None else 'No'}")
        print("=" * 50)

        return grouped_AP, top_tokens_df

    def get_feature_summary(self, feature_level: str) -> Dict[str, Any]:
        """
        Get summary statistics for a processed feature level.

        Args:
            feature_level: Taxonomic level to summarize

        Returns:
            Dictionary with summary statistics
        """
        if feature_level not in self.processed_data:
            raise ValueError(f"Feature level '{feature_level}' not processed yet")

        data = self.processed_data[feature_level]
        grouped_probs = data['grouped_probabilities']

        # Calculate statistics
        summary = {
            'feature_level': feature_level,
            'k_value': self.k_value,
            'num_topics': len(grouped_probs),
            'num_features': len(grouped_probs.columns),
            'mapped_asvs': data['mapping_count'],
            'total_probability_mass': grouped_probs.sum().sum(),
            'avg_probability_per_feature': grouped_probs.mean(axis=0).mean(),
            'max_probability': grouped_probs.max().max(),
            'min_probability': grouped_probs.min().min(),
            'sparsity': (grouped_probs == 0).sum().sum() / grouped_probs.size,
            'top_features_per_topic': {}
        }

        # Get top feature for each topic
        for topic in grouped_probs.index:
            top_feature = grouped_probs.loc[topic].idxmax()
            top_prob = grouped_probs.loc[topic].max()
            summary['top_features_per_topic'][topic] = {
                'feature': top_feature,
                'probability': top_prob
            }

        return summary



class LDAModelVisualizerInteractive:
    """
    An interactive visualization class for LDA model results using Plotly.

    This class handles:
    - Loading model results for a specific K value
    - Creating interactive clustered heatmaps with metadata annotations
    - Topic-taxon distribution visualizations with hover information
    - Customizable color schemes for categorical and continuous variables
    - Web-ready interactive visualizations
    """

    def __init__(self, base_directory: str, k_value: int, metadata_path: str,
                 universal_headers: List[str], continuous_headers: List[str] = None,
                 top_asv_count: int = 7, id_column: str = 'ID'):
        """
        Initialize the Interactive LDA Model Visualizer.

        Args:
            base_directory: Base directory where LDA results are stored
            k_value: Number of topics (K) to visualize
            metadata_path: Path to metadata CSV file
            universal_headers: List of categorical metadata columns to include (required)
            continuous_headers: List of continuous metadata columns to include (optional)
            top_asv_count: Number of top ASVs to show in heatmaps (default: 7)
            id_column: Column name to use for sample IDs (default: 'ID')
        """
        self.base_directory = base_directory
        self.k_value = k_value
        self.metadata_path = metadata_path

        # Configure Plotly renderer for better compatibility
        self._configure_plotly_renderer()

        # Set up paths
        self._setup_paths()

        # Initialize configuration
        self.config = {
            'universal_headers': universal_headers,
            'continuous_headers': continuous_headers or [],
            'top_asv_count': top_asv_count,
            'id_column': id_column,
            'custom_colors': {},  # Will be set with defaults or user values
            'continuous_cmaps': {},  # Will be set with defaults or user values
            'figure_width': 1200,
            'figure_height': 800,
            'heatmap_colorscale': 'Viridis',
            'show_dendrograms': True
        }

        # Set default colors
        self._set_default_colors()

        # Load data
        self._load_data()

    def _configure_plotly_renderer(self):
        """
        Configure Plotly renderer for better compatibility across different environments.
        
        This sets the default renderer to 'browser' to avoid issues with 
        notebook renderers in different environments.
        """
        try:
            # Try to set the renderer to browser for better compatibility
            pio.renderers.default = "browser"
            print("✓ Plotly renderer configured to 'browser'")
        except Exception as e:
            # Fallback to plotly_mimetype if browser doesn't work
            try:
                pio.renderers.default = "plotly_mimetype+notebook"
                print("✓ Plotly renderer configured to 'plotly_mimetype+notebook' (fallback)")
            except Exception as e2:
                # If all else fails, just print a warning but continue
                print(f"⚠️ Could not configure Plotly renderer: {e}. Using default renderer.")

    def _setup_paths(self):
        """Set up all necessary paths."""
        self.loop_directory = os.path.join(self.base_directory, 'lda_loop')
        self.inter_directory = os.path.join(self.base_directory, 'intermediate')
        self.viz_directory = os.path.join(self.base_directory, 'lda_visualization')
        self.lda_directory = os.path.join(self.base_directory, 'lda_results')
        self.MC_sample_directory = os.path.join(self.lda_directory, 'MC_Sample')
        self.MC_feature_directory = os.path.join(self.lda_directory, 'MC_Feature')

        # Create visualization directory
        os.makedirs(self.viz_directory, exist_ok=True)

        # Set up file paths for this specific K value
        self.path_to_DirichletComponentProbabilities = os.path.join(
            self.MC_sample_directory, 
            f"MC_Sample_probabilities{self.k_value}.csv"
        )
        self.path_to_ASVProbabilities = os.path.join(
            self.MC_feature_directory, 
            f"MC_Feature_Probabilities_{self.k_value}.csv"
        )
        self.path_to_new_taxa = os.path.join(
            self.inter_directory, 
            "intermediate_taxa.csv"
        )

    def _set_default_colors(self):
        """Set up default color configurations for Plotly."""
        # Plotly color palettes
        self.config['custom_colors'] = {}
        self.config['continuous_cmaps'] = {}
        
        # Default categorical color sequences
        self.default_color_sequences = [
            px.colors.qualitative.Set1,
            px.colors.qualitative.Set2,
            px.colors.qualitative.Set3,
            px.colors.qualitative.Pastel1,
            px.colors.qualitative.Dark2
        ]
        

    def _load_data(self):
        """Load all necessary data files."""
        try:
            # Load Dirichlet Component Probabilities (sample-topic distributions)
            DMP = pd.read_csv(self.path_to_DirichletComponentProbabilities, index_col=0)
            self.DM_distributions = DMP.values.tolist()
            self.sample_topic_df = DMP  # Keep as DataFrame for easier manipulation

            # Load ASV probabilities (topic-feature distributions)
            self.ASV_probabilities = pd.read_csv(self.path_to_ASVProbabilities, index_col=0)

            # Load metadata
            self.metadata = pd.read_csv(self.metadata_path, index_col=0)

            # Load taxonomy data if available
            if os.path.exists(self.path_to_new_taxa):
                self.taxa_data = pd.read_csv(self.path_to_new_taxa, index_col=0)
            else:
                self.taxa_data = None

        except Exception as e:
            raise FileNotFoundError(f"Error loading data: {e}")

    def configure_colors(self, 
                       custom_colors: Optional[Dict] = None,
                       continuous_cmaps: Optional[Dict] = None,
                       **kwargs):
        """
        Configure color schemes for visualization.

        Args:
            custom_colors: Dictionary of custom colors for categorical variables
            continuous_cmaps: Dictionary of colormaps for continuous variables
            **kwargs: Additional configuration parameters
        """
        if custom_colors is not None:
            self.config['custom_colors'].update(custom_colors)
        if continuous_cmaps is not None:
            self.config['continuous_cmaps'].update(continuous_cmaps)

        # Update any additional parameters
        for key, value in kwargs.items():
            if key in self.config:
                self.config[key] = value

        print("✓ Interactive color configuration updated")

    def prepare_heatmap_data(self, headers_to_include: Optional[List[str]] = None) -> pd.DataFrame:
        """
        Create a heatmap DataFrame with metadata information.

        Args:
            headers_to_include: List of column names to include

        Returns:
            DataFrame ready for interactive heatmap visualization
        """
        # Use configured headers if none provided
        if headers_to_include is None:
            headers_to_include = self.config['universal_headers'] + self.config['continuous_headers']

        # Get the sample-topic distribution data
        heatmap_data = self.sample_topic_df.copy()
        
        # Fill NaN values in metadata with 'Missing'
        metadata_df_filled = self.metadata.fillna('Missing')
        
        # Get sample IDs that match between data and metadata
        sample_ids = heatmap_data.columns.tolist()
        
        # Create metadata dictionary for easy lookup
        metadata_dict = {}
        for sample_id in sample_ids:
            # Find matching metadata row
            metadata_row = metadata_df_filled[
                metadata_df_filled[self.config['id_column']] == sample_id
            ]
            
            if not metadata_row.empty:
                sample_metadata = {}
                for header in headers_to_include:
                    if header in metadata_row.columns:
                        sample_metadata[header] = metadata_row[header].iloc[0]
                    else:
                        sample_metadata[header] = 'Missing'
                metadata_dict[sample_id] = sample_metadata
            else:
                # If no metadata found, fill with Missing
                metadata_dict[sample_id] = {header: 'Missing' for header in headers_to_include}
        
        # Store metadata for use in hover information
        self.sample_metadata = metadata_dict
        self.headers_for_display = headers_to_include
        
        return heatmap_data

    def create_clustered_heatmap_interactive(self, 
                                           heatmap_data: Optional[pd.DataFrame] = None,
                                           custom_filename: Optional[str] = None,
                                           show_dendrograms: bool = True,
                                           colorscale: str = 'Reds',
                                           order_by_metadata: Optional[str] = None) -> go.Figure:
        """
        Create an interactive clustered heatmap with dendrogram and metadata annotation bars.
        Structure: Dendrogram (top) -> Annotation bars (middle) -> Heatmap (bottom)

        Args:
            heatmap_data: DataFrame to visualize (if None, will prepare data)
            custom_filename: Custom filename for output (without extension)
            show_dendrograms: Whether to show dendrograms
            colorscale: Plotly colorscale name
            order_by_metadata: Name of metadata column to order by (overrides clustering)

        Returns:
            Plotly Figure object
        """
        if heatmap_data is None:
            heatmap_data = self.prepare_heatmap_data()

        # --- Determine sample ordering ---
        if order_by_metadata is not None:
            # Order by metadata column instead of clustering
            if order_by_metadata not in self.config['universal_headers'] + self.config['continuous_headers']:
                print(f"Warning: '{order_by_metadata}' not found in configured headers. Using clustering instead.")
                order_by_metadata = None
            else:
                # Get metadata values for ordering
                metadata_for_ordering = []
                sample_ids = heatmap_data.columns.tolist()
                
                for sample_id in sample_ids:
                    if hasattr(self, 'sample_metadata') and sample_id in self.sample_metadata:
                        value = self.sample_metadata[sample_id].get(order_by_metadata, 'Missing')
                        # Convert to numeric if continuous, keep as string if categorical
                        if order_by_metadata in self.config['continuous_headers']:
                            try:
                                if value == 'Missing' or pd.isna(value) or value == '':
                                    value = float('inf')  # Put missing values at the end
                                else:
                                    value = float(value)
                            except (ValueError, TypeError):
                                value = float('inf')
                        metadata_for_ordering.append((sample_id, value))
                    else:
                        # If no metadata, put at end
                        metadata_for_ordering.append((sample_id, 'Missing' if order_by_metadata in self.config['universal_headers'] else float('inf')))
                
                # Sort by metadata values
                metadata_for_ordering.sort(key=lambda x: x[1])
                dendro_leaves = [item[0] for item in metadata_for_ordering]
                sample_order = [list(heatmap_data.columns).index(sample_id) for sample_id in dendro_leaves]
                
                print(f"✓ Samples ordered by metadata column '{order_by_metadata}'")
        
        if order_by_metadata is None:
            # Use dendrogram clustering for sample order
            if show_dendrograms:
                dendro = ff.create_dendrogram(
                    heatmap_data.T.values, 
                    orientation='bottom',
                    labels=heatmap_data.columns.tolist()
                )
                dendro_leaves = dendro['layout']['xaxis']['ticktext']
                sample_order = [list(heatmap_data.columns).index(label) for label in dendro_leaves]
            else:
                dendro_leaves = heatmap_data.columns.tolist()
                sample_order = list(range(len(dendro_leaves)))

        # --- Reorder data based on sample order ---
        clustered_data = heatmap_data.iloc[:, sample_order]
        
        # --- Create reordered metadata for annotation bars ---
        reordered_metadata = {}
        for header in self.config['universal_headers'] + self.config['continuous_headers']:
            metadata_values = []
            for sample_id in dendro_leaves:
                if hasattr(self, 'sample_metadata') and sample_id in self.sample_metadata:
                    value = self.sample_metadata[sample_id].get(header, 'Missing')
                else:
                    value = 'Missing'
                metadata_values.append(value)
            reordered_metadata[header] = metadata_values

        # --- Calculate subplot layout ---
        num_annotation_rows = len(self.config['universal_headers']) + len(self.config['continuous_headers'])
        
        # Determine if we should show dendrogram (only when using clustering, not metadata ordering)
        show_dendrogram = show_dendrograms and (order_by_metadata is None)
        
        if show_dendrogram:
            total_rows = 1 + num_annotation_rows + 1  # dendrogram + annotations + heatmap
            # Heights: dendrogram (25%), annotation bars (5% each), heatmap (rest)
            dendro_height = 0.25
            annotation_height = 0.05
            heatmap_height = 1.0 - dendro_height - (num_annotation_rows * annotation_height)
            row_heights = [dendro_height] + [annotation_height] * num_annotation_rows + [heatmap_height]
        else:
            total_rows = num_annotation_rows + 1  # annotations + heatmap (no dendrogram)
            # Heights: annotation bars (8% each), heatmap (rest)
            annotation_height = 0.08
            heatmap_height = 1.0 - (num_annotation_rows * annotation_height)
            row_heights = [annotation_height] * num_annotation_rows + [heatmap_height]
        
        # --- Create figure with subplots ---
        fig = make_subplots(
            rows=total_rows,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.01,
            row_heights=row_heights
        )

        # --- Add dendrogram (top row) ---
        if show_dendrogram:
            # Store dendrogram linkage matrix for cluster extraction BEFORE processing traces
            self._store_dendrogram_info(heatmap_data, dendro_leaves)
            
            # Create a mapping of dendrogram Y coordinates to actual scipy heights
            dendro_y_to_scipy_height = self._create_dendrogram_height_mapping(dendro)
            
            for i, trace in enumerate(dendro['data']):
                trace['showlegend'] = False
                trace['hoverinfo'] = 'text'
                if 'x' in trace and 'y' in trace:
                    # Enhanced hover info with cluster extraction hints
                    trace['hovertext'] = self._create_dendrogram_hover_text(
                        trace['x'], trace['y'], i, dendro_y_to_scipy_height
                    )
                fig.add_trace(trace, row=1, col=1)

        # --- Add annotation bars (middle rows) ---
        current_row = 2 if show_dendrogram else 1  # Start after dendrogram (if present) or from row 1
        for header in self.config['universal_headers'] + self.config['continuous_headers']:
            metadata_values = reordered_metadata[header]
            
            if header in self.config['continuous_headers']:
                self._add_continuous_annotation_bar(fig, metadata_values, dendro_leaves, header, current_row)
            else:
                self._add_categorical_annotation_bar(fig, metadata_values, dendro_leaves, header, current_row)
            
            current_row += 1

        # --- Add main heatmap (bottom row) ---
        fig.add_trace(go.Heatmap(
            z=clustered_data.values,
            x=dendro_leaves,
            y=clustered_data.index,
            colorscale=colorscale,
            colorbar=dict(title='MC Probability'),
            hovertemplate='Sample: %{x}<br>MC: %{y}<br>Probability: %{z:.4f}<extra></extra>',
            xgap=1,
            ygap=1
        ), row=total_rows, col=1)

        # --- Final layout tweaks ---
        fig.update_layout(
            height=700 + (num_annotation_rows * 40),
            width=self.config['figure_width'],
            title=f"Interactive Clustered Heatmap with Metadata Annotations (K={self.k_value})",
            margin=dict(t=60, b=40),
            showlegend=False
        )
        
        # Update axes for each row
        for row_idx in range(1, total_rows + 1):
            # Don't show x-axis labels to avoid scaling issues with many samples
            fig.update_xaxes(
                showticklabels=False,
                row=row_idx, col=1
            )
            
            # Special formatting for dendrogram
            if row_idx == 1 and show_dendrogram:
                fig.update_xaxes(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    row=row_idx, col=1
                )
                fig.update_yaxes(
                    showgrid=False,
                    zeroline=False,
                    showline=False,
                    showticklabels=False,
                    row=row_idx, col=1
                )
        
        # Add axis labels to main heatmap (bottom row)
        fig.update_xaxes(
            title_text="Samples",
            title_standoff=25,
            row=total_rows, col=1
        )
        fig.update_yaxes(
            title_text="MCs", 
            title_standoff=25,
            row=total_rows, col=1
        )

        # Save if filename provided
        if custom_filename:
            # Save as SVG
            svg_path = os.path.join(self.viz_directory, f"{custom_filename}.svg")
            kaleido.write_fig_sync(fig, path=svg_path)
            print(f"✓ SVG saved: {svg_path}")
            
            # Save as PNG
            try:
                png_path = os.path.join(self.viz_directory, f"{custom_filename}.png")
                fig.write_image(png_path, width=1200, height=800, format='png')
                print(f"✓ PNG saved: {png_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save PNG: {e}")
            
            # Save as HTML
            try:
                html_path = os.path.join(self.viz_directory, f"{custom_filename}.html")
                fig.write_html(html_path)
                print(f"✓ HTML saved: {html_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save HTML: {e}")

        return fig

    def _store_dendrogram_info(self, heatmap_data: pd.DataFrame, sample_order: List[str]):
        """
        Store dendrogram and clustering information for interactive cluster extraction.
        """
        # Perform hierarchical clustering to get linkage matrix
        sample_data = heatmap_data.T.values  # Transpose to samples x topics
        distance_matrix = distance.pdist(sample_data, metric='cosine')
        linkage_matrix = hierarchy.linkage(distance_matrix, method='average')
        
        # Store information needed for interactive cluster extraction
        self.dendrogram_info = {
            'linkage_matrix': linkage_matrix,
            'sample_order': sample_order,
            'original_data': heatmap_data,
            'sample_data': sample_data
        }
        
        print(f"🔗 Stored dendrogram clustering info for interactive extraction")
    
    def _create_dendrogram_height_mapping(self, dendro: go.Figure) -> Tuple[float, float]:
        """
        Calculate the linear transformation parameters between Plotly and scipy heights.
        Plotly applies a linear scaling: plotly_y = scale * scipy_height + offset
        
        Args:
            dendro: Plotly dendrogram figure from ff.create_dendrogram
            
        Returns:
            Tuple of (scale, offset) for the linear transformation
        """
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        real_heights = linkage_matrix[:, 2]  # All scipy heights
        
        # Extract all Y coordinates from the dendrogram traces (excluding y=0 leaves)
        plotly_ys = []
        for trace in dendro['data']:
            if 'y' in trace:
                for y in trace['y']:
                    if y > 0:  # Exclude leaf nodes at y=0
                        plotly_ys.append(y)
        
        if len(plotly_ys) == 0 or len(real_heights) == 0:
            return (1.0, 0.0)  # Identity mapping if no data
        
        # Get min and max from both ranges
        min_real = real_heights.min()
        max_real = real_heights.max()
        min_plotly = min(plotly_ys)
        max_plotly = max(plotly_ys)
        
        # Calculate linear transformation: plotly_y = scale * real_h + offset
        # We have: min_plotly = scale * min_real + offset
        #          max_plotly = scale * max_real + offset
        
        if max_real > min_real and max_plotly > min_plotly:
            scale = (max_plotly - min_plotly) / (max_real - min_real)
            offset = min_plotly - scale * min_real
        else:
            scale = 1.0
            offset = 0.0
        
        print(f"📐 Dendrogram height mapping: plotly_y = {scale:.6f} * scipy_h + {offset:.6f}")
        print(f"   Scipy range: [{min_real:.6f}, {max_real:.6f}]")
        print(f"   Plotly range: [{min_plotly:.6f}, {max_plotly:.6f}]")
        
        return (scale, offset)
    
    def _create_dendrogram_hover_text(self, x_coords: List[float], y_coords: List[float], 
                                     trace_idx: int, dendro_transform: Tuple[float, float]) -> List[str]:
        """
        Create hover text for dendrogram points showing REAL scipy linkage heights.
        Maps Plotly's scaled heights back to actual scipy heights for accurate cluster cutting.
        
        Args:
            x_coords: X coordinates of dendrogram trace points
            y_coords: Y coordinates of dendrogram trace points  
            trace_idx: Index of the trace (for debugging)
            dendro_transform: Tuple of (scale, offset) for inverse transformation
        """
        hover_texts = []
        scale, offset = dendro_transform
        
        # Get the real linkage matrix for reference
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        real_heights = linkage_matrix[:, 2]  # Real scipy heights
        
        for i, (x, y) in enumerate(zip(x_coords, y_coords)):
            if y > 0:  # Only for actual dendrogram nodes (not leaf nodes)
                # Inverse transformation: scipy_height = (plotly_y - offset) / scale
                if abs(scale) > 1e-10:  # Avoid division by zero
                    real_height = (y - offset) / scale
                else:
                    real_height = real_heights.mean()  # Fallback
                
                # Find the closest actual node height for accuracy
                closest_height_idx = np.argmin(np.abs(real_heights - real_height))
                closest_real_height = real_heights[closest_height_idx]
                node_sample_count = int(linkage_matrix[closest_height_idx, 3])
                
                hover_text = f"""<b>🔗 Dendrogram Node</b><br>
<b>Real Height:</b> {closest_real_height:.6f}<br>
<b>Samples:</b> {node_sample_count}<br>
<b>Plotly Y:</b> {y:.3f}<br>
<br><i>💡 Use height {closest_real_height:.6f} for cluster cutting</i>"""
                hover_texts.append(hover_text)
            else:
                # Leaf node (sample)
                hover_texts.append(f"<b>📊 Sample</b><br>Position: {x:.0f}")
        
        return hover_texts
    
    def get_node_info(self) -> Dict[str, Any]:
        """
        Get information about all nodes in the dendrogram for debugging.
        
        Returns:
            Dictionary with node information
        """
        if not hasattr(self, 'dendrogram_info'):
            raise ValueError("No dendrogram information stored. Create a clustered heatmap first.")
        
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        
        print(f"📊 Dendrogram contains {len(linkage_matrix)} internal nodes (merges)")
        print(f"📊 Node indices range from 0 to {len(linkage_matrix)-1}")
        print("\nFirst 10 nodes:")
        for i in range(min(10, len(linkage_matrix))):
            left, right, height, count = linkage_matrix[i]
            print(f"  Node {i}: height={height:.3f}, contains {int(count)} samples")
        
        return {
            'total_nodes': len(linkage_matrix),
            'linkage_matrix': linkage_matrix,
            'sample_count': len(self.dendrogram_info['sample_order'])
        }
    
    def find_node_by_height_position(self, target_height: float, tolerance: float = 0.01) -> List[int]:
        """
        Find node indices that have a specific height (useful for mapping from hover info).
        
        Args:
            target_height: Height value from hover text
            tolerance: Tolerance for height matching
            
        Returns:
            List of node indices that match the height
        """
        if not hasattr(self, 'dendrogram_info'):
            raise ValueError("No dendrogram information stored. Create a clustered heatmap first.")
        
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        matching_nodes = []
        
        for i, (left, right, height, count) in enumerate(linkage_matrix):
            if abs(height - target_height) <= tolerance:
                matching_nodes.append(i)
        
        print(f"🔍 Found {len(matching_nodes)} nodes at height {target_height:.3f} (±{tolerance:.3f}):")
        for node_idx in matching_nodes:
            _, _, height, count = linkage_matrix[node_idx]
            print(f"  Node {node_idx}: exact height={height:.3f}, contains {int(count)} samples")
        
        return matching_nodes
    
    def get_clusters_at_height(self, height: float) -> List[List[str]]:
        """
        Get clusters by cutting the dendrogram at a specific height.
        Returns a simple list of lists where each inner list contains sample IDs for one cluster.
        
        Args:
            height: Height at which to cut the dendrogram
            
        Returns:
            List of clusters, where each cluster is a list of sample IDs
        
        Example:
            clusters = viz.get_clusters_at_height(2.5)
            # Returns: [['sample1', 'sample2'], ['sample3', 'sample4', 'sample5'], ...]
            
            # Access individual clusters
            cluster_0 = clusters[0]  # First cluster
            cluster_1 = clusters[1]  # Second cluster
        """
        if not hasattr(self, 'dendrogram_info'):
            raise ValueError("No dendrogram information stored. Create a clustered heatmap first.")
        
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        sample_names = self.dendrogram_info['sample_order']
        
        # Get height information from linkage matrix
        max_height = linkage_matrix[:, 2].max()
        min_height = linkage_matrix[:, 2].min()
        
        print(f"🎯 Cutting dendrogram at height {height:.3f}...")
        print(f"   Dendrogram height range: {min_height:.3f} to {max_height:.3f}")
        
        # Handle edge cases
        if height > max_height:
            print(f"   ⚠️  Height {height:.3f} is above maximum height {max_height:.3f}")
            print(f"   → Will result in 1 cluster containing all samples")
        elif height <= min_height:
            print(f"   ⚠️  Height {height:.3f} is at or below minimum height {min_height:.3f}")
            print(f"   → Will result in {len(sample_names)} clusters (each sample separate)")
        
        # Cut dendrogram at specified height
        cluster_assignments = fcluster(
            linkage_matrix, 
            height, 
            criterion='distance'
        )
        
        # Create list of clusters (each cluster is a list of samples)
        clusters_dict = {}
        for i, cluster_id in enumerate(cluster_assignments):
            if cluster_id not in clusters_dict:
                clusters_dict[cluster_id] = []
            clusters_dict[cluster_id].append(sample_names[i])
        
        # Convert to simple list of lists, sorted by cluster size (largest first)
        clusters_list = sorted(clusters_dict.values(), key=len, reverse=True)
        
        # Print summary
        num_clusters = len(clusters_list)
        print(f"✓ Found {num_clusters} clusters at height {height:.3f}:")
        for i, cluster in enumerate(clusters_list):
            print(f"   Cluster {i}: {len(cluster)} samples")
            if len(cluster) <= 5:
                print(f"     → {cluster}")
            else:
                print(f"     → {cluster[:3]} ... {cluster[-2:]} (showing first 3 and last 2)")
        
        return clusters_list
    
    def _get_node_descendants(self, node_idx: int, linkage_matrix: np.ndarray, sample_names: List[str]) -> List[str]:
        """
        Recursively get all sample descendants of a node in the linkage matrix.
        
        Args:
            node_idx: Index of the node in the linkage matrix
            linkage_matrix: The hierarchical clustering linkage matrix
            sample_names: List of sample names in original order
            
        Returns:
            List of sample names that are descendants of this node
        """
        n_samples = len(sample_names)
        
        def get_descendants(cluster_id):
            if cluster_id < n_samples:
                # This is a leaf node (original sample)
                return [sample_names[int(cluster_id)]]
            else:
                # This is an internal node
                node_idx = int(cluster_id - n_samples)
                if node_idx >= len(linkage_matrix):
                    return []
                
                left_child = int(linkage_matrix[node_idx, 0])
                right_child = int(linkage_matrix[node_idx, 1])
                
                left_descendants = get_descendants(left_child)
                right_descendants = get_descendants(right_child)
                
                return left_descendants + right_descendants
        
        # The cluster ID for this node is (node_idx + n_samples)
        cluster_id = node_idx + n_samples
        return get_descendants(cluster_id)
    
    def find_node_by_samples(self, target_samples: List[str]) -> List[Dict[str, Any]]:
        """
        Find dendrogram nodes that contain all the specified samples.
        
        Args:
            target_samples: List of sample IDs to find in the dendrogram
            
        Returns:
            List of dictionaries with node information (height, samples, etc.)
        """
        if not hasattr(self, 'dendrogram_info'):
            raise ValueError("No dendrogram information stored. Create a clustered heatmap first.")
        
        print(f"🔍 Finding nodes containing samples: {target_samples}")
        
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        sample_names = self.dendrogram_info['sample_order']
        target_set = set(target_samples)
        
        # Check if all target samples exist
        available_samples = set(sample_names)
        missing_samples = target_set - available_samples
        if missing_samples:
            print(f"⚠️  Warning: Samples not found in dendrogram: {missing_samples}")
            target_set = target_set & available_samples
        
        if not target_set:
            print("❌ No valid target samples found")
            return []
        
        matching_nodes = []
        
        # Check each node in the dendrogram
        for i, (left, right, height, count) in enumerate(linkage_matrix):
            node_samples = self._get_node_descendants(i, linkage_matrix, sample_names)
            node_set = set(node_samples)
            
            # Check if this node contains all target samples
            if target_set.issubset(node_set):
                matching_nodes.append({
                    'node_index': i,
                    'height': height,
                    'total_samples': len(node_samples),
                    'target_samples': list(target_set),
                    'all_samples': node_samples,
                    'contains_only_targets': node_set == target_set
                })
        
        # Sort by height (smallest first - most specific nodes)
        matching_nodes.sort(key=lambda x: x['height'])
        
        print(f"✓ Found {len(matching_nodes)} nodes containing all target samples:")
        for i, node in enumerate(matching_nodes[:5]):  # Show first 5
            specificity = "exact match" if node['contains_only_targets'] else f"{node['total_samples']} total samples"
            print(f"  {i+1}. Node {node['node_index']}: height={node['height']:.3f}, {specificity}")
        
        return matching_nodes
    
    def extract_clusters_at_height(self, height_threshold: float) -> Dict[int, List[str]]:
        """
        Extract clusters by cutting dendrogram at a specific height.
        
        Args:
            height_threshold: Height at which to cut the dendrogram
            
        Returns:
            Dictionary mapping cluster numbers to sample lists
        """
        if not hasattr(self, 'dendrogram_info'):
            raise ValueError("No dendrogram information stored. Create a clustered heatmap first.")
        
        print(f"🎯 Extracting clusters by cutting at height {height_threshold:.3f}...")
        
        # Cut dendrogram at specified height
        cluster_assignments = fcluster(
            self.dendrogram_info['linkage_matrix'], 
            height_threshold, 
            criterion='distance'
        )
        
        # Get sample names from stored order
        sample_names = self.dendrogram_info['sample_order']
        
        # Create clusters dictionary
        clusters = {}
        unique_clusters = np.unique(cluster_assignments)
        
        for cluster_id in unique_clusters:
            # Find samples in this cluster
            cluster_samples = [sample_names[i] for i, cid in enumerate(cluster_assignments) if cid == cluster_id]
            clusters[int(cluster_id)] = cluster_samples
        
        # Print summary
        num_clusters = len(clusters)
        print(f"✓ Extracted {num_clusters} clusters at height {height_threshold:.3f}:")
        for cluster_id, samples in clusters.items():
            print(f"   Cluster {cluster_id}: {len(samples)} samples")
            if len(samples) <= 5:
                print(f"     → {samples}")
            else:
                print(f"     → {samples[:3]} ... {samples[-2:]} (showing first 3 and last 2)")
        
        # Store for later use
        self.latest_clusters = clusters
        return clusters
    
    def get_cluster_suggestions(self) -> Dict[str, Any]:
        """
        Get suggestions for good cluster numbers based on dendrogram structure.
        
        Returns:
            Dictionary with cluster suggestions and metrics
        """
        if not hasattr(self, 'dendrogram_info'):
            raise ValueError("No dendrogram information stored. Create a clustered heatmap first.")
        
        print("🔍 Analyzing dendrogram structure for cluster suggestions...")
        
        linkage_matrix = self.dendrogram_info['linkage_matrix']
        suggestions = {}
        
        # Test different numbers of clusters
        for num_clusters in range(2, min(21, len(self.dendrogram_info['sample_order']) // 2)):
            try:
                cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
                
                # Calculate cluster quality metrics
                unique_clusters = len(np.unique(cluster_assignments))
                cluster_sizes = [np.sum(cluster_assignments == i) for i in range(1, num_clusters + 1)]
                
                # Silhouette score (simplified approximation)
                try:
                    from sklearn.metrics import silhouette_score
                    sil_score = silhouette_score(self.dendrogram_info['sample_data'], cluster_assignments)
                except ImportError:
                    # sklearn not available
                    sil_score = 0.0
                except Exception:
                    sil_score = 0.0
                
                suggestions[num_clusters] = {
                    'actual_clusters': unique_clusters,
                    'cluster_sizes': cluster_sizes,
                    'min_size': min(cluster_sizes) if cluster_sizes else 0,
                    'max_size': max(cluster_sizes) if cluster_sizes else 0,
                    'size_balance': min(cluster_sizes) / max(cluster_sizes) if cluster_sizes and max(cluster_sizes) > 0 else 0,
                    'silhouette_score': sil_score
                }
                
            except Exception as e:
                print(f"Warning: Could not evaluate {num_clusters} clusters: {e}")
        
        # Find best suggestions based on balance and silhouette score
        best_suggestions = []
        for k, metrics in suggestions.items():
            if metrics['actual_clusters'] == k and metrics['min_size'] >= 2:
                score = metrics['size_balance'] * 0.3 + metrics['silhouette_score'] * 0.7
                best_suggestions.append((k, score, metrics))
        
        # Sort by score
        best_suggestions.sort(key=lambda x: x[1], reverse=True)
        
        print("\n🎯 Top cluster number suggestions:")
        for i, (k, score, metrics) in enumerate(best_suggestions[:5]):
            print(f"   {i+1}. {k} clusters (score: {score:.3f})")
            print(f"      Sizes: {metrics['cluster_sizes']} (balance: {metrics['size_balance']:.2f})")
            print(f"      Silhouette: {metrics['silhouette_score']:.3f}")
        
        return {
            'all_evaluations': suggestions,
            'top_suggestions': best_suggestions[:5],
            'recommended': best_suggestions[0][0] if best_suggestions else 3
        }

    def _add_categorical_annotation_bar(self, fig, metadata_values, dendro_leaves, header, row_idx):
        """Add a categorical annotation bar following the demo pattern."""
        # Get unique categories and create color mapping
        unique_categories = list(sorted(set(metadata_values)))
        
        # Use custom colors if provided, otherwise generate default colors
        if header in self.config['custom_colors']:
            color_map = self.config['custom_colors'][header]
        else:
            # Use Plotly qualitative colors
            colors = px.colors.qualitative.Set1[:len(unique_categories)]
            if len(unique_categories) > len(colors):
                colors = colors * (len(unique_categories) // len(colors) + 1)
            color_map = dict(zip(unique_categories, colors[:len(unique_categories)]))
        
        # Create category to number mapping for discrete colorscale
        category_to_num = {cat: i for i, cat in enumerate(unique_categories)}
        
        # Build custom discrete colorscale
        colorscale = [
            [i / (len(unique_categories) - 1) if len(unique_categories) > 1 else 0, color_map[cat]]
            for i, cat in enumerate(unique_categories)
        ]
        
        # Create hover text for each cell
        hover_texts = [[f"Sample: {sample}<br>{header}: {value}" 
                       for sample, value in zip(dendro_leaves, metadata_values)]]
        
        # Add annotation bar heatmap
        fig.add_trace(go.Heatmap(
            z=[[category_to_num[val] for val in metadata_values]],  # 2D list, shape (1, N)
            x=dendro_leaves,
            y=[header],
            showscale=False,
            colorscale=colorscale,
            hoverinfo='text',
            hovertext=hover_texts,
            xgap=1,
            ygap=1
        ), row=row_idx, col=1)
    
    def _add_continuous_annotation_bar(self, fig, metadata_values, dendro_leaves, header, row_idx):
        """Add a continuous annotation bar following the demo pattern."""
        # Convert to numeric, handle non-numeric values
        numeric_values = []
        for val in metadata_values:
            try:
                if val == 'Missing' or pd.isna(val) or val == '':
                    numeric_values.append(np.nan)
                else:
                    numeric_values.append(float(val))
            except (ValueError, TypeError):
                numeric_values.append(np.nan)
        
        # Use custom colorscale if provided, otherwise default
        colorscale = self.config['continuous_cmaps'].get(header, 'Viridis')
        
        # Create hover text for each cell
        hover_texts = [[f"Sample: {sample}<br>{header}: {value:.2f}" if not pd.isna(value) 
                       else f"Sample: {sample}<br>{header}: Missing"
                       for sample, value in zip(dendro_leaves, numeric_values)]]
        
        # Add continuous annotation bar heatmap
        fig.add_trace(go.Heatmap(
            z=[numeric_values],  # 2D list, shape (1, N)
            x=dendro_leaves,
            y=[header],
            showscale=False,
            colorscale=colorscale,
            hoverinfo='text',
            hovertext=hover_texts,
            xgap=1,
            ygap=1
        ), row=row_idx, col=1)

    def _add_dendrograms_to_heatmap(self, fig, sample_linkage, topic_linkage, data):
        """Add dendrograms to the heatmap using subplots."""
        # Create dendrograms
        sample_dendro = ff.create_dendrogram(
            sample_linkage, 
            orientation='bottom',
            labels=data.columns.tolist(),
            colorscale='Blues'
        )
        
        topic_dendro = ff.create_dendrogram(
            topic_linkage, 
            orientation='left',
            labels=data.index.tolist(),
            colorscale='Blues'
        )

        # Create subplot figure with dendrograms
        fig_with_dendro = make_subplots(
            rows=2, cols=2,
            row_heights=[0.2, 0.8],
            column_widths=[0.8, 0.2],
            subplot_titles=('Sample Clustering', '', 'MC-Sample Heatmap', 'MC Clustering'),
            specs=[[{"secondary_y": False}, {"secondary_y": False}],
                   [{"secondary_y": False}, {"secondary_y": False}]]
        )

        # Add heatmap to main position
        for trace in fig.data:
            fig_with_dendro.add_trace(trace, row=2, col=1)

        # Add dendrograms
        for trace in sample_dendro.data:
            fig_with_dendro.add_trace(trace, row=1, col=1)
        
        for trace in topic_dendro.data:
            fig_with_dendro.add_trace(trace, row=2, col=2)

        # Update layout
        fig_with_dendro.update_layout(
            title=f'Interactive Clustered Heatmap with Dendrograms (K={self.k_value})',
            width=self.config['figure_width'] + 200,
            height=self.config['figure_height'] + 200,
            showlegend=False
        )

        return fig_with_dendro

    def create_topic_feature_heatmap_interactive(self,
                                               feature_level: str = 'genus_ID',
                                               use_top_tokens: bool = True,
                                               top_n: int = 10,
                                               highlight_features: Optional[Dict] = None,
                                               custom_filename: Optional[str] = None,
                                               colorscale: str = 'Blues') -> go.Figure:
        """
        Create an interactive topic-feature heatmap using taxonomic levels.
        
        This method integrates TopicFeatureProcessor functionality to create
        interactive visualizations of topic-feature relationships at specified
        taxonomic levels.

        Args:
            feature_level: Taxonomic level ('genus_ID', 'Genus', 'Family', etc.)
            use_top_tokens: Whether to show only top features per topic
            top_n: Number of top features to show if use_top_tokens=True
            highlight_features: Dict with colors as keys and feature lists as values
            custom_filename: Custom filename for output
            colorscale: Plotly colorscale name

        Returns:
            Plotly Figure object
        """
        # Initialize topic feature processor
        processor = TopicFeatureProcessor(self.base_directory, self.k_value)
        
        try:
            # Process the feature level
            grouped_probs, top_tokens = processor.process_feature_level(
                feature_level=feature_level, 
                top_n=top_n
            )
            
            # Choose data to plot
            if use_top_tokens and top_tokens is not None:
                data_to_plot = top_tokens
                plot_title = f'Top {top_n} {feature_level} Features by MC (K={self.k_value})'
            else:
                data_to_plot = grouped_probs
                plot_title = f'All {feature_level} Features by MC (K={self.k_value})'
            
            # Create hover text with detailed information
            hover_text = []
            for topic_idx, topic_name in enumerate(data_to_plot.index):
                row_hover = []
                for feature_idx, feature_name in enumerate(data_to_plot.columns):
                    probability = data_to_plot.iloc[topic_idx, feature_idx]
                    
                    hover_info = [
                        f"MC: {topic_name}",
                        f"{feature_level}: {feature_name}",
                        f"Probability: {probability:.6f}"
                    ]
                    
                    # Add taxonomy hierarchy if available
                    if self.taxa_data is not None:
                        # Find the ASV that maps to this feature
                        matching_asvs = processor.new_taxa[processor.new_taxa[feature_level] == feature_name]
                        if not matching_asvs.empty:
                            # Show taxonomy hierarchy for first matching ASV
                            first_asv = matching_asvs.iloc[0]
                            for tax_level in ['Kingdom', 'Phylum', 'Class', 'Order', 'Family', 'Genus']:
                                if tax_level in first_asv.index and pd.notna(first_asv[tax_level]):
                                    hover_info.append(f"{tax_level}: {first_asv[tax_level]}")
                    
                    row_hover.append("<br>".join(hover_info))
                hover_text.append(row_hover)

            # Create the interactive heatmap
            fig = go.Figure(data=go.Heatmap(
                z=data_to_plot.values,
                x=data_to_plot.columns,
                y=data_to_plot.index,
                colorscale=colorscale,
                hovertemplate='%{text}<extra></extra>',
                text=hover_text,
                colorbar=dict(
                    title=f"{feature_level} Probability",
                    thickness=15,
                    len=0.8
                )
            ))

            # Apply highlighting if provided
            if highlight_features:
                annotations = []
                for color, feature_list in highlight_features.items():
                    for feature in feature_list:
                        if feature in data_to_plot.columns:
                            x_pos = list(data_to_plot.columns).index(feature)
                            annotations.append(
                                dict(
                                    x=x_pos,
                                    y=-0.1,
                                    text=f"● {feature}",
                                    showarrow=False,
                                    font=dict(color=color, size=10, family="Arial Black"),
                                    xref="x",
                                    yref="paper"
                                )
                            )
                fig.update_layout(annotations=annotations)

            # Update layout with white background, no grid, and black outline
            fig.update_layout(
                title=plot_title,
                width=max(1200, len(data_to_plot.columns) * 20),
                height=600,
                xaxis_title=f"{feature_level} Features",
                yaxis_title="MCs",
                font=dict(size=12),
                plot_bgcolor='white',
                paper_bgcolor='white',
                xaxis=dict(
                    tickangle=45,
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True
                ),
                yaxis=dict(
                    showgrid=False,
                    showline=True,
                    linewidth=1,
                    linecolor='black',
                    mirror=True
                ),
                margin=dict(b=120)  # Extra bottom margin for rotated labels
            )

            # Save if filename provided
            if custom_filename:
                # Save as SVG
                svg_path = os.path.join(self.viz_directory, f"{custom_filename}.svg")
                kaleido.write_fig_sync(fig, path=svg_path)
                print(f"✓ SVG saved: {svg_path}")
                
                # Save as PNG
                try:
                    png_path = os.path.join(self.viz_directory, f"{custom_filename}.png")
                    fig.write_image(png_path, width=max(1200, len(data_to_plot.columns) * 20), height=600, format='png')
                    print(f"✓ PNG saved: {png_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save PNG: {e}")
                
                # Save as HTML
                try:
                    html_path = os.path.join(self.viz_directory, f"{custom_filename}.html")
                    fig.write_html(html_path)
                    print(f"✓ HTML saved: {html_path}")
                except Exception as e:
                    print(f"⚠️ Warning: Could not save HTML: {e}")
            
            return fig
            
        except Exception as e:
            print(f"Error creating topic-feature heatmap: {e}")
            # Fallback to basic ASV visualization
            return self._create_basic_asv_heatmap(highlight_features, custom_filename, colorscale)
    
    def _create_basic_asv_heatmap(self, highlight_features=None, custom_filename=None, colorscale='Blues'):
        """Fallback method for basic ASV visualization."""
        data = self.ASV_probabilities.copy()
        
        # Create basic hover text
        hover_text = []
        for topic_idx, topic_name in enumerate(data.index):
            row_hover = []
            for feature_idx, feature_name in enumerate(data.columns):
                probability = data.iloc[topic_idx, feature_idx]
                hover_info = [
                    f"MC: {topic_name}",
                    f"ASV: {feature_name}",
                    f"Probability: {probability:.6f}"
                ]
                row_hover.append("<br>".join(hover_info))
            hover_text.append(row_hover)

        # Create basic heatmap
        fig = go.Figure(data=go.Heatmap(
            z=data.values,
            x=data.columns,
            y=data.index,
            colorscale=colorscale,
            hovertemplate='%{text}<extra></extra>',
            text=hover_text,
            colorbar=dict(
                title="ASV Probability",
                thickness=15,
                len=0.8
            )
        ))

        fig.update_layout(
            title=f'MC-ASV Distribution (K={self.k_value})',
            width=1400,
            height=600,
            xaxis_title="ASVs",
            yaxis_title="MCs",
            font=dict(size=10),
            xaxis=dict(tickangle=45),
            margin=dict(b=100)
        )
        
        if custom_filename:
            # Save as SVG
            svg_path = os.path.join(self.viz_directory, f"{custom_filename}.svg")
            kaleido.write_fig_sync(fig, path=svg_path)
            print(f"✓ SVG saved: {svg_path}")
            
            # Save as PNG
            try:
                png_path = os.path.join(self.viz_directory, f"{custom_filename}.png")
                fig.write_image(png_path, width=1400, height=600, format='png')
                print(f"✓ PNG saved: {png_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save PNG: {e}")
            
            # Save as HTML
            try:
                html_path = os.path.join(self.viz_directory, f"{custom_filename}.html")
                fig.write_html(html_path)
                print(f"✓ HTML saved: {html_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save HTML: {e}")
            
        return fig

    def create_topic_composition_sunburst(self, 
                                        top_n_features: int = 10,
                                        custom_filename: Optional[str] = None) -> go.Figure:
        """
        Create an interactive sunburst chart showing MC composition.

        Args:
            top_n_features: Number of top features to show per MC
            custom_filename: Custom filename for output

        Returns:
            Plotly Figure object
        """
        # Get top features for each MC
        sunburst_data = []
        
        for topic_idx, topic_name in enumerate(self.ASV_probabilities.index):
            topic_probs = self.ASV_probabilities.iloc[topic_idx]
            top_features = topic_probs.nlargest(top_n_features)
            
            for feature_name, probability in top_features.items():
                # Add taxonomy information if available
                taxonomy_info = "Unknown"
                if self.taxa_data is not None and feature_name in self.taxa_data.index:
                    # Use genus if available, otherwise family, etc.
                    for tax_level in ['Genus', 'Family', 'Order', 'Class', 'Phylum']:
                        if tax_level in self.taxa_data.columns:
                            tax_value = self.taxa_data.loc[feature_name, tax_level]
                            if pd.notna(tax_value) and tax_value != '':
                                taxonomy_info = f"{tax_level}: {tax_value}"
                                break
                
                sunburst_data.append({
                    'Topic': f"MC {topic_name}",
                    'Feature': feature_name,
                    'Taxonomy': taxonomy_info,
                    'Probability': probability
                })
        
        df_sunburst = pd.DataFrame(sunburst_data)
        
        # Create sunburst chart
        fig = go.Figure(go.Sunburst(
            ids=df_sunburst['Feature'],
            labels=df_sunburst['Feature'],
            parents=df_sunburst['Topic'],
            values=df_sunburst['Probability'],
            branchvalues="total",
            hovertemplate='<b>%{label}</b><br>Probability: %{value:.4f}<br>%{customdata}<extra></extra>',
            customdata=df_sunburst['Taxonomy'],
            maxdepth=2
        ))
        
        fig.update_layout(
            title=f'MC Composition Sunburst (K={self.k_value}, Top {top_n_features} features per MC)',
            width=800,
            height=800,
            font=dict(size=12)
        )

        # Save if filename provided
        if custom_filename:
            # Save as SVG
            svg_path = os.path.join(self.viz_directory, f"{custom_filename}.svg")
            kaleido.write_fig_sync(fig, path=svg_path)
            print(f"✓ SVG saved: {svg_path}")
            
            # Save as PNG
            try:
                png_path = os.path.join(self.viz_directory, f"{custom_filename}.png")
                fig.write_image(png_path, width=800, height=800, format='png')
                print(f"✓ PNG saved: {png_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save PNG: {e}")
            
            # Save as HTML
            try:
                html_path = os.path.join(self.viz_directory, f"{custom_filename}.html")
                fig.write_html(html_path)
                print(f"✓ HTML saved: {html_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save HTML: {e}")

        return fig

    def create_sample_topic_distribution(self, 
                                       custom_filename: Optional[str] = None,
                                       group_by_metadata: Optional[str] = None,
                                       aggregate_groups: bool = False) -> go.Figure:
        """
        Create an interactive stacked bar chart of sample-MC distributions.

        Args:
            custom_filename: Custom filename for output
            group_by_metadata: Name of metadata column to group samples by (optional)
            aggregate_groups: If True, aggregate samples by group_by_metadata using mean values
                            If False, just order samples by group_by_metadata

        Returns:
            Plotly Figure object with stacked bar chart
        """
        data = self.sample_topic_df.T  # Transpose to have samples as rows, MCs as columns
        
        # Prepare sample metadata if not already done
        if not hasattr(self, 'sample_metadata'):
            self.prepare_heatmap_data()
        
        # Group and aggregate samples by metadata if requested
        if group_by_metadata is not None and aggregate_groups:
            if group_by_metadata in self.config['universal_headers'] + self.config['continuous_headers']:
                # Create mapping of samples to groups
                sample_to_group = {}
                for sample_id in data.index:
                    if sample_id in self.sample_metadata:
                        group_value = self.sample_metadata[sample_id].get(group_by_metadata, 'Missing')
                    else:
                        group_value = 'Missing'
                    sample_to_group[sample_id] = group_value
                
                # Add group column to data
                data_with_groups = data.copy()
                data_with_groups['_group'] = data_with_groups.index.map(sample_to_group)
                
                # Aggregate by groups (mean)
                aggregated_data = data_with_groups.groupby('_group').mean()
                
                # Sort groups
                aggregated_data = aggregated_data.sort_index()
                
                # Update data to use aggregated version
                data = aggregated_data
                
                print(f"✓ Samples aggregated by '{group_by_metadata}' (mean values)")
                print(f"  Groups: {list(data.index)}")
                print(f"  Sample counts per group: {data_with_groups.groupby('_group').size().to_dict()}")
                title_suffix = f" (Aggregated by {group_by_metadata})"
                
                # Create group metadata for hover
                group_sample_counts = data_with_groups.groupby('_group').size().to_dict()
                self.group_metadata = {
                    group: {group_by_metadata: group, 'Sample_Count': count}
                    for group, count in group_sample_counts.items()
                }
            else:
                print(f"Warning: '{group_by_metadata}' not found in configured headers. Using original order.")
                title_suffix = ""
        elif group_by_metadata is not None:
            # Just order by metadata (original behavior)
            if group_by_metadata in self.config['universal_headers'] + self.config['continuous_headers']:
                # Get metadata values for ordering
                sample_groups = []
                for sample_id in data.index:
                    if sample_id in self.sample_metadata:
                        group_value = self.sample_metadata[sample_id].get(group_by_metadata, 'Missing')
                    else:
                        group_value = 'Missing'
                    sample_groups.append((sample_id, group_value))
                
                # Sort by group value
                sample_groups.sort(key=lambda x: (x[1] == 'Missing', x[1]))  # Missing values last
                ordered_samples = [item[0] for item in sample_groups]
                
                # Reorder data
                data = data.reindex(ordered_samples)
                
                print(f"✓ Samples ordered by '{group_by_metadata}'")
                title_suffix = f" (Ordered by {group_by_metadata})"
            else:
                print(f"Warning: '{group_by_metadata}' not found in configured headers. Using original order.")
                title_suffix = ""
        else:
            title_suffix = ""
        
        # Create the stacked bar chart
        fig = go.Figure()
        
        for topic in data.columns:
            # Create hover text with metadata information
            hover_texts = []
            for sample_id in data.index:
                hover_parts = [
                    f'Sample: {sample_id}',
                    f'MC {topic}: {data.loc[sample_id, topic]:.4f}'
                ]
                
                # Add metadata information to hover
                if hasattr(self, 'sample_metadata') and sample_id in self.sample_metadata:
                    for header in self.config['universal_headers'] + self.config['continuous_headers']:
                        value = self.sample_metadata[sample_id].get(header, 'Missing')
                        hover_parts.append(f'{header}: {value}')
                
                hover_texts.append('<br>'.join(hover_parts))
            
            fig.add_trace(go.Bar(
                x=data.index,
                y=data[topic],
                name=f'MC {topic}',
                hovertext=hover_texts,
                hoverinfo='text'
            ))
        
        fig.update_layout(
            title=f'Sample-MC Distribution - Stacked Bars (K={self.k_value}){title_suffix}',
            xaxis_title="Samples",
            yaxis_title="MC Probability",
            barmode='stack',
            xaxis=dict(
                tickangle=45,  # Rotate sample names for better readability
                showticklabels=True  # Show x-axis labels (sample/group names)
            )
        )

        # Update common layout properties
        fig.update_layout(
            width=max(800, len(data.index) * 20),
            height=600,
            font=dict(size=12),
            hovermode='closest'
        )

        # Save if filename provided
        if custom_filename:
            # Save as SVG
            svg_path = os.path.join(self.viz_directory, f"{custom_filename}.svg")
            kaleido.write_fig_sync(fig, path=svg_path)
            print(f"✓ SVG saved: {svg_path}")
            
            # Save as PNG
            try:
                png_path = os.path.join(self.viz_directory, f"{custom_filename}.png")
                fig.write_image(png_path, width=max(800, len(data.index) * 20), height=600, format='png')
                print(f"✓ PNG saved: {png_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save PNG: {e}")
            
            # Save as HTML
            try:
                html_path = os.path.join(self.viz_directory, f"{custom_filename}.html")
                fig.write_html(html_path)
                print(f"✓ HTML saved: {html_path}")
            except Exception as e:
                print(f"⚠️ Warning: Could not save HTML: {e}")

        return fig

    def create_all_visualizations_interactive(self,
                                            custom_prefix: Optional[str] = None,
                                            highlight_taxa_dict: Optional[Dict] = None) -> Dict[str, go.Figure]:
        """
        Create all interactive visualizations for the model.

        Args:
            custom_prefix: Custom prefix for all output files
            highlight_taxa_dict: Dictionary for highlighting specific taxa

        Returns:
            Dictionary containing all visualization figures
        """
        print(f"🎨 Creating all interactive visualizations for K={self.k_value}...")
        print("=" * 60)

        results = {}

        # 1. Prepare data
        print("1️⃣ Preparing heatmap data...")
        heatmap_data = self.prepare_heatmap_data()
        
        # 2. Create clustered heatmap
        print("\n2️⃣ Creating interactive clustered heatmap...")
        filename = f"{custom_prefix}_clustered_heatmap" if custom_prefix else "clustered_heatmap_interactive"
        clustered_fig = self.create_clustered_heatmap_interactive(
            heatmap_data=heatmap_data,
            custom_filename=filename
        )
        results['clustered_heatmap'] = clustered_fig

        # 3. Create MC-feature heatmap
        print("\n3️⃣ Creating interactive MC-feature heatmap...")
        filename = f"{custom_prefix}_mc_feature" if custom_prefix else "mc_feature_interactive"
        topic_feature_fig = self.create_topic_feature_heatmap_interactive(
            feature_level='Genus',  # Default to Genus level
            use_top_tokens=True,
            top_n=15,
            highlight_features=highlight_taxa_dict,
            custom_filename=filename,
            colorscale='Blues'
        )
        results['mc_feature_heatmap'] = topic_feature_fig

        # 4. Create sample-MC distribution
        print("\n4️⃣ Creating sample-MC distribution...")
        filename = f"{custom_prefix}_sample_mc_stacked" if custom_prefix else "sample_mc_stacked"
        sample_fig = self.create_sample_topic_distribution(
            custom_filename=filename
        )
        results['sample_mc_stacked'] = sample_fig

        print("\n" + "=" * 60)
        print("🎉 All interactive visualizations created successfully!")
        print(f"📁 Outputs saved to: {self.viz_directory}")
        print("📊 SVG files saved for publication quality figures")
        print("🖼️ PNG images saved for presentations and documents")
        print("🌐 HTML files saved for interactive web viewing")
        print("=" * 60)

        return results

    def perform_hierarchical_clustering(self, 
                                      heatmap_data: Optional[pd.DataFrame] = None,
                                      method: str = 'average',
                                      metric: str = 'cosine') -> Tuple[np.ndarray, List[str]]:
        """
        Perform hierarchical clustering on samples and return linkage matrix and sample order.
        
        Args:
            heatmap_data: DataFrame to cluster (if None, will prepare data)
            method: Clustering linkage method ('average', 'complete', 'ward', 'single')
            metric: Distance metric for clustering ('cosine', 'euclidean', 'correlation', etc.)
            
        Returns:
            Tuple of (linkage_matrix, ordered_sample_list)
        """
        if heatmap_data is None:
            heatmap_data = self.prepare_heatmap_data()
        
        print(f"🔗 Performing hierarchical clustering...")
        print(f"   Method: {method}, Metric: {metric}")
        print(f"   Data shape: {heatmap_data.shape} (topics × samples)")
        
        # Transpose to have samples as rows for clustering
        sample_data = heatmap_data.T.values
        
        # Compute distance matrix
        if metric == 'euclidean' and method == 'ward':
            # Ward method requires euclidean distance
            distance_matrix = distance.pdist(sample_data, metric='euclidean')
        else:
            distance_matrix = distance.pdist(sample_data, metric=metric)
        
        # Perform hierarchical clustering
        linkage_matrix = hierarchy.linkage(distance_matrix, method=method)
        
        # Get dendrogram leaf order
        dendro = hierarchy.dendrogram(linkage_matrix, labels=heatmap_data.columns.tolist(), no_plot=True)
        ordered_samples = dendro['ivl']  # Leaves in dendrogram order
        
        print(f"✓ Clustering completed: {len(ordered_samples)} samples ordered")
        
        # Store clustering results for future use
        self.clustering_results = {
            'linkage_matrix': linkage_matrix,
            'ordered_samples': ordered_samples,
            'original_data': heatmap_data,
            'method': method,
            'metric': metric
        }
        
        return linkage_matrix, ordered_samples
    
    def extract_clusters(self, 
                        num_clusters: int,
                        heatmap_data: Optional[pd.DataFrame] = None,
                        method: str = 'average',
                        metric: str = 'cosine') -> Dict[int, List[str]]:
        """
        Extract sample clusters from hierarchical clustering.
        
        Args:
            num_clusters: Number of clusters to extract
            heatmap_data: DataFrame to cluster (if None, will use stored or prepare data)
            method: Clustering linkage method (default: 'average')
            metric: Distance metric for clustering (default: 'cosine')
            
        Returns:
            Dictionary mapping cluster numbers (1-based) to lists of sample IDs
        """
        # Use stored clustering results if available and parameters match
        if (hasattr(self, 'clustering_results') and 
            self.clustering_results.get('method') == method and 
            self.clustering_results.get('metric') == metric and
            heatmap_data is None):
            linkage_matrix = self.clustering_results['linkage_matrix']
            sample_names = self.clustering_results['original_data'].columns.tolist()
        else:
            # Perform new clustering
            linkage_matrix, ordered_samples = self.perform_hierarchical_clustering(
                heatmap_data, method, metric
            )
            sample_names = heatmap_data.columns.tolist() if heatmap_data is not None else self.sample_topic_df.columns.tolist()
        
        print(f"🎯 Extracting {num_clusters} clusters from dendrogram...")
        
        # Cut the dendrogram to get cluster assignments
        cluster_assignments = fcluster(linkage_matrix, num_clusters, criterion='maxclust')
        
        # Create dictionary mapping cluster numbers to sample lists
        clusters = {}
        for i in range(1, num_clusters + 1):
            # Find samples assigned to this cluster
            cluster_samples = [sample_names[j] for j, cluster_id in enumerate(cluster_assignments) if cluster_id == i]
            clusters[i] = cluster_samples
        
        # Print summary
        print(f"✓ Extracted {num_clusters} clusters:")
        for cluster_id, samples in clusters.items():
            print(f"   Cluster {cluster_id}: {len(samples)} samples")
            if len(samples) <= 5:
                print(f"     → {samples}")
            else:
                print(f"     → {samples[:3]} ... {samples[-2:]} (showing first 3 and last 2)")
        
        return clusters
    
    def get_cluster_metadata_summary(self, 
                                   clusters: Dict[int, List[str]],
                                   metadata_columns: Optional[List[str]] = None) -> Dict[int, Dict[str, Any]]:
        """
        Get metadata summary for each cluster.
        
        Args:
            clusters: Dictionary mapping cluster numbers to sample lists
            metadata_columns: List of metadata columns to summarize (if None, use all configured)
            
        Returns:
            Dictionary with cluster summaries
        """
        if metadata_columns is None:
            metadata_columns = self.config['universal_headers'] + self.config['continuous_headers']
        
        # Prepare metadata if not already done
        if not hasattr(self, 'sample_metadata'):
            self.prepare_heatmap_data()
        
        print(f"📊 Summarizing metadata for {len(clusters)} clusters...")
        
        cluster_summaries = {}
        
        for cluster_id, sample_list in clusters.items():
            print(f"\nCluster {cluster_id} ({len(sample_list)} samples):")
            cluster_summary = {
                'sample_count': len(sample_list),
                'samples': sample_list,
                'metadata_summary': {}
            }
            
            for col in metadata_columns:
                if col in self.config['universal_headers']:
                    # Categorical variable - count frequencies
                    values = []
                    for sample_id in sample_list:
                        if sample_id in self.sample_metadata:
                            values.append(self.sample_metadata[sample_id].get(col, 'Missing'))
                        else:
                            values.append('Missing')
                    
                    value_counts = pd.Series(values).value_counts()
                    cluster_summary['metadata_summary'][col] = {
                        'type': 'categorical',
                        'counts': value_counts.to_dict(),
                        'most_common': value_counts.index[0] if len(value_counts) > 0 else 'Missing',
                        'diversity': len(value_counts)
                    }
                    
                    print(f"  {col}: {dict(value_counts)} (most common: {value_counts.index[0]})")
                    
                elif col in self.config['continuous_headers']:
                    # Continuous variable - calculate statistics
                    values = []
                    for sample_id in sample_list:
                        if sample_id in self.sample_metadata:
                            val = self.sample_metadata[sample_id].get(col, np.nan)
                            try:
                                if val != 'Missing' and not pd.isna(val):
                                    values.append(float(val))
                            except (ValueError, TypeError):
                                pass
                    
                    if values:
                        values_series = pd.Series(values)
                        cluster_summary['metadata_summary'][col] = {
                            'type': 'continuous',
                            'mean': values_series.mean(),
                            'std': values_series.std(),
                            'median': values_series.median(),
                            'min': values_series.min(),
                            'max': values_series.max(),
                            'count': len(values)
                        }
                        print(f"  {col}: mean={values_series.mean():.2f}±{values_series.std():.2f}, median={values_series.median():.2f} (n={len(values)})")
                    else:
                        cluster_summary['metadata_summary'][col] = {
                            'type': 'continuous',
                            'mean': np.nan,
                            'std': np.nan,
                            'median': np.nan,
                            'min': np.nan,
                            'max': np.nan,
                            'count': 0
                        }
                        print(f"  {col}: No valid numeric values")
            
            cluster_summaries[cluster_id] = cluster_summary
        
        return cluster_summaries
    
    def visualize_clusters_with_metadata(self, 
                                       num_clusters: int,
                                       metadata_columns: Optional[List[str]] = None,
                                       custom_filename: Optional[str] = None) -> go.Figure:
        """
        Create a visualization showing clusters with their metadata characteristics.
        
        Args:
            num_clusters: Number of clusters to extract
            metadata_columns: List of metadata columns to visualize
            custom_filename: Custom filename for output
            
        Returns:
            Plotly Figure with cluster visualization
        """
        # Extract clusters
        clusters = self.extract_clusters(num_clusters)
        
        # Get cluster metadata summaries
        cluster_summaries = self.get_cluster_metadata_summary(clusters, metadata_columns)
        
        # Create visualization - this is a simple bar chart showing cluster sizes
        # You can extend this to show more sophisticated visualizations
        
        cluster_ids = list(clusters.keys())
        cluster_sizes = [len(clusters[cid]) for cid in cluster_ids]
        
        fig = go.Figure(data=[
            go.Bar(
                x=[f"Cluster {cid}" for cid in cluster_ids],
                y=cluster_sizes,
                text=[f"n={size}" for size in cluster_sizes],
                textposition='auto',
                hovertemplate='<b>%{x}</b><br>Samples: %{y}<extra></extra>'
            )
        ])
        
        fig.update_layout(
            title=f'Sample Clusters from Dendrogram (K={self.k_value}, {num_clusters} clusters)',
            xaxis_title='Clusters',
            yaxis_title='Number of Samples',
            width=800,
            height=500
        )
        
        # Save if filename provided
        if custom_filename:
            svg_path = os.path.join(self.viz_directory, f"{custom_filename}_cluster_sizes.svg")
            kaleido.write_fig_sync(fig, path=svg_path)
            print(f"✓ Cluster visualization saved: {svg_path}")
        
        # Store results for access
        self.latest_clusters = clusters
        self.latest_cluster_summaries = cluster_summaries
        
        return fig
    
    def get_samples_from_cluster(self, cluster_number: int) -> List[str]:
        """
        Get sample IDs from a specific cluster number.
        
        Args:
            cluster_number: Cluster number (1-based)
            
        Returns:
            List of sample IDs in that cluster
        """
        if not hasattr(self, 'latest_clusters') or self.latest_clusters is None:
            raise ValueError("No clusters extracted yet. Run extract_clusters() or visualize_clusters_with_metadata() first.")
        
        if cluster_number not in self.latest_clusters:
            available_clusters = list(self.latest_clusters.keys())
            raise ValueError(f"Cluster {cluster_number} not found. Available clusters: {available_clusters}")
        
        samples = self.latest_clusters[cluster_number]
        print(f"✓ Cluster {cluster_number} contains {len(samples)} samples:")
        print(f"  {samples}")
        
        return samples
    
    def export_cluster_results(self, 
                             output_filename: Optional[str] = None) -> str:
        """
        Export cluster results to CSV files.
        
        Args:
            output_filename: Base filename for export (without extension)
            
        Returns:
            Path to the main output file
        """
        if not hasattr(self, 'latest_clusters') or self.latest_clusters is None:
            raise ValueError("No clusters extracted yet. Run extract_clusters() or visualize_clusters_with_metadata() first.")
        
        if output_filename is None:
            output_filename = f"cluster_results_K{self.k_value}_{len(self.latest_clusters)}clusters"
        
        # Export sample-cluster assignments
        sample_cluster_data = []
        for cluster_id, samples in self.latest_clusters.items():
            for sample_id in samples:
                sample_cluster_data.append({
                    'Sample_ID': sample_id,
                    'Cluster': cluster_id
                })
        
        sample_cluster_df = pd.DataFrame(sample_cluster_data)
        
        # Add metadata if available
        if hasattr(self, 'sample_metadata'):
            for sample_id in sample_cluster_df['Sample_ID']:
                if sample_id in self.sample_metadata:
                    for key, value in self.sample_metadata[sample_id].items():
                        if key not in sample_cluster_df.columns:
                            sample_cluster_df[key] = np.nan
                        sample_cluster_df.loc[sample_cluster_df['Sample_ID'] == sample_id, key] = value
        
        # Save main results
        output_path = os.path.join(self.viz_directory, f"{output_filename}.csv")
        sample_cluster_df.to_csv(output_path, index=False)
        print(f"✓ Cluster assignments saved: {output_path}")
        
        # Save cluster summaries if available
        if hasattr(self, 'latest_cluster_summaries'):
            summary_path = os.path.join(self.viz_directory, f"{output_filename}_summary.json")
            with open(summary_path, 'w') as f:
                # Convert numpy types to JSON serializable
                json_summaries = {}
                for cluster_id, summary in self.latest_cluster_summaries.items():
                    json_summary = {'sample_count': summary['sample_count'], 'samples': summary['samples'], 'metadata_summary': {}}
                    for col, meta_info in summary['metadata_summary'].items():
                        json_meta = {}
                        for key, value in meta_info.items():
                            if isinstance(value, (np.integer, np.floating)):
                                json_meta[key] = float(value)
                            elif pd.isna(value):
                                json_meta[key] = None
                            else:
                                json_meta[key] = value
                        json_summary['metadata_summary'][col] = json_meta
                    json_summaries[str(cluster_id)] = json_summary
                
                import json
                json.dump(json_summaries, f, indent=2)
            print(f"✓ Cluster summaries saved: {summary_path}")
        
        return output_path

    def get_summary(self) -> Dict[str, Any]:
        """
        Get a summary of the loaded model and configuration.

        Returns:
            Dictionary containing model summary information
        """
        return {
            'k_value': self.k_value,
            'base_directory': self.base_directory,
            'visualization_type': 'Interactive (Plotly)',
            'num_samples': len(self.DM_distributions),
            'num_topics': len(self.DM_distributions[0]) if self.DM_distributions else 0,
            'num_features': self.ASV_probabilities.shape[1] if hasattr(self, 'ASV_probabilities') else 0,
            'metadata_columns': list(self.metadata.columns) if hasattr(self, 'metadata') else [],
            'configured_headers': {
                'universal': self.config['universal_headers'],
                'continuous': self.config['continuous_headers']
            },
            'visualization_config': {
                'top_asv_count': self.config['top_asv_count'],
                'figure_width': self.config['figure_width'],
                'figure_height': self.config['figure_height'],
                'colorscale': self.config['heatmap_colorscale']
            },
            'output_directory': self.viz_directory
        }


# Convenience function for quick visualization
def create_interactive_lda_visualization(base_directory: str, 
                                       k_value: int, 
                                       metadata_path: str,
                                       universal_headers: List[str],
                                       continuous_headers: Optional[List[str]] = None,
                                       custom_prefix: Optional[str] = None,
                                       highlight_taxa: Optional[Dict] = None) -> Dict[str, go.Figure]:
    """
    Convenience function to quickly create all interactive LDA visualizations.
    
    Args:
        base_directory: Base directory where LDA results are stored
        k_value: Number of topics (K) to visualize
        metadata_path: Path to metadata CSV file
        universal_headers: List of categorical metadata columns
        continuous_headers: List of continuous metadata columns (optional)
        custom_prefix: Custom prefix for output files
        highlight_taxa: Dictionary for highlighting specific taxa
    
    Returns:
        Dictionary containing all visualization figures
    """
    # Create visualizer
    viz = LDAModelVisualizerInteractive(
        base_directory=base_directory,
        k_value=k_value,
        metadata_path=metadata_path,
        universal_headers=universal_headers,
        continuous_headers=continuous_headers
    )
    
    # Create all visualizations
    results = viz.create_all_visualizations_interactive(
        custom_prefix=custom_prefix,
        highlight_taxa_dict=highlight_taxa
    )
    
    return results


class MCComparison:
    """
    A class for comparing two microbial components (MCs) by analyzing samples with high representation. 
    Provides functionality to plot metadata and feature comparisons for selected MCs.
    
    Enhanced version with:
    - Automatic taxa mapping from random IDs to taxonomic names
    - Proper MC index naming conversion
    - Flexible taxonomic level selection
    - Better error handling and validation
    """
    
    def __init__(self, base_directory: str, metadata_path: str, taxonomic_level: str = 'genus_ID'):
        """
        Initialize the MCComparison class.
        
        Args:
            base_directory: Base directory containing LDA results and intermediate files
            metadata_path: Path to metadata CSV file
            taxonomic_level: Taxonomic level to use for feature mapping (default: 'genus_ID')
        """
        self.base_directory = base_directory
        self.metadata_path = metadata_path
        self.taxonomic_level = taxonomic_level
        
        # Load metadata
        self.metadata_df = pd.read_csv(metadata_path, index_col=0)
        
        # Load taxa mapping
        self.taxa_df = None
        self.random_asvid_mapping = None
        self._load_taxa_mapping()
        
        # Cache for processed data
        self._mc_feature_cache = {}
        self._mc_sample_cache = {}
        
        print(f"MCComparison initialized:")
        print(f"  Base directory: {self.base_directory}")
        print(f"  Taxonomic level: {self.taxonomic_level}")
        print(f"  Metadata shape: {self.metadata_df.shape}")
        print(f"  Taxa mapping: {len(self.random_asvid_mapping) if self.random_asvid_mapping else 0} ASVs")

    def _load_taxa_mapping(self):
        """Load taxa mapping from intermediate_taxa.csv and create random ID mapping."""
        taxa_path = os.path.join(self.base_directory, 'intermediate', 'intermediate_taxa.csv')
        
        try:
            self.taxa_df = pd.read_csv(taxa_path, index_col=0)
            print(f"✓ Loaded taxa data: {self.taxa_df.shape}")
            
            # Validate taxonomic level
            if self.taxonomic_level not in self.taxa_df.columns:
                available_levels = [col for col in self.taxa_df.columns if col != 'randomID']
                raise ValueError(f"Taxonomic level '{self.taxonomic_level}' not found. Available: {available_levels}")
            
            # Create mapping from randomID to taxonomic level
            self.random_asvid_mapping = dict(zip(self.taxa_df['randomID'], self.taxa_df[self.taxonomic_level]))
            print(f"✓ Created mapping for {len(self.random_asvid_mapping)} ASVs to {self.taxonomic_level}")
            
        except FileNotFoundError:
            raise FileNotFoundError(f"Taxa mapping file not found: {taxa_path}")
        except Exception as e:
            raise Exception(f"Error loading taxa mapping: {e}")

    def _convert_mc_index(self, mc_asv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Convert MC index from raw format to proper K{value}_MC{number} format.
        
        Args:
            mc_asv_df: DataFrame with raw MC indices
            
        Returns:
            DataFrame with converted indices
        """
        old_index = mc_asv_df.index.values.tolist()
        
        # Convert index format: from raw to K{value}_MC{number}
        new_names = []
        for item in old_index:
            try:
                parts = item.split('_')
                k_value = int(parts[0]) - 1  # Adjust K value
                mc_number = int(parts[1]) - 1  # Adjust MC number
                new_name = f"K{k_value}_MC{mc_number}"
                new_names.append(new_name)
            except (ValueError, IndexError) as e:
                print(f"Warning: Could not parse index '{item}': {e}")
                new_names.append(item)  # Keep original if parsing fails
        
        mc_asv_df_converted = mc_asv_df.copy()
        mc_asv_df_converted.index = new_names
        
        print(f"✓ Converted {len(new_names)} MC indices")
        return mc_asv_df_converted

    def _apply_taxa_mapping(self, mc_asv_df: pd.DataFrame) -> pd.DataFrame:
        """
        Apply taxa mapping to convert random IDs to taxonomic names.
        
        Args:
            mc_asv_df: DataFrame with random ID columns
            
        Returns:
            DataFrame with taxonomic names as columns
        """
        mc_asv_mapped = mc_asv_df.copy()
        
        # Map column names from random IDs to taxonomic names
        original_columns = mc_asv_mapped.columns.tolist()
        mc_asv_mapped.columns = mc_asv_mapped.columns.map(self.random_asvid_mapping)
        
        # Count successful mappings
        mapped_count = sum(1 for orig_col in original_columns if orig_col in self.random_asvid_mapping)
        
        print(f"✓ Mapped {mapped_count}/{len(original_columns)} columns to {self.taxonomic_level}")
        
        # Group by taxonomic level and sum (in case of duplicates)
        mc_asv_grouped = mc_asv_mapped.groupby(level=0, axis=1).sum()
        
        if mc_asv_grouped.shape[1] != mc_asv_mapped.shape[1]:
            print(f"✓ Grouped {mc_asv_mapped.shape[1]} → {mc_asv_grouped.shape[1]} unique {self.taxonomic_level} features")
        
        return mc_asv_grouped

    def _load_and_process_mc_feature_data(self, k_value: str) -> pd.DataFrame:
        """
        Load and process MC-feature probability data with proper mapping.
        
        Args:
            k_value: K value as string (e.g., '5')
            
        Returns:
            Processed DataFrame with proper indices and taxonomic columns
        """
        if k_value in self._mc_feature_cache:
            return self._mc_feature_cache[k_value]
        
        # Load raw MC-feature probabilities
        filepath = os.path.join(self.base_directory, 'lda_results', 'MC_Feature', 
                               f'MC_Feature_Probabilities_{k_value}.csv')
        
        try:
            mc_feature_df = pd.read_csv(filepath, index_col=0)
            print(f"✓ Loaded MC-feature data: {mc_feature_df.shape}")
            
            # Apply taxa mapping to columns
            mc_feature_mapped = self._apply_taxa_mapping(mc_feature_df)
            
            # Convert index format
            mc_feature_final = self._convert_mc_index(mc_feature_mapped)
            
            # Cache the result
            self._mc_feature_cache[k_value] = mc_feature_final
            
            return mc_feature_final
            
        except FileNotFoundError:
            raise FileNotFoundError(f"MC-feature file not found: {filepath}")

    def _load_and_process_mc_sample_data(self, k_value: str) -> pd.DataFrame:
        """
        Load and process MC-sample probability data with proper mapping.
        
        Args:
            k_value: K value as string (e.g., '5')
            
        Returns:
            Processed DataFrame with proper indices
        """
        if k_value in self._mc_sample_cache:
            return self._mc_sample_cache[k_value]
        
        # Load raw MC-sample probabilities
        filepath = os.path.join(self.base_directory, 'lda_results', 'MC_Sample', 
                               f'MC_Sample_probabilities{k_value}.csv')
        
        try:
            mc_sample_df = pd.read_csv(filepath, index_col=0)
            print(f"✓ Loaded MC-sample data: {mc_sample_df.shape}")
            
            # Convert index format
            mc_sample_final = self._convert_mc_index(mc_sample_df)
            
            # Cache the result
            self._mc_sample_cache[k_value] = mc_sample_final
            
            return mc_sample_final
            
        except FileNotFoundError:
            raise FileNotFoundError(f"MC-sample file not found: {filepath}")

    def get_high_representation_samples(self, topic_id: str, high_threshold: float = 0.67) -> List[str]:
        """
        Retrieve samples with high representation for a specific topic.

        Args:
            topic_id: Topic ID in format 'K5_MC1'
            high_threshold: Probability threshold for high representation

        Returns:
            List of sample IDs with high representation
        """
        k_value = topic_id.split('_')[0][1:]  # Extract K value
        
        # Load processed data
        sample_mc_df = self._load_and_process_mc_sample_data(k_value)
        
        # Validate topic ID exists
        if topic_id not in sample_mc_df.index:
            available_topics = sample_mc_df.index.tolist()
            raise ValueError(f"Topic ID '{topic_id}' not found. Available: {available_topics}")
        
        # Get samples with high representation
        mc_samples = sample_mc_df.loc[topic_id]
        high_samples = mc_samples[mc_samples > high_threshold].index.tolist()
        
        print(f"✓ Found {len(high_samples)} samples with >{high_threshold:.0%} representation in {topic_id}")
        
        return high_samples

    def get_feature_probabilities(self, mc_id: str) -> pd.Series:
        """
        Retrieve feature probabilities for a given MC with proper taxonomic mapping.

        Args:
            mc_id: MC ID, e.g., 'K5_MC1'

        Returns:
            A Series with feature probabilities using taxonomic names
        """
        k_value = mc_id.split('_')[0][1:]  # Extract the K value
        
        # Load processed data
        mc_feature_df = self._load_and_process_mc_feature_data(k_value)
        
        # Validate MC ID exists
        if mc_id not in mc_feature_df.index:
            available_mcs = mc_feature_df.index.tolist()
            raise ValueError(f"MC ID '{mc_id}' not found. Available: {available_mcs}")
        
        feature_probs = mc_feature_df.loc[mc_id]
        
        print(f"✓ Retrieved feature probabilities for {mc_id} ({len(feature_probs)} {self.taxonomic_level} features)")
        
        return feature_probs

    def compare_metadata(self, samples1: List[str], samples2: List[str], features: List[str]) -> Tuple[pd.DataFrame, pd.DataFrame]:
        """
        Compare metadata between two sample groups.

        Args:
            samples1: List of sample IDs for MC1
            samples2: List of sample IDs for MC2
            features: List of metadata features to compare
            
        Returns:
            Two DataFrames containing metadata for each group
        """
        # Validate features exist
        available_features = [f for f in features if f in self.metadata_df.columns]
        missing_features = [f for f in features if f not in self.metadata_df.columns]
        
        if missing_features:
            print(f"Warning: Features not found in metadata: {missing_features}")
            print(f"Available features: {list(self.metadata_df.columns)}")
        
        if not available_features:
            raise ValueError("No valid features found in metadata")
        
        # Get metadata for both groups, handling missing samples
        valid_samples1 = [s for s in samples1 if s in self.metadata_df.index]
        valid_samples2 = [s for s in samples2 if s in self.metadata_df.index]
        
        if len(valid_samples1) != len(samples1):
            print(f"Warning: {len(samples1) - len(valid_samples1)} samples from group 1 not found in metadata")
        
        if len(valid_samples2) != len(samples2):
            print(f"Warning: {len(samples2) - len(valid_samples2)} samples from group 2 not found in metadata")
        
        group1_metadata = self.metadata_df.loc[valid_samples1, available_features]
        group2_metadata = self.metadata_df.loc[valid_samples2, available_features]
        
        return group1_metadata, group2_metadata

    def plot_metadata_comparison(self, df1: pd.DataFrame, df2: pd.DataFrame, feature: str, 
                               mc1_name: str = "MC1", mc2_name: str = "MC2"):
        """
        Plot a comparison bar chart for a specific metadata feature.

        Args:
            df1: DataFrame for MC1
            df2: DataFrame for MC2
            feature: The metadata feature to compare
            mc1_name: Name for MC1 in the plot
            mc2_name: Name for MC2 in the plot
        """
        if feature not in df1.columns or feature not in df2.columns:
            print(f"Warning: Feature '{feature}' not found in one or both datasets")
            return
        
        # Calculate proportions
        mc1_counts = df1[feature].value_counts(normalize=True)
        mc2_counts = df2[feature].value_counts(normalize=True)
        
        # Combine data
        combined_df = pd.concat([mc1_counts, mc2_counts], axis=1, sort=False)
        combined_df.columns = [mc1_name, mc2_name]
        combined_df = combined_df.fillna(0)  # Fill missing values with 0
        
        # Create plot
        fig, ax = plt.subplots(figsize=(10, 6))
        combined_df.plot(kind='bar', ax=ax, color=['skyblue', 'lightcoral'])
        
        plt.title(f"Comparison of {feature}\n({mc1_name}: n={len(df1)}, {mc2_name}: n={len(df2)})")
        plt.ylabel('Proportion')
        plt.xlabel(feature)
        plt.xticks(rotation=45, ha='right')
        plt.legend(title='Microbial Component')
        plt.tight_layout()
        plt.show()

    def plot_top_features_heatmap(self, mc1_probs: pd.Series, mc2_probs: pd.Series, 
                                top_n: int = 10, mc1_name: str = "MC1", mc2_name: str = "MC2"):
        """
        Plot heatmap comparing top features from two MCs by probabilities

        Args:
            mc1_probs: Series containing MC1 feature probabilities
            mc2_probs: Series containing MC2 feature probabilities
            top_n: Number of top features to display
            mc1_name: Name for MC1 in the plot
            mc2_name: Name for MC2 in the plot
        """
        # Get top features from both MCs
        all_top_features = pd.concat([mc1_probs.nlargest(top_n), mc2_probs.nlargest(top_n)])
        
        # Sort by mean probability across both MCs
        sorted_features = all_top_features.groupby(all_top_features.index).mean().sort_values(ascending=False)
        top_features = sorted_features.head(top_n * 2).index
        
        # Create heatmap data
        heatmap_data = pd.DataFrame({
            mc1_name: mc1_probs.reindex(top_features, fill_value=0),
            mc2_name: mc2_probs.reindex(top_features, fill_value=0)
        })
        
        # Create plot
        plt.figure(figsize=(8, max(6, len(top_features) * 0.4)))
        
        sns.heatmap(
            heatmap_data, 
            annot=True, 
            cmap='Blues', 
            cbar_kws={'label': f'Probability ({self.taxonomic_level})'},
            fmt='.3f',
            linewidths=0.5
        )
        
        plt.title(f"Top {self.taxonomic_level} Features Comparison\n{mc1_name} vs {mc2_name}")
        plt.xlabel("Microbial Components")
        plt.ylabel(f"{self.taxonomic_level} Features")
        plt.tight_layout()
        plt.show()

    def get_feature_summary(self, mc_id: str, top_n: int = 10) -> Dict[str, Any]:
        """
        Get a summary of top features for a specific MC.
        
        Args:
            mc_id: MC ID (e.g., 'K5_MC1')
            top_n: Number of top features to include
            
        Returns:
            Dictionary with feature summary
        """
        feature_probs = self.get_feature_probabilities(mc_id)
        top_features = feature_probs.nlargest(top_n)
        
        summary = {
            'mc_id': mc_id,
            'taxonomic_level': self.taxonomic_level,
            'total_features': len(feature_probs),
            'top_features': top_features.to_dict(),
            'max_probability': feature_probs.max(),
            'total_probability_mass': feature_probs.sum(),
            'non_zero_features': (feature_probs > 0).sum(),
            'entropy': -np.sum(feature_probs * np.log2(feature_probs + 1e-10))  # Shannon entropy
        }
        
        return summary

    def compare_two_mcs(self, mc1_id: str, mc2_id: str, metadata: Optional[List[str]] = None, 
                       high_threshold: float = 0.67, top_n: int = 10):
        """
        Compare two MCs by their metadata and top features.

        Args:
            mc1_id: ID of the first MC
            mc2_id: ID of the second MC
            metadata: List of metadata columns to compare (if None, will ask for column names)
            high_threshold: Threshold for high representation samples
            top_n: Number of top features to display in heatmap
        """
        print(f"COMPARING {mc1_id} vs {mc2_id}")
        print("=" * 50)
        
        # Check if metadata columns are provided
        if metadata is None:
            available_columns = list(self.metadata_df.columns)
            print(f"\nNo metadata columns specified for comparison.")
            print(f"Available metadata columns: {available_columns}")
            print(f"\nPlease specify the 'metadata' parameter with a list of column names you want to compare.")
            print(f"Example: metadata=['Diagnosis', 'Age', 'Location']")
            print(f"\nSkipping metadata comparison and proceeding with taxonomic analysis only...")
            metadata = []  # Empty list to skip metadata comparison
        
        # Get high representation samples
        print(f"1. Finding high representation samples (threshold: {high_threshold:.0%})")
        samples1 = self.get_high_representation_samples(mc1_id, high_threshold)
        samples2 = self.get_high_representation_samples(mc2_id, high_threshold)
        
        print(f"   {mc1_id}: {len(samples1)} samples")
        print(f"   {mc2_id}: {len(samples2)} samples")
        print(f"   Overlap: {len(set(samples1) & set(samples2))} samples")
        
        # Compare metadata if samples and metadata columns are available
        if metadata and samples1 and samples2:
            print(f"\n2. Comparing metadata columns: {metadata}")
            group1_metadata, group2_metadata = self.compare_metadata(samples1, samples2, metadata)
            
            # Plot metadata comparison
            for feature in metadata:
                if feature in group1_metadata.columns and feature in group2_metadata.columns:
                    self.plot_metadata_comparison(group1_metadata, group2_metadata, feature, mc1_id, mc2_id)
        
        # Compare feature probabilities
        print(f"\n3. Comparing {self.taxonomic_level} feature probabilities")
        mc1_probs = self.get_feature_probabilities(mc1_id)
        mc2_probs = self.get_feature_probabilities(mc2_id)
        
        # Display top features
        print(f"\n   Top 5 {self.taxonomic_level} features:")
        print(f"   {mc1_id}:")
        for feature, prob in mc1_probs.nlargest(5).items():
            print(f"     {feature}: {prob:.4f}")
        
        print(f"   {mc2_id}:")
        for feature, prob in mc2_probs.nlargest(5).items():
            print(f"     {feature}: {prob:.4f}")
        
        # Plot top features heatmap
        print(f"\n4. Creating feature comparison heatmap (top {top_n})")
        self.plot_top_features_heatmap(mc1_probs, mc2_probs, top_n, mc1_id, mc2_id)
        
        # Find distinctive features
        print(f"\n5. Identifying distinctive features")
        threshold = 0.01  # Minimum probability to consider
        
        mc1_significant = mc1_probs[mc1_probs > threshold]
        mc2_significant = mc2_probs[mc2_probs > threshold]
        
        # Features high in MC1 but low in MC2
        mc1_distinctive = mc1_significant[mc1_significant.index.isin(mc2_probs[mc2_probs < threshold/2].index)]
        
        # Features high in MC2 but low in MC1  
        mc2_distinctive = mc2_significant[mc2_significant.index.isin(mc1_probs[mc1_probs < threshold/2].index)]
        
        if len(mc1_distinctive) > 0:
            print(f"   {mc1_id} distinctive features ({len(mc1_distinctive)}):")
            for feature, prob in mc1_distinctive.nlargest(3).items():
                print(f"     {feature}: {prob:.4f}")
        
        if len(mc2_distinctive) > 0:
            print(f"   {mc2_id} distinctive features ({len(mc2_distinctive)}):")
            for feature, prob in mc2_distinctive.nlargest(3).items():
                print(f"     {feature}: {prob:.4f}")
        
        print(f"\n✓ Comparison completed!")
        
        return {
            'mc1_samples': len(samples1),
            'mc2_samples': len(samples2),
            'mc1_distinctive': mc1_distinctive.to_dict() if len(mc1_distinctive) > 0 else {},
            'mc2_distinctive': mc2_distinctive.to_dict() if len(mc2_distinctive) > 0 else {}
        }

    def get_available_topics(self, k_value: str) -> List[str]:
        """
        Get list of available topics for a given K value.
        
        Args:
            k_value: K value as string (e.g., '5')
            
        Returns:
            List of available topic IDs
        """
        mc_sample_df = self._load_and_process_mc_sample_data(k_value)
        return mc_sample_df.index.tolist()

    def set_taxonomic_level(self, new_level: str):
        """
        Change the taxonomic level and clear cache.
        
        Args:
            new_level: New taxonomic level to use
        """
        if new_level not in self.taxa_df.columns:
            available_levels = [col for col in self.taxa_df.columns if col != 'randomID']
            raise ValueError(f"Taxonomic level '{new_level}' not found. Available: {available_levels}")
        
        self.taxonomic_level = new_level
        self.random_asvid_mapping = dict(zip(self.taxa_df['randomID'], self.taxa_df[new_level]))
        
        # Clear cache to force reprocessing with new taxonomic level
        self._mc_feature_cache.clear()
        
        print(f"✓ Taxonomic level changed to: {new_level}")
        print(f"✓ Cache cleared - next operations will use {new_level} level")
