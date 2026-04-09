"""
LDA4Microbiome: A comprehensive workflow for LDA analysis of microbiome data using MALLET

This package provides tools for:
- Taxonomic data preprocessing
- LDA model training with MALLET
- Model selection and evaluation
- Results visualization and analysis
- Interactive Sankey diagrams with sample tracing (StripeSankeyInline)
"""

__version__ = "0.1.21"
__author__ = "Peiyang Huo"
__email__ = "peiyang.huo@kuleuven.be"

# Import main classes for easy access
# Using the renamed module files for proper imports
from .preprocessing import TaxonomyProcessor
from .training import LDATrainer
from .selection import SankeyDataProcessor
from .visualization import TopicFeatureProcessor, LDAModelVisualizerInteractive, MCComparison
from .stripesankey import StripeSankeyInline
from .stripesankey_v2 import StripeSankeyV2
from .stripesankey_v3 import StripeSankeyV3

__all__ = [
    'TaxonomyProcessor',
    'LDATrainer', 
    'SankeyDataProcessor',
    'TopicFeatureProcessor',
    'LDAModelVisualizerInteractive',
    'MCComparison',
    'StripeSankeyInline',
    'StripeSankeyV2',
    'StripeSankeyV3',
]

