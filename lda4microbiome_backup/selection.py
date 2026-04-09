"""
Model selection and Sankey diagram data processing - IMPROVED VERSION.

This module contains classes for:
- Processing LDA results into Sankey diagram data
- Using comprehensive metrics CSV files (much simpler than MALLET XML parsing)
- Model selection based on metrics
"""

import os
import pandas as pd
import numpy as np
import json
import glob
import re
from collections import defaultdict
from typing import List, Dict, Optional

class SankeyDataProcessor:
    """
    A class for processing LDA results into Sankey diagram data.

    This improved class integrates with LDATrainer to automatically load and process:
    - Sample-MC probability data
    - Comprehensive metrics data (perplexity and coherence per topic)
    - Flow calculations between different K values
    
    IMPROVEMENTS over original:
    - Uses comprehensive_MC_metrics_X-Y.csv instead of parsing MALLET XML files
    - Simplified metrics loading (no XML parsing needed)
    - Both perplexity and coherence are available per topic
    - More robust and easier to maintain
    """

    def __init__(self, base_directory: str, MC_range: List[int], 
                 high_threshold: float = 0.67, medium_threshold: float = 0.33,
                 range_str: Optional[str] = None):
        """
        Initialize the Sankey Data Processor.

        Args:
            base_directory: Base directory where LDATrainer saved results
            MC_range: List of K values (number of topics) to process
            high_threshold: Threshold for high representation (default 0.67)
            medium_threshold: Threshold for medium representation (default 0.33)
            range_str: String representation of range for file naming
        """
        self.base_directory = base_directory
        self.MC_range = MC_range
        self.high_threshold = high_threshold
        self.medium_threshold = medium_threshold

        if range_str is None:
            self.range_str = f"{min(MC_range)}-{max(MC_range)}"
        else:
            self.range_str = range_str

        # Set up paths automatically based on base_directory
        self._setup_paths()

        print(f"SankeyDataProcessor (Improved) initialized:")
        print(f"  Base directory: {self.base_directory}")
        print(f"  K range: {self.MC_range}")
        print(f"  Thresholds: High={self.high_threshold}, Medium={self.medium_threshold}")

    def _setup_paths(self):
        """Set up all necessary paths based on base directory."""
        self.sample_mc_folder = os.path.join(self.base_directory, 'lda_results', 'MC_Sample')
        self.mc_feature_folder = os.path.join(self.base_directory, 'lda_results', 'MC_Feature')
        self.lda_results_folder = os.path.join(self.base_directory, 'lda_results')
        
        # Use comprehensive metrics file instead of MALLET XML files
        self.comprehensive_metrics_path = os.path.join(
            self.lda_results_folder, f'comprehensive_MC_metrics_{self.range_str}.csv'
        )

        print(f"  Sample MC folder: {self.sample_mc_folder}")
        print(f"  MC feature folder: {self.mc_feature_folder}")
        print(f"  Comprehensive metrics: {self.comprehensive_metrics_path}")

    @classmethod
    def from_lda_trainer(cls, lda_trainer, high_threshold: float = 0.67, 
                        medium_threshold: float = 0.33):
        """
        Create SankeyDataProcessor from an LDATrainer instance.

        Args:
            lda_trainer: LDATrainer instance that has completed training
            high_threshold: Threshold for high representation
            medium_threshold: Threshold for medium representation

        Returns:
            SankeyDataProcessor instance
        """
        if not hasattr(lda_trainer, 'all_metrics') or lda_trainer.all_metrics.empty:
            raise ValueError("LDATrainer must have completed training before creating SankeyDataProcessor")

        # Get MC_range from the trainer's results (now using 'K' column)
        MC_range = sorted(lda_trainer.all_metrics['K'].unique().astype(int).tolist())

        return cls(
            base_directory=lda_trainer.base_directory,
            MC_range=MC_range,
            high_threshold=high_threshold,
            medium_threshold=medium_threshold
        )

    def load_sample_mc_data(self):
        """Load all sample-MC probability files"""
        sample_mc_data = {}

        for k in self.MC_range:
            filename = f'MC_Sample_probabilities{k}.csv'
            filepath = os.path.join(self.sample_mc_folder, filename)

            if os.path.exists(filepath):
                df = pd.read_csv(filepath, index_col=0)
                sample_mc_data[k] = df
                print(f"Loaded K={k}: {df.shape[0]} topics (MCs), {df.shape[1]} samples")

                if df.shape[0] != k:
                    print(f"WARNING: K={k} has {df.shape[0]} topics, expected {k}")
            else:
                print(f"File not found: {filename}")

        return sample_mc_data

    def categorize_sample_assignments(self, sample_mc_data):
        """Categorize samples into high/medium representation levels"""
        categorized_data = {}

        for k, df in sample_mc_data.items():
            k_data = {
                'nodes': {},
                'sample_assignments': {}
            }

            # Process each topic
            for topic_idx in range(df.shape[0]):
                topic_name = f"K{k}_MC{topic_idx}"
                topic_probs = df.iloc[topic_idx, :]

                high_samples = []
                medium_samples = []
                total_prob = 0

                for sample_name, prob in topic_probs.items():
                    if prob >= self.high_threshold:
                        high_samples.append((sample_name, prob))
                        total_prob += prob
                    elif prob >= self.medium_threshold:
                        medium_samples.append((sample_name, prob))
                        total_prob += prob

                k_data['nodes'][topic_name] = {
                    'high_samples': high_samples,
                    'medium_samples': medium_samples,
                    'high_count': len(high_samples),
                    'medium_count': len(medium_samples),
                    'total_probability': total_prob
                }

            # Find primary topic assignments
            for sample_idx, sample_name in enumerate(df.columns):
                sample_column = df.iloc[:, sample_idx]
                max_prob = 0
                assigned_topic = None
                assignment_level = None

                for topic_idx, prob in enumerate(sample_column):
                    if prob >= self.medium_threshold and prob > max_prob:
                        max_prob = prob
                        assigned_topic = f"K{k}_MC{topic_idx}"
                        assignment_level = 'high' if prob >= self.high_threshold else 'medium'

                if assigned_topic:
                    k_data['sample_assignments'][sample_name] = {
                        'assigned_topic': assigned_topic,
                        'probability': max_prob,
                        'level': assignment_level
                    }

            categorized_data[k] = k_data
            print(f"K={k}: {len(k_data['sample_assignments'])} samples assigned to topics")

        return categorized_data

    def calculate_flows(self, categorized_data):
        """Calculate flows between consecutive K values"""
        flows = []
        k_values = sorted(categorized_data.keys())

        for i in range(len(k_values) - 1):
            source_k = k_values[i]
            target_k = k_values[i + 1]

            source_assignments = categorized_data[source_k]['sample_assignments']
            target_assignments = categorized_data[target_k]['sample_assignments']

            flow_counts = defaultdict(lambda: defaultdict(int))
            flow_samples = defaultdict(lambda: defaultdict(list))

            common_samples = set(source_assignments.keys()) & set(target_assignments.keys())
            print(f"K{source_k}→K{target_k}: {len(common_samples)} samples to track")

            for sample in common_samples:
                source_info = source_assignments[sample]
                target_info = target_assignments[sample]

                source_segment = f"{source_info['assigned_topic']}_{source_info['level']}"
                target_segment = f"{target_info['assigned_topic']}_{target_info['level']}"

                flow_counts[source_segment][target_segment] += 1
                flow_samples[source_segment][target_segment].append({
                    'sample': sample,
                    'source_prob': source_info['probability'],
                    'target_prob': target_info['probability']
                })

            for source_segment, targets in flow_counts.items():
                for target_segment, count in targets.items():
                    if count > 0:
                        avg_prob = np.mean([
                            (s['source_prob'] + s['target_prob']) / 2 
                            for s in flow_samples[source_segment][target_segment]
                        ])

                        flows.append({
                            'source_k': source_k,
                            'target_k': target_k,
                            'source_segment': source_segment,
                            'target_segment': target_segment,
                            'sample_count': count,
                            'average_probability': avg_prob,
                            'samples': flow_samples[source_segment][target_segment]
                        })

        print(f"Total flows calculated: {len(flows)}")
        return flows

    def load_metrics_data(self):
        """
        Load metrics data from comprehensive CSV file.
        
        This replaces the complex MALLET XML parsing with a simple CSV read.
        The comprehensive metrics file contains both perplexity and coherence
        for each individual topic.
        
        Returns:
            Dictionary mapping topic names to their metrics
        """
        try:
            if not os.path.exists(self.comprehensive_metrics_path):
                print(f"❌ Comprehensive metrics file not found: {self.comprehensive_metrics_path}")
                return {}

            df = pd.read_csv(self.comprehensive_metrics_path)
            
            # Validate required columns
            required_columns = ['Topic_Name', 'K', 'Perplexity', 'Coherence']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                print(f"❌ Missing columns in comprehensive metrics: {missing_columns}")
                return {}

            metrics_dict = {}
            
            for _, row in df.iterrows():
                topic_name = str(row['Topic_Name'])
                
                metrics_dict[topic_name] = {
                    'k_value': int(row['K']),
                    'perplexity': float(row['Perplexity']),
                    'coherence': float(row['Coherence']),
                    'topic_name': topic_name
                }

            print(f"✅ Loaded comprehensive metrics for {len(metrics_dict)} topics")
            print(f"   K values covered: {sorted(df['K'].unique())}")
            print(f"   Perplexity range: {df['Perplexity'].min():.2f} - {df['Perplexity'].max():.2f}")
            print(f"   Coherence range: {df['Coherence'].min():.4f} - {df['Coherence'].max():.4f}")
            
            return metrics_dict

        except Exception as e:
            print(f"❌ Error loading comprehensive metrics: {e}")
            return {}

    def save_processed_data(self, sankey_data, output_filename='sankey_data.json'):
        """Save processed data to JSON file"""
        if sankey_data is None:
            print("❌ No data to save")
            return

        def convert_numpy(obj):
            if isinstance(obj, np.integer):
                return int(obj)
            elif isinstance(obj, np.floating):
                return float(obj)
            elif isinstance(obj, np.ndarray):
                return obj.tolist()
            return obj

        def deep_convert(data):
            if isinstance(data, dict):
                return {k: deep_convert(v) for k, v in data.items()}
            elif isinstance(data, list):
                return [deep_convert(item) for item in data]
            else:
                return convert_numpy(data)

        converted_data = deep_convert(sankey_data)

        # Save to the LDA results directory
        output_path = os.path.join(self.lda_results_folder, output_filename)

        with open(output_path, 'w') as f:
            json.dump(converted_data, f, indent=2)

        print(f"💾 Data saved to {output_path}")
        return output_path

    def process_all_data(self, output_filename='fully_integrated_sankey_data_improved.json'):
        """
        🚀 SINGLE METHOD TO GET FULLY INTEGRATED DATA - IMPROVED VERSION

        This method does everything in one call using the simplified approach:
        1. Loads sample-MC data
        2. Categorizes assignments
        3. Calculates flows
        4. Integrates comprehensive metrics (perplexity + coherence per topic)
        5. Saves the result

        Returns the fully integrated data ready for use.
        """
        print("🚀 Starting complete data processing pipeline (IMPROVED)...")
        print("=" * 70)

        # Step 1: Load and process base data
        print("1️⃣ Loading sample-MC data...")
        sample_mc_data = self.load_sample_mc_data()

        if not sample_mc_data:
            print("❌ No sample data loaded. Stopping process.")
            return None

        print("\n2️⃣ Categorizing sample assignments...")
        categorized_data = self.categorize_sample_assignments(sample_mc_data)

        print("\n3️⃣ Calculating flows...")
        flows = self.calculate_flows(categorized_data)

        # Step 2: Build base sankey structure
        print("\n4️⃣ Building base sankey structure...")
        sankey_data = {
            'nodes': {},
            'flows': flows,
            'k_range': list(sample_mc_data.keys()),
            'thresholds': {
                'high': self.high_threshold,
                'medium': self.medium_threshold
            },
            'metadata': {
                'total_samples': sample_mc_data[list(sample_mc_data.keys())[0]].shape[1] if sample_mc_data else 0,
                'k_values_processed': list(sample_mc_data.keys()),
                'processing_timestamp': pd.Timestamp.now().isoformat(),
                'method': 'improved_comprehensive_metrics'
            }
        }

        # Collect all node data
        for k, k_data in categorized_data.items():
            for topic_name, node_data in k_data['nodes'].items():
                sankey_data['nodes'][topic_name] = node_data

        print(f"✅ Base structure created: {len(sankey_data['nodes'])} nodes, {len(flows)} flows")

        # Step 3: Integrate MALLET diagnostics (from comprehensive metrics)
        print("\n5️⃣ Integrating MALLET diagnostics...")
        metrics_dict = self.load_metrics_data()

        if metrics_dict:
            integrated_count = 0

            for topic_name, node_data in sankey_data['nodes'].items():
                if topic_name in metrics_dict:
                    metrics_data = metrics_dict[topic_name]
                    
                    # Add MALLET diagnostics in exact original format
                    node_data['mallet_diagnostics'] = {
                        'coherence': metrics_data['coherence'],
                        'tokens': 0.0,  # Not available in comprehensive metrics
                        'document_entropy': 0.0,
                        'word_length': 0.0,
                        'uniform_dist': 0.0,
                        'corpus_dist': 0.0,
                        'eff_num_words': 0.0,
                        'token_doc_diff': 0.0,
                        'rank_1_docs': 0.0,
                        'allocation_ratio': 0.0,
                        'allocation_count': 0.0,
                        'exclusivity': 0.0
                    }
                    
                    integrated_count += 1

            sankey_data['metadata']['mallet_integration'] = {
                'integrated_topics': integrated_count,
                'total_mallet_topics': len(metrics_dict)
            }
            
            print(f"✅ MALLET integration: {integrated_count} topics enriched")
        else:
            print("⚠️ No MALLET data available")

        # Step 4: Integrate perplexity data (from comprehensive metrics)
        print("\n6️⃣ Integrating perplexity data...")
        if metrics_dict:
            integrated_count = 0

            for topic_name, node_data in sankey_data['nodes'].items():
                if topic_name in metrics_dict:
                    metrics_data = metrics_dict[topic_name]
                    
                    # Add model_metrics in exact original format
                    if 'model_metrics' not in node_data:
                        node_data['model_metrics'] = {}
                    
                    node_data['model_metrics']['perplexity'] = metrics_data['perplexity']
                    node_data['model_metrics']['k_value'] = metrics_data['k_value']
                    integrated_count += 1

            sankey_data['metadata']['perplexity_integration'] = {
                'integrated_topics': integrated_count,
                'total_k_values_available': len(set(m['k_value'] for m in metrics_dict.values()))
            }
            
            print(f"✅ Perplexity integration: {integrated_count} topics enriched")
        else:
            print("⚠️ No perplexity data available")

        # Step 5: Save the final result
        print("\n7️⃣ Saving fully integrated data...")
        output_path = self.save_processed_data(sankey_data, output_filename)

        # Final summary
        print("\n" + "=" * 60)
        print("🎉 COMPLETE DATA PROCESSING FINISHED!")
        print(f"📊 Final Summary:")
        print(f"   - K values: {sankey_data['k_range']}")
        print(f"   - Total topics: {len(sankey_data['nodes'])}")
        print(f"   - Total flows: {len(sankey_data['flows'])}")
        print(f"   - Samples tracked: {sankey_data['metadata']['total_samples']}")

        if 'mallet_integration' in sankey_data['metadata']:
            print(f"   - MALLET enriched topics: {sankey_data['metadata']['mallet_integration']['integrated_topics']}")

        if 'perplexity_integration' in sankey_data['metadata']:
            print(f"   - Perplexity enriched topics: {sankey_data['metadata']['perplexity_integration']['integrated_topics']}")

        print(f"   - Data saved to: {output_path}")
        print("=" * 60)

        return sankey_data

    # Legacy methods for backward compatibility (if needed)
    def load_perplexity_data(self):
        """
        Legacy method for backward compatibility.
        Now simply extracts perplexity from comprehensive metrics.
        """
        metrics_dict = self.load_metrics_data()
        
        perplexity_dict = {}
        for topic_name, metrics in metrics_dict.items():
            k_value = metrics['k_value']
            if k_value not in perplexity_dict:
                perplexity_dict[k_value] = metrics['perplexity']
        
        return perplexity_dict

    def load_all_mallet_diagnostics(self):
        """
        Legacy method for backward compatibility.
        Now converts comprehensive metrics to DataFrame format similar to MALLET diagnostics.
        """
        metrics_dict = self.load_metrics_data()
        
        if not metrics_dict:
            return pd.DataFrame()
        
        # Convert to DataFrame format similar to MALLET diagnostics
        data_rows = []
        for topic_name, metrics in metrics_dict.items():
            data_rows.append({
                'global_topic_id': topic_name,
                'topic_id': int(topic_name.split('_MC')[1]),
                'k_value': metrics['k_value'],
                'coherence': metrics['coherence'],
                'tokens': 0.0,  # Not available in comprehensive metrics
                'document_entropy': 0.0,  # Not available in comprehensive metrics
                'word_length': 0.0,  # Not available in comprehensive metrics
                'uniform_dist': 0.0,  # Not available in comprehensive metrics
                'corpus_dist': 0.0,  # Not available in comprehensive metrics
                'eff_num_words': 0.0,  # Not available in comprehensive metrics
                'token_doc_diff': 0.0,  # Not available in comprehensive metrics
                'rank_1_docs': 0.0,  # Not available in comprehensive metrics
                'allocation_ratio': 0.0,  # Not available in comprehensive metrics
                'allocation_count': 0.0,  # Not available in comprehensive metrics
                'exclusivity': 0.0  # Not available in comprehensive metrics
            })
        
        df = pd.DataFrame(data_rows)
        print(f"✅ Converted comprehensive metrics to MALLET-like format: {len(df)} topics")
        
        return df

