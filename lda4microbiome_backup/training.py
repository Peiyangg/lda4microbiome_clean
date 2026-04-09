"""
LDA model training using both Gensim and MALLET.

This module contains the LDATrainer class for:
- Setting up directory structure for LDA analysis
- Training LDA models (Gensim by default, MALLET for MCMC) for different numbers of topics
- Processing and saving model outputs
- Calculating model metrics (perplexity, coherence)
"""

from typing import List, Dict, Tuple, Any, Optional
import os
import pandas as pd
import numpy as np
import subprocess
import pickle
import json
import math
import warnings
from collections import defaultdict
from gensim import corpora
from gensim.models import LdaModel, CoherenceModel
from gensim.corpora import Dictionary
import little_mallet_wrapper as lmw

# Import the clean metrics implementation
from .metrics import (
    compute_gensim_perplexity,
    compute_gensim_coherence,
    compute_gensim_coherence_per_topic,
    mallet_perplexity_from_log,
    parse_mallet_diagnostics_coherence,
)


class LDATrainer:
    """
    A class for training LDA models across multiple topic numbers and managing results.

    This class handles:
    - Setting up directory structure for LDA analysis
    - Training MALLET LDA models for different numbers of topics
    - Processing and saving model outputs
    - Calculating model metrics (perplexity, coherence)
    - Aggregating results across all models
    """

    def __init__(self, base_directory: str, path_to_mallet: Optional[str] = None, 
                 implementation: str = 'gensim', **gensim_params):
        """
        Initialize the LDA Trainer.

        Args:
            base_directory (str): Base directory for storing all results
            path_to_mallet (str, optional): Path to MALLET executable (required for MALLET implementation)
            implementation (str): Implementation to use ('gensim' or 'mallet'). Default is 'gensim'.
            **gensim_params: Additional parameters for Gensim LDA model
                           Default: passes=20, iterations=400, alpha='auto', eta='auto', 
                           random_state=42, chunksize=2000, eval_every=None
        """
        self.base_directory = base_directory
        self.path_to_mallet = path_to_mallet
        self.implementation = implementation.lower()
        
        # Validate implementation choice
        if self.implementation not in ['gensim', 'mallet']:
            raise ValueError("Implementation must be either 'gensim' or 'mallet'")
        
        # Validate MALLET path if using MALLET
        if self.implementation == 'mallet' and not path_to_mallet:
            raise ValueError("path_to_mallet is required when using MALLET implementation")
        
        # Set default Gensim parameters
        self.gensim_params = {
            'passes': 20,
            'iterations': 400,
            'alpha': 'auto',
            'eta': 'auto',
            'random_state': 42
        }
        # Update with user-provided parameters
        self.gensim_params.update(gensim_params)

        # Set up directory structure
        self.paths = self._setup_directories()

        # Initialize result storage
        self.all_df_probabilities_rel = pd.DataFrame()
        self.all_metrics = pd.DataFrame(columns=['K', 'Perplexity', 'Coherence'])
        self.all_coherence_scores = pd.DataFrame()  # Store all individual coherence scores
        self.models_data = {}  # Store all model data for comprehensive metrics

        # Data containers (to be set later)
        self.flattened_nested_list = None
        self.sampletable_genusid = None
        self.gensim_corpus = None
        self.gensim_dictionary = None
        self.processed_texts = None
        self.custom_data_loaded = False  # Track if custom data was loaded
        
        
        print(f"LDATrainer initialized with {self.implementation.upper()} implementation")
        if self.implementation == 'gensim':
            print(f"  Gensim parameters: {self.gensim_params}")

    def _setup_directories(self) -> Dict[str, str]:
        """Set up directory structure and return path dictionary."""
        intermediate_directory = os.path.join(self.base_directory, 'intermediate')
        loop_directory = os.path.join(self.base_directory, 'lda_loop')
        lda_directory = os.path.join(self.base_directory, 'lda_results')
        MC_sample_directory = os.path.join(lda_directory, 'MC_Sample')
        MC_feature_directory = os.path.join(lda_directory, 'MC_Feature')
        diagnostics_directory = os.path.join(lda_directory, 'Diagnostics')

        # Create all directories
        directories = [
            intermediate_directory, loop_directory, lda_directory,
            MC_sample_directory, MC_feature_directory, diagnostics_directory
        ]

        print(f"Setting up LDA directory structure in: {self.base_directory}")
        for directory in directories:
            os.makedirs(directory, exist_ok=True)
            print(f"  ✓ Created/verified: {directory}")

        return {
            'intermediate_directory': intermediate_directory,
            'loop_directory': loop_directory,
            'lda_directory': lda_directory,
            'MC_sample_directory': MC_sample_directory,
            'MC_feature_directory': MC_feature_directory,
            'diagnostics_directory': diagnostics_directory,
            'path_to_training_data': os.path.join(loop_directory, 'training.txt'),
            'path_to_formatted_training_data': os.path.join(loop_directory, 'mallet.training')
        }

    def set_custom_gensim_data(self, dictionary: Dictionary, corpus: list, 
                               processed_texts: list, sample_index: Optional[pd.Index] = None):
        """
        Set custom Gensim dictionary and corpus for training.
        
        This allows you to bypass the standard data loading pipeline and provide
        your own preprocessed dictionary and corpus.
        
        Args:
            dictionary (gensim.corpora.Dictionary): Gensim Dictionary object mapping tokens to IDs
            corpus (list): List of bag-of-words documents [(token_id, count), ...]
            processed_texts (list): List of tokenized documents (list of lists of tokens)
                                   Required for coherence calculation
            sample_index (pd.Index, optional): Index for samples (for output DataFrames)
                                              If None, will use integer indices
        
        Example:
            >>> from gensim.corpora import Dictionary
            >>> texts = [["genus1", "genus2"], ["genus1", "genus3"]]
            >>> dictionary = Dictionary(texts)
            >>> corpus = [dictionary.doc2bow(text) for text in texts]
            >>> trainer.set_custom_gensim_data(dictionary, corpus, texts)
            >>> trainer.train_models(MC_range=[2, 3, 4])
        
        Note:
            - Only works with Gensim implementation (not MALLET)
            - You must call this BEFORE train_models()
            - This skips load_training_data() - don't call both
        """
        if self.implementation != 'gensim':
            raise ValueError("Custom data can only be set when using Gensim implementation")
        
        # Validate inputs
        if not isinstance(dictionary, Dictionary):
            raise TypeError("dictionary must be a gensim.corpora.Dictionary object")
        
        if not isinstance(corpus, list):
            raise TypeError("corpus must be a list of bag-of-words documents")
        
        if not isinstance(processed_texts, list):
            raise TypeError("processed_texts must be a list of tokenized documents")
        
        if len(corpus) != len(processed_texts):
            raise ValueError(f"corpus ({len(corpus)}) and processed_texts ({len(processed_texts)}) must have the same length")
        
        # Set the data
        self.gensim_dictionary = dictionary
        self.gensim_corpus = corpus
        self.processed_texts = processed_texts
        
        # Create a sample table placeholder for compatibility with output generation
        if sample_index is not None:
            if len(sample_index) != len(corpus):
                raise ValueError(f"sample_index length ({len(sample_index)}) must match corpus length ({len(corpus)})")
            self.sampletable_genusid = pd.DataFrame(index=sample_index)
        else:
            # Use integer indices if no sample names provided
            self.sampletable_genusid = pd.DataFrame(index=range(len(corpus)))
        
        self.custom_data_loaded = True
        
        print("Custom Gensim data loaded successfully:")
        print(f"  ✓ Dictionary: {len(self.gensim_dictionary)} unique terms")
        print(f"  ✓ Corpus: {len(self.gensim_corpus)} documents")
        print(f"  ✓ Processed texts: {len(self.processed_texts)} documents")
        print(f"  ✓ Sample index: {len(self.sampletable_genusid)} samples")
        print("  ⚠ You can now call train_models() directly (skip load_training_data())")

    def load_training_data(self):
        """
        Load training data from intermediate files.

        Input: Reads from intermediate directory:
               - For Gensim: annotated_genusid.csv (matrix with genus IDs)
               - For MALLET: training.txt (text file with randomIDs) and annotaed_randomid.csv
        
        Output: Sets instance attributes:
                - self.flattened_nested_list: List of document strings (MALLET only)
                - self.sampletable_genusid: DataFrame with sample abundances
                - self.gensim_corpus: Bag-of-words corpus (if using Gensim)
                - self.gensim_dictionary: Gensim Dictionary (if using Gensim)
                - self.processed_texts: List of tokenized documents (if using Gensim)
        
        Note: If you've already called set_custom_gensim_data(), this method will be skipped.
        """
        if self.custom_data_loaded:
            print("Custom data already loaded. Skipping load_training_data().")
            return
        if self.implementation == 'gensim':
            # Gensim: Load matrix directly with genus IDs (readable names)
            sampletable_path = os.path.join(self.paths['intermediate_directory'], 'annotated_genusid.csv')
            if not os.path.exists(sampletable_path):
                raise FileNotFoundError(f"Sample table not found: {sampletable_path}. Run TaxonomyProcessor first.")
            
            self.sampletable_genusid = pd.read_csv(sampletable_path, index_col=0)
            print(f"Loading Gensim data directly from matrix (genus IDs)...")
            print(f"  ✓ Loaded sample table: {self.sampletable_genusid.shape}")
            
            # Prepare Gensim data directly from matrix
            self._prepare_gensim_data_from_matrix()
            
        else:
            # MALLET: Load text file with random IDs (MALLET requires text input)
            sampletable_path = os.path.join(self.paths['intermediate_directory'], 'annotaed_randomid.csv')
            if not os.path.exists(sampletable_path):
                raise FileNotFoundError(f"Sample table not found: {sampletable_path}. Run TaxonomyProcessor first.")

            self.sampletable_genusid = pd.read_csv(sampletable_path, index_col=0)
            print(f"Loading MALLET data from text file (random IDs)...")
            print(f"  ✓ Loaded sample table: {self.sampletable_genusid.shape}")

            # Load the flattened nested list from training data
            training_data_path = self.paths['path_to_training_data']
            if not os.path.exists(training_data_path):
                raise FileNotFoundError(f"Training data not found: {training_data_path}. Run TaxonomyProcessor first.")

            with open(training_data_path, 'r') as f:
                self.flattened_nested_list = [line.strip() for line in f]

            print(f"  ✓ Loaded training documents: {len(self.flattened_nested_list)} documents")
        
        print("Training data loaded successfully.")

    def _prepare_gensim_data(self):
        """
        Prepare data for Gensim training from text file (legacy method).
        
        Input: self.flattened_nested_list - List of document strings (space-separated tokens)
        
        Output: Sets instance attributes:
                - self.processed_texts: List of lists of tokens
                - self.gensim_dictionary: Gensim Dictionary mapping tokens to IDs
                - self.gensim_corpus: List of bag-of-words documents [(token_id, count), ...]
        """
        print("Preparing data for Gensim from text file...")
        
        # Convert documents to list of word lists
        self.processed_texts = []
        for doc in self.flattened_nested_list:
            words = [word.strip() for word in doc.split() if word.strip()]
            self.processed_texts.append(words)
        
        # Create Gensim dictionary and corpus
        self.gensim_dictionary = Dictionary(self.processed_texts)
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in self.processed_texts]
        
        print(f"  ✓ Created dictionary with {len(self.gensim_dictionary)} unique terms")
        print(f"  ✓ Created corpus with {len(self.gensim_corpus)} documents")
    
    def _prepare_gensim_data_from_matrix(self):
        """
        Prepare data for Gensim training directly from abundance matrix.
        More efficient than text file - no intermediate file needed.
        
        Input: self.sampletable_genusid - DataFrame with samples as rows, genus_IDs as columns
        
        Output: Sets instance attributes:
                - self.processed_texts: List of lists of tokens (for coherence calculation)
                - self.gensim_dictionary: Gensim Dictionary mapping genus_IDs to IDs
                - self.gensim_corpus: List of bag-of-words documents [(token_id, count), ...]
        """
        print("Preparing data for Gensim directly from matrix...")
        
        # Create processed_texts by repeating genus names by their counts
        self.processed_texts = []
        for idx, row in self.sampletable_genusid.iterrows():
            doc_tokens = []
            for genus_id, count in row.items():
                if count > 0:
                    # Repeat genus_id by its count
                    doc_tokens.extend([str(genus_id)] * int(count))
            self.processed_texts.append(doc_tokens)
        
        # Create Gensim dictionary from processed texts (not just column names!)
        # This ensures the dictionary sees the words in their document context
        self.gensim_dictionary = Dictionary(self.processed_texts)
        
        # Create corpus using the dictionary
        self.gensim_corpus = [self.gensim_dictionary.doc2bow(text) for text in self.processed_texts]
        
        print(f"  ✓ Created dictionary with {len(self.gensim_dictionary)} unique terms")
        print(f"  ✓ Created corpus with {len(self.gensim_corpus)} documents")
        print(f"  ✓ Using genus IDs (readable names) for interpretable results")

    def _generate_file_paths(self, num_topics: int) -> Dict[str, str]:
        """Generate all file paths for a specific number of topics."""
        loop_path = self.paths['loop_directory']
        diagnostics_path = self.paths['diagnostics_directory']
        mc_sample_path = self.paths['MC_sample_directory']
        mc_feature_path = self.paths['MC_feature_directory']

        if self.implementation == 'gensim':
            return {
                'model': os.path.join(loop_path, f'gensim_model_{num_topics}.pkl'),
                'dictionary': os.path.join(loop_path, f'gensim_dictionary_{num_topics}.pkl'),
                'coherence': os.path.join(diagnostics_path, f'coherence_{num_topics}.json'),
                'sample_probs': os.path.join(mc_sample_path, f'MC_Sample_probabilities{num_topics}.csv'),
                'feature_probs': os.path.join(mc_feature_path, f'MC_Feature_Probabilities_{num_topics}.csv')
            }
        else:
            return {
                'model': os.path.join(loop_path, f'mallet.model.{num_topics}'),
                'topic_keys': os.path.join(loop_path, f'mallet.topic_keys.{num_topics}'),
                'topic_distributions': os.path.join(loop_path, f'mallet.topic_distributions.{num_topics}'),
                'word_weights': os.path.join(loop_path, f'mallet.word_weights.{num_topics}'),
                'diagnostics': os.path.join(diagnostics_path, f'mallet.diagnostics.{num_topics}.xml'),
                'sample_probs': os.path.join(mc_sample_path, f'MC_Sample_probabilities{num_topics}.csv'),
                'feature_probs': os.path.join(mc_feature_path, f'MC_Feature_Probabilities_{num_topics}.csv')
            }

    def _create_topic_index(self, num_topics: int) -> List[str]:
        """Create index names for topics."""
        return [f"K{num_topics}_MC{i}" for i in range(num_topics)]

    def _train_single_model(self, num_topics: int, file_paths: Dict[str, str]) -> None:
        """Train a single LDA model using the specified implementation."""
        print(f"DEBUG: Using implementation: {self.implementation}")
        if self.implementation == 'gensim':
            self._train_gensim_model(num_topics, file_paths)
        else:
            self._train_mallet_model(num_topics, file_paths)

    def _train_gensim_model(self, num_topics: int, file_paths: Dict[str, str]) -> None:
        """Train a single Gensim LDA model."""
        print(f"Training Gensim LDA model with {num_topics} topics...")
        
        # Train the model
        lda_model = LdaModel(
            corpus=self.gensim_corpus,
            id2word=self.gensim_dictionary,
            num_topics=num_topics,
            **self.gensim_params
        )
        
        # Save model and dictionary
        with open(file_paths['model'], 'wb') as f:
            pickle.dump(lda_model, f)
        with open(file_paths['dictionary'], 'wb') as f:
            pickle.dump(self.gensim_dictionary, f)
        
        print(f"Completed Gensim LDA training for {num_topics} topics.")

    def _train_mallet_model(self, num_topics: int, file_paths: Dict[str, str]) -> None:
        """Train a single MALLET LDA model."""
        lmw.import_data(
            self.path_to_mallet,
            self.paths['path_to_training_data'],
            self.paths['path_to_formatted_training_data'],
            self.flattened_nested_list
        )

        # Construct MALLET command
        mallet_command = [
            self.path_to_mallet,
            'train-topics',
            '--input', self.paths['path_to_formatted_training_data'],
            '--num-topics', str(num_topics),
            '--output-state', file_paths['model'],
            '--output-topic-keys', file_paths['topic_keys'],
            '--output-doc-topics', file_paths['topic_distributions'],
            '--word-topic-counts-file', file_paths['word_weights'],
            '--diagnostics-file', file_paths['diagnostics'],
            '--optimize-interval', '10',
            '--num-iterations', '1000',
            '--random-seed', '43'
        ]

        # Run MALLET and capture output for perplexity calculation
        log_path = os.path.join(self.paths['loop_directory'], f'mallet_train_{num_topics}.log')
        print(f"Running MALLET for {num_topics} microbial components...")
        print(f"  Capturing training output to: {log_path}")
        
        with open(log_path, 'w') as log_file:
            result = subprocess.run(
                mallet_command, 
                stdout=log_file, 
                stderr=subprocess.STDOUT,  # Redirect stderr to stdout so everything goes to log
                check=True
            )
        
        print(f"Completed MALLET for {num_topics} microbial components.")
        print(f"  Training log saved to: {log_path}")

    def _process_model_output(self, num_topics: int, file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, List, List]:
        """Process model output and save individual results."""
        if self.implementation == 'gensim':
            return self._process_gensim_output(num_topics, file_paths)
        else:
            return self._process_mallet_output(num_topics, file_paths)

    def _process_gensim_output(self, num_topics: int, file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, List, List]:
        """Process Gensim model output and save individual results."""
        # Load the trained model
        with open(file_paths['model'], 'rb') as f:
            lda_model = pickle.load(f)
        
        # Get topic distributions for documents
        topic_distributions = []
        for doc_bow in self.gensim_corpus:
            doc_topics = dict(lda_model.get_document_topics(doc_bow, minimum_probability=0))
            # Ensure all topics are represented (fill missing with 0)
            topic_probs = [doc_topics.get(topic_id, 0.0) for topic_id in range(num_topics)]
            topic_distributions.append(topic_probs)
        
        # Get word-topic distributions using get_topics() for full precision
        # This returns a (num_topics x vocab_size) matrix with full floating-point probabilities
        topic_word_matrix = lda_model.get_topics()  # Shape: (num_topics, vocab_size)
        
        # Create topic index
        topic_index = self._create_topic_index(num_topics)
        
        # Get vocabulary in the correct order
        vocab = [self.gensim_dictionary[i] for i in range(len(self.gensim_dictionary))]
        
        # Create feature probabilities DataFrame directly from the matrix
        # This preserves full precision and includes all words
        df_asv_probabilities = pd.DataFrame(
            topic_word_matrix,
            index=topic_index,
            columns=vocab
        )
        
        # For backward compatibility, also create word_topics list
        # Convert matrix back to list of tuples for any downstream code that expects it
        word_topics = []
        for topic_id in range(num_topics):
            for word_id, prob in enumerate(topic_word_matrix[topic_id]):
                if prob > 0:  # Only include non-zero probabilities
                    word = self.gensim_dictionary[word_id]
                    # Store as (topic_id, word, probability) - keeping full precision
                    word_topics.append((topic_id, word, prob))
        
        # Create topic distribution DataFrame
        df_topic_dist = pd.DataFrame(
            topic_distributions,
            index=self.sampletable_genusid.index,
            columns=topic_index
        )
        df_topic_dist_wide = df_topic_dist.T
        
        # Calculate coherence score using clean metrics.py implementation
        coherence_score = compute_gensim_coherence(
            lda_model,
            texts=self.processed_texts,
            dictionary=self.gensim_dictionary,
            coherence='c_v',
            topn=20
        )
        
        # Save coherence score
        coherence_data = {
            'num_topics': num_topics,
            'coherence_score': coherence_score,
            'method': 'gensim_c_v',
            'formula': 'gensim CoherenceModel with c_v measure'
        }
        with open(file_paths['coherence'], 'w') as f:
            json.dump(coherence_data, f, indent=2)
        
        # Save individual results
        df_topic_dist_wide.to_csv(file_paths['sample_probs'], index=True)
        df_asv_probabilities.to_csv(file_paths['feature_probs'], index=True)
        print(f"Saved individual model results for {num_topics} topics.")
        
        return df_asv_probabilities, topic_distributions, word_topics

    def _process_mallet_output(self, num_topics: int, file_paths: Dict[str, str]) -> Tuple[pd.DataFrame, List, List]:
        """Process MALLET model output and save individual results."""
        # Load model output
        topic_distributions, word_topics = self._load_mallet_model_output(
            file_paths['topic_distributions'], 
            file_paths['word_weights']
        )

        # Create topic index
        topic_index = self._create_topic_index(num_topics)

        # Process ASV data
        df_asv = pd.DataFrame(word_topics, columns=['MC', 'Term', 'Frequency'])
        df_asv_pivot = df_asv.pivot_table(index='MC', columns='Term', values='Frequency', fill_value=0)
        df_asv_probabilities = df_asv_pivot.div(df_asv_pivot.sum(axis=1), axis=0)
        df_asv_probabilities.index = topic_index

        # Create topic distribution DataFrame
        df_topic_dist = pd.DataFrame(
            topic_distributions,
            index=self.sampletable_genusid.index,
            columns=topic_index
        )
        df_topic_dist_wide = df_topic_dist.T

        # Save individual results
        df_topic_dist_wide.to_csv(file_paths['sample_probs'], index=True)
        df_asv_probabilities.to_csv(file_paths['feature_probs'], index=True)
        print(f"Saved individual model results for {num_topics} topics.")

        return df_asv_probabilities, topic_distributions, word_topics

    def _load_mallet_model_output(self, topic_distributions_path: str, word_weights_path: str) -> Tuple[List, List]:
        """
        Load MALLET model output files.

        Args:
            topic_distributions_path: Path to topic distributions file
            word_weights_path: Path to word weights file

        Returns:
            Tuple of (topic_distributions, word_topics)
        """
        # Load topic distributions

        topic_distributions = lmw.load_topic_distributions(topic_distributions_path)

        # Load word weights
        word_topics = []
        with open(word_weights_path, 'r') as f:
            for line in f:
                parts = line.split()
                try:
                    if len(parts) < 2:
                        raise ValueError("Line does not have enough parts")

                    word = parts[1]
                    topic_freq_pairs = parts[2:]

                    for pair in topic_freq_pairs:
                        topic_id, frequency = pair.split(':')
                        word_topics.append((int(topic_id), word, int(frequency)))

                except ValueError as e:
                    # Log or print the problematic line for debugging
                    print(f"Skipping line due to format issues: {line} - Error: {e}")

        return topic_distributions, word_topics

    def _calculate_perplexity(self, topic_distributions: List, epsilon: float = 1e-10) -> float:
        """
        Calculate perplexity for topic distributions.

        Args:
            topic_distributions: List of topic probability distributions
            epsilon: Small value to avoid log(0)

        Returns:
            Average perplexity across all samples
        """
        perplexities = []

        for distribution in topic_distributions:
            # Ensure the distribution doesn't have zero values by clipping
            distribution = np.clip(distribution, epsilon, 1.0)
            # Calculate the entropy for this distribution
            entropy = -np.sum(np.log(distribution) * distribution)
            # Calculate perplexity and store it
            perplexities.append(np.exp(entropy))

        # Return the average perplexity over all samples
        return np.mean(perplexities)

    def _calculate_coherence(self, word_topics: List, texts: List, top_n: int = 10) -> float:
        """
        Calculate coherence score for topics.

        Args:
            word_topics: List of (topic_id, word, frequency) tuples
            texts: List of document texts
            top_n: Number of top words to use for coherence calculation

        Returns:
            Coherence score
        """
        try:
            # Ensure texts are in the correct format (list of lists of words)
            processed_texts = []
            for text in texts:
                if isinstance(text, str):
                    # Split by whitespace and filter out empty strings
                    words = [word.strip() for word in text.split() if word.strip()]
                    processed_texts.append(words)
                elif isinstance(text, list):
                    # Already a list, but ensure all elements are strings
                    words = [str(word).strip() for word in text if str(word).strip()]
                    processed_texts.append(words)
                else:
                    print(f"Warning: Unexpected input type: {type(text)}. Skipping.")
                    continue

            if not processed_texts:
                raise ValueError("No valid texts found after processing")

            # Create dictionary from processed texts
            id2word = Dictionary(processed_texts)

            # Group word_topics by topic number
            topics_dict = defaultdict(list)
            for topic_num, word, freq in word_topics:
                # Ensure word is a string and exists in dictionary
                word_str = str(word).strip()
                if word_str in id2word.token2id:
                    topics_dict[topic_num].append((word_str, float(freq)))

            if not topics_dict:
                raise ValueError("No valid topics found in word_topics")

            # Extract top N words for each topic
            topics = []
            for topic_num, word_freqs in topics_dict.items():
                # Sort words by frequency (descending) and take top N
                top_words = [word for word, freq in sorted(word_freqs, key=lambda x: x[1], reverse=True)[:top_n]]
                if top_words:  # Only add non-empty topics
                    topics.append(top_words)

            if not topics:
                raise ValueError("No valid topics extracted")

            # Calculate coherence using 'c_v' measure with texts (not corpus)
            coherence_model = CoherenceModel(
                topics=topics, 
                texts=processed_texts,  # Use texts, not corpus
                dictionary=id2word, 
                coherence='c_v'
            )

            coherence_score = coherence_model.get_coherence()

            print(f"Coherence calculated successfully: {coherence_score:.4f}")
            return coherence_score

        except Exception as e:
            print(f"Error calculating coherence: {str(e)}")
            # Return a default value or re-raise depending on your needs
            return 0.0

    def _calculate_mallet_style_coherence(self, model, num_topics, top_n=10, beta=0.01):
        """Calculate coherence score using MALLET's formula."""
        
        # Get top words for each topic
        topics = []
        for topic_id in range(num_topics):
            top_words = [word for word, prob in model.show_topic(topic_id, topn=top_n)]
            topics.append(top_words)
        
        # Pre-calculate document frequencies for all words in the dictionary
        doc_freq = {self.gensim_dictionary[term_id]: freq for term_id, freq in self.gensim_dictionary.dfs.items()}
        
        total_coherence = 0
        for topic_words in topics:
            topic_coherence = 0
            for i in range(1, len(topic_words)):
                for j in range(i):
                    w_i = topic_words[i]
                    w_j = topic_words[j]
                    
                    # D(w) = document frequency of word w
                    d_wi = doc_freq.get(w_i, 0)
                    
                    if d_wi == 0:
                        continue
                    
                    # D(w_i, w_j) = co-document frequency of w_i and w_j
                    co_doc_freq = 0
                    for doc_bow in self.gensim_corpus:
                        doc_words = {self.gensim_dictionary[term_id] for term_id, _ in doc_bow}
                        if w_i in doc_words and w_j in doc_words:
                            co_doc_freq += 1
                    
                    # Score for this pair
                    score = np.log((co_doc_freq + beta) / d_wi)
                    topic_coherence += score
            
            total_coherence += topic_coherence
            
        return total_coherence

    def _get_training_perplexity_from_log(self, log_path: str) -> float:
        """
        Parse a MALLET training log and return the perplexity based on the last
        reported "LL/token: <value>" line.

        This function scans the file for occurrences of the pattern
        "LL/token: <number>" and uses the final value, assuming the log corresponds
        to a single training run where later iterations reflect more converged values.

        Parameters
        ----------
        log_path : str
            Path to a text file containing MALLET's stdout/stderr from train-topics.

        Returns
        -------
        float
            The perplexity computed as exp(-LL/token) using the last observed LL/token.

        Raises
        ------
        ValueError
            If no LL/token line can be found in the provided log file.
        """
        import re
        pattern = re.compile(r"LL/token:\s*(-?\d+(?:\.\d+)?)")
        last_ll = None
        with open(log_path, "r") as f:
            for line in f:
                m = pattern.search(line)
                if m:
                    try:
                        last_ll = float(m.group(1))
                    except ValueError:
                        # Skip unparsable values
                        continue
        if last_ll is None:
            raise ValueError(f"No 'LL/token' line found in log file: {log_path}")
        # Calculate perplexity = exp(-LL/token)
        return float(math.exp(-last_ll))

    def _create_comprehensive_metrics_dataframe(self, 
                                             models_data: Dict[int, Dict[str, Any]],
                                             range_str: str = "") -> pd.DataFrame:
        """
        Create a comprehensive metrics DataFrame with individual topic entries.
        
        Args:
            models_data: Dictionary mapping num_topics to model data
            range_str: String identifier for the range (for filename)
            
        Returns:
            DataFrame with columns: Topic_Name, K, Perplexity, Coherence
        """
        metrics_rows = []
        
        for num_topics, data in models_data.items():
            perplexity = data.get('perplexity', 0.0)
            topic_coherences = data.get('topic_coherences', {})
            
            # Create entries for each topic
            for topic_idx in range(num_topics):
                topic_name = f"K{num_topics}_MC{topic_idx}"
                
                # Coherence is topic-specific
                coherence = topic_coherences.get(f"MC{topic_idx}", 0.0)
                
                metrics_rows.append({
                    'Topic_Name': topic_name,
                    'K': num_topics,
                    'Perplexity': perplexity,  # Same for all topics in a model
                    'Coherence': coherence     # Different for each topic
                })
        
        metrics_df = pd.DataFrame(metrics_rows)
        
        # Sort by number of topics, then by topic name
        metrics_df = metrics_df.sort_values(['K', 'Topic_Name']).reset_index(drop=True)
        
        return metrics_df

    def train_models(self, MC_range: List[int] = None, range_str: str = None) -> Dict[str, Any]:
        """
        Train LDA models for a range of topic numbers using MALLET-style metrics.

        Args:
            MC_range: List of numbers of topics to train. If None, defaults to range(2, 21)
            range_str: String representation of range for file naming

        Returns:
            Dictionary containing all results and metrics
        """
        # Set default range if not provided
        if MC_range is None:
            MC_range = list(range(2, 21))
            print("No range provided. Using default range: 2-20 topics")
        
        # Ensure MC_range is a list
        if not isinstance(MC_range, list):
            MC_range = list(MC_range)
        
        # Load training data automatically (unless custom data was set)
        if not self.custom_data_loaded:
            print("Loading training data...")
            self.load_training_data()
        else:
            print("Using custom data (skipping automatic data loading)...")

        if range_str is None:
            range_str = f"{min(MC_range)}-{max(MC_range)}"

        print("="*60)
        print(f"Starting LDA training for {len(MC_range)} different topic numbers...")
        print(f"Topic range: {MC_range}")
        print(f"Using MALLET-style metrics for consistent evaluation")
        print("="*60)

        for num_topics in MC_range:
            print(f"\n--- Processing {num_topics} topics ---")

            # Generate file paths
            file_paths = self._generate_file_paths(num_topics)

            # Train model
            self._train_single_model(num_topics, file_paths)

            # Process output
            df_asv_probabilities, topic_distributions, word_topics = self._process_model_output(num_topics, file_paths)

            # Calculate metrics using implementation-specific methods from metrics.py
            if self.implementation == 'gensim':
                # Load the trained model for metrics
                with open(file_paths['model'], 'rb') as f:
                    lda_model = pickle.load(f)
                
                print(f"Calculating Gensim perplexity for {num_topics} topics...")
                perplexity = compute_gensim_perplexity(lda_model, self.gensim_corpus)
                
                print(f"Calculating Gensim coherence (c_v) for {num_topics} topics...")
                avg_coherence = compute_gensim_coherence(
                    lda_model,
                    texts=self.processed_texts,
                    dictionary=self.gensim_dictionary,
                    coherence='c_v',
                    topn=20
                )
                
                # Get per-topic coherence scores (each topic gets its own value)
                per_topic_scores = compute_gensim_coherence_per_topic(
                    lda_model,
                    texts=self.processed_texts,
                    dictionary=self.gensim_dictionary,
                    coherence='c_v',
                    topn=20
                )
                topic_coherences = {f"MC{i}": per_topic_scores[i] for i in range(num_topics)}
                
            else:  # MALLET implementation
                print(f"Calculating MALLET coherence for {num_topics} topics...")
                try:
                    avg_coherence, per_topic_coherences = parse_mallet_diagnostics_coherence(
                        file_paths['diagnostics']
                    )
                    # Create topic_coherences dict
                    topic_coherences = {f"MC{i}": coh for i, coh in enumerate(per_topic_coherences)}
                except Exception as e:
                    warnings.warn(f"Failed to parse MALLET diagnostics: {e}")
                    avg_coherence = 0.0
                    topic_coherences = {f"MC{i}": 0.0 for i in range(num_topics)}
                
                print(f"Calculating MALLET perplexity for {num_topics} topics...")
                # Try to find a log file for perplexity
                log_candidates = [
                    os.path.join(self.paths['loop_directory'], f'mallet_train_{num_topics}.log'),
                    os.path.join(self.paths['loop_directory'], f'train_topics_{num_topics}.log'),
                    os.path.join(self.paths['loop_directory'], f'mallet_{num_topics}.log')
                ]
                
                perplexity = math.nan
                for log_path in log_candidates:
                    if os.path.exists(log_path):
                        try:
                            perplexity = mallet_perplexity_from_log(log_path)
                            print(f"  Found perplexity from log: {log_path}")
                            break
                        except Exception as e:
                            continue
                
                if math.isnan(perplexity):
                    warnings.warn(f"No MALLET log found for perplexity calculation. Using fallback method.")
                    # Fallback to simple perplexity calculation
                    perplexity = self._calculate_perplexity(topic_distributions)

            # Store model data for comprehensive metrics
            self.models_data[num_topics] = {
                'perplexity': perplexity,
                'topic_coherences': topic_coherences,
                'avg_coherence': avg_coherence,
                'topic_distributions': topic_distributions,
                'word_topics': word_topics
            }

            # Store results (maintain backward compatibility)
            self.all_df_probabilities_rel = pd.concat([self.all_df_probabilities_rel, df_asv_probabilities])

            new_row = pd.DataFrame([{
                'K': num_topics,
                'Perplexity': perplexity,
                'Coherence': avg_coherence
            }])
            self.all_metrics = pd.concat([self.all_metrics, new_row], ignore_index=True)

            print(f"✓ Perplexity: {perplexity:.2f}")
            print(f"✓ Average coherence: {avg_coherence:.4f}")
            print(f"✓ Individual topic coherences: {len(topic_coherences)} topics")
            print(f"Processed and appended results for {num_topics} MCs.")

        # Save enhanced final results
        self._save_enhanced_results(range_str)

        return {
            'probabilities': self.all_df_probabilities_rel,
            'metrics': self.all_metrics,
            'models_data': self.models_data,
            'paths': self.paths
        }

    def _save_enhanced_results(self, range_str: str):
        """Save enhanced final results with comprehensive metrics DataFrame."""
        # Save traditional results (backward compatibility)
        prob_path = os.path.join(self.paths['loop_directory'], f'all_MC_probabilities_rel_{range_str}.csv')
        metrics_path = os.path.join(self.paths['lda_directory'], f'all_MC_metrics_{range_str}.csv')

        self.all_df_probabilities_rel.to_csv(prob_path)
        self.all_metrics.to_csv(metrics_path)

        # Create and save comprehensive metrics DataFrame
        comprehensive_metrics = self._create_comprehensive_metrics_dataframe(
            self.models_data, range_str
        )
        
        # Save comprehensive metrics with improved naming
        comprehensive_path = os.path.join(self.paths['lda_directory'], f'comprehensive_MC_metrics_{range_str}.csv')
        comprehensive_metrics.to_csv(comprehensive_path, index=False)
        
        # Save individual model data for future analysis
        models_data_path = os.path.join(self.paths['lda_directory'], f'models_data_{range_str}.json')
        # Convert numpy arrays to lists for JSON serialization
        serializable_data = {}
        for num_topics, data in self.models_data.items():
            serializable_data[str(num_topics)] = {
                'perplexity': float(data['perplexity']),
                'avg_coherence': float(data['avg_coherence']),
                'topic_coherences': {k: float(v) for k, v in data['topic_coherences'].items()},
                'num_topics': num_topics,
                'num_documents': len(data['topic_distributions'])
            }
        
        with open(models_data_path, 'w') as f:
            json.dump(serializable_data, f, indent=2)

        print("="*60)
        print("Training complete! Enhanced results saved:")
        print(f"  • Traditional probabilities: {prob_path}")
        print(f"  • Traditional metrics: {metrics_path}")
        print(f"  • Comprehensive metrics: {comprehensive_path}")
        print(f"  • Models summary data: {models_data_path}")
        print(f"")
        print(f"Comprehensive metrics summary:")
        print(f"  - Total topics evaluated: {len(comprehensive_metrics)}")
        print(f"  - Models range: {comprehensive_metrics['K'].min()}-{comprehensive_metrics['K'].max()}")
        print(f"  - Average perplexity: {comprehensive_metrics['Perplexity'].mean():.2f}")
        print(f"  - Average coherence: {comprehensive_metrics['Coherence'].mean():.4f}")
        print(f"  - Perplexity range: {comprehensive_metrics['Perplexity'].min():.2f} - {comprehensive_metrics['Perplexity'].max():.2f}")
        print(f"  - Coherence range: {comprehensive_metrics['Coherence'].min():.4f} - {comprehensive_metrics['Coherence'].max():.4f}")
        print("="*60)

    def _save_final_results(self, range_str: str):
        """Save final combined results (legacy method for backward compatibility)."""
        prob_path = os.path.join(self.paths['loop_directory'], f'all_MC_probabilities_rel_{range_str}.csv')
        metrics_path = os.path.join(self.paths['lda_directory'], f'all_MC_metrics_{range_str}.csv')

        self.all_df_probabilities_rel.to_csv(prob_path)
        self.all_metrics.to_csv(metrics_path)

        print("="*60)
        print("Training complete! Final results saved:")
        print(f"  • Probabilities: {prob_path}")
        print(f"  • Metrics: {metrics_path}")
        print("="*60)






















