"""
Taxonomic data preprocessing for LDA analysis.

This module contains the TaxonomyProcessor class for handling:
- ASV table and taxonomy data processing
- Genus name updating based on taxonomic hierarchy
- Unique ID assignment
- MALLET input document preparation
"""

import os
import pandas as pd
import random
import string


class TaxonomyProcessor:
    """
    A class for processing taxonomic data and preparing it for LDA analysis.

    This class handles:
    - Reading and processing ASV tables and taxonomy data
    - Updating genus names based on taxonomic hierarchy
    - Assigning unique IDs to taxa
    - Creating directory structure for analysis
    - Preparing Mallet input documents
    """

    def __init__(self, asvtable_path, taxonomy_path, base_directory):
        """
        Initialize the TaxonomyProcessor.

        Args:
            asvtable_path (str): Path to the ASV table CSV file
            taxonomy_path (str): Path to the taxonomy CSV file
            base_directory (str): Base directory for storing results
        """
        self.asvtable_path = asvtable_path
        self.taxonomy_path = taxonomy_path
        self.base_directory = base_directory

        # Initialize counters and tracking variables
        self.unknown_count = [0]
        self.taxa_counts = {}
        self.all_generated_ids = set()

        # Initialize directory paths
        self._setup_directories()

        # Initialize data containers
        self.asvtable = None
        self.taxa_split = None
        self.sampletable_randomID = None  # For MALLET (uses randomID)
        self.sampletable_genusID = None   # For Gensim (uses genus_ID)

    def _setup_directories(self):
        """Create necessary directories for analysis."""
        self.intermediate_directory = os.path.join(self.base_directory, 'intermediate')
        self.loop_directory = os.path.join(self.base_directory, 'lda_loop')
        self.lda_directory = os.path.join(self.base_directory, 'lda_results')
        self.MC_sample_directory = os.path.join(self.lda_directory, 'MC_Sample')
        self.MC_feature_directory = os.path.join(self.lda_directory, 'MC_Feature')

        # Create directories
        for directory in [self.intermediate_directory, self.loop_directory, 
                         self.lda_directory, self.MC_sample_directory, 
                         self.MC_feature_directory]:
            os.makedirs(directory, exist_ok=True)

        # Set up path variables
        self.loop_output_directory_path = self.loop_directory
        self.Loop_2tables_directory_path = self.lda_directory
        self.Loop_MC_sample_directory_path = self.MC_sample_directory
        self.Loop_MC_feature_directory_directory_path = self.MC_feature_directory
        self.path_to_training_data = os.path.join(self.loop_output_directory_path, 'training.txt')
        self.path_to_formatted_training_data = os.path.join(self.loop_output_directory_path, 'mallet.training')

    def update_genus_new(self, row):
        """
        Update genus names based on taxonomic hierarchy.

        Args:
            row: DataFrame row containing taxonomic information

        Returns:
            str: Updated genus name
        """
        # Case 1: If Genus is 'g__uncultured' or NaN or 'd__Bacteria'
        if row['Genus'] == 'g__uncultured' or pd.isna(row['Genus']) or row['Genus'] == 'd__Bacteria':
            # Try Family first
            if pd.notna(row['Family']) and row['Family'] != 'f__uncultured':
                return f"{row['Family']}"
            # Try Order
            elif pd.notna(row['Order']) and row['Order'] != 'o__uncultured':
                return f"{row['Order']}"
            # Try Class
            elif pd.notna(row['Class']) and row['Class'] != 'c__uncultured':
                return f"{row['Class']}"
            # Try Phylum
            elif pd.notna(row['Phylum']) and row['Phylum'] != 'p__uncultured':
                return f"{row['Phylum']}"
            # If all taxonomic levels are uncultured or NaN, use unknown_count
            else:
                self.unknown_count[0] += 1
                return f"unknown_{self.unknown_count[0]}"
        # Case 2: Return original Genus if it's valid
        return row['Genus']

    def assign_id(self, genus_based):
        """
        Assign sequential IDs to genus names.

        Args:
            genus_based (str): Genus name to assign ID to

        Returns:
            str: Genus name with count suffix
        """
        # Initialize count if this is the first time seeing this genus
        if genus_based not in self.taxa_counts:
            self.taxa_counts[genus_based] = 0
        else:
            # Increment count for subsequent occurrences
            self.taxa_counts[genus_based] += 1
        # Return genus name with count suffix
        return f"{genus_based}_{self.taxa_counts[genus_based]}"

    def generate_single_id(self, min_length=5):
        """
        Generate a unique random ID.

        Args:
            min_length (int): Minimum length of the ID

        Returns:
            str: Unique random ID
        """
        # Generate a random string with the minimal length
        new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))
        # If the ID already exists, regenerate until we get a unique ID
        while new_id in self.all_generated_ids:
            new_id = ''.join(random.choices(string.ascii_lowercase, k=min_length))
        self.all_generated_ids.add(new_id)
        return new_id

    def load_and_process_data(self):
        """Load and process ASV table and taxonomy data."""
        # Load ASV table
        sampletable = pd.read_csv(self.asvtable_path, index_col=0)
        self.asvtable = sampletable.T

        # Load and process taxonomy
        self.taxa_split = pd.read_csv(self.taxonomy_path, index_col=0)
        self.taxa_split['Genus_based'] = self.taxa_split.apply(lambda row: self.update_genus_new(row), axis=1)
        self.taxa_split['genus_ID'] = self.taxa_split['Genus_based'].apply(lambda x: self.assign_id(x))
        self.taxa_split['randomID'] = self.taxa_split.apply(
            lambda row: self.generate_single_id(min_length=5), axis=1
        )

        # Save intermediate taxonomy file
        self.taxa_split.to_csv(os.path.join(self.intermediate_directory, 'intermediate_taxa.csv'), index=True)

        # Create sample table with genus IDs for Gensim (readable names)
        # This makes Gensim results interpretable and comparable
        genus_mapping_dict = self.taxa_split['genus_ID'].to_dict()
        self.sampletable_genusID = sampletable.copy()
        new_columns_genus = {}
        for col in sampletable.columns:
            if col in genus_mapping_dict:
                new_columns_genus[col] = genus_mapping_dict[col]
            else:
                new_columns_genus[col] = col
        self.sampletable_genusID = self.sampletable_genusID.rename(columns=new_columns_genus)
        self.sampletable_genusID.to_csv(os.path.join(self.intermediate_directory, 'annotated_genusid.csv'), index=True)
        
        # Create sample table with random IDs for MALLET (backward compatibility)
        random_mapping_dict = self.taxa_split['randomID'].to_dict()
        self.sampletable_randomID = sampletable.copy()
        new_columns_random = {}
        for col in sampletable.columns:
            if col in random_mapping_dict:
                new_columns_random[col] = random_mapping_dict[col]
            else:
                new_columns_random[col] = col
        self.sampletable_randomID = self.sampletable_randomID.rename(columns=new_columns_random)
        self.sampletable_randomID.to_csv(os.path.join(self.intermediate_directory, 'annotaed_randomid.csv'), index=True)

    def create_mallet_input(self):
        """Create MALLET input documents from processed data (uses randomID)."""
        if self.sampletable_randomID is None:
            raise ValueError("Data must be loaded and processed first. Call load_and_process_data().")

        doc_list = []
        # Each sample becomes a document where ASVs are repeated based on their abundance
        for index, row in self.sampletable_randomID.iterrows():
            doc = []
            for asvs_id1, abundance in row.items():
                if abundance > 0:
                    doc.extend([str(asvs_id1)] * int(abundance))
            doc_list.append(doc)

        flattened_nested_list = [' '.join(sublist) for sublist in doc_list]

        with open(self.path_to_training_data, 'w') as f:
            for document in flattened_nested_list:
                f.write(document + '\n')

    def process_all(self):
        """
        Run the complete processing pipeline.
        Creates MALLET text file (randomID) and both CSV matrices.
        Gensim loads directly from annotated_genusid.csv matrix.

        Returns:
            dict: Dictionary containing processed data and file paths
        """
        self.load_and_process_data()
        self.create_mallet_input()  # For MALLET (randomID) - creates training.txt
        # Note: Gensim loads directly from annotated_genusid.csv, no text file needed

        return {
            'asvtable': self.asvtable,
            'taxa_split': self.taxa_split,
            'sampletable_randomID': self.sampletable_randomID,  # For MALLET
            'sampletable_genusID': self.sampletable_genusID,    # For Gensim
            'paths': {
                'intermediate_directory': self.intermediate_directory,
                'loop_directory': self.loop_directory,
                'lda_directory': self.lda_directory,
                'MC_sample_directory': self.MC_sample_directory,
                'MC_feature_directory': self.MC_feature_directory,
                'path_to_training_data': self.path_to_training_data,  # MALLET training.txt (randomID)
                'path_to_formatted_training_data': self.path_to_formatted_training_data
            }
        }
