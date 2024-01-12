#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import argparse

# Local imports
import filter
import normalize
import process

"""make_dataset.py: Filter raw .csv files and preserve specific columns. 
                    Then normalize data and save as torch tensor."""

__author__ = "Vratislav Besta"
__group__ = "50"
__version__ = "1.0.1"
__email__ = "bestavra@fel.cvut.cz"
__date__ = "2024/01/10"


def main() -> None:
    """The main function."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some CSV files.")

    # Add the arguments
    parser.add_argument('raw_data_path', type=str, nargs='?', default='data/raw', help='The path to the raw data')
    parser.add_argument('filtered_data_path', type=str, nargs='?', default='data/filtered',
                        help='The path to store the filtered data')
    parser.add_argument(
        'normalized_data_path', type=str, nargs='?', default='data/normalized',
        help='The path to store the normalized data')
    parser.add_argument(
        'labels_data_path', type=str, nargs='?', default='data/FT_dataset_labels.csv',
        help='The path to the labels data')
    parser.add_argument(
        'processed_data_path', type=str, nargs='?', default='data/processed',
        help='The path to store the processed data')

    # Parse the arguments
    args = parser.parse_args()

    # Set to True if the script is run in the git repository
    run_in_git = True

    # Filter the raw CSV files
    filter.process_all_csv_files(
        args.raw_data_path,
        args.filtered_data_path,
        run_in_git=run_in_git
    )

    # Normalize the filtered CSV files
    normalize.process_files(
        args.filtered_data_path,
        args.labels_data_path,
        args.normalized_data_path,
        run_in_git=run_in_git
    )

    # Save normalized CSV files as torch tensors
    concatenated_csv_file_name = 'dataset_concatenated.csv'
    process.save_CSVs_as_tensors_and_concatenate(
        args.normalized_data_path,
        args.processed_data_path,
        concatenated_csv_file_name,
        run_in_git=run_in_git
    )


if __name__ == '__main__':
    main()
