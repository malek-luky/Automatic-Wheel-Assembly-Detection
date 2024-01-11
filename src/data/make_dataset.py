#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations


"""make_dataset.py: Filter raw .csv files and preserve specific columns. 
                    Then process filtered .csv files, normalize data and save as torch tensor."""

__author__  = "Vratislav Besta"
__group__   = "50"
__version__ = "1.0.0"
__email__   = "bestavra@fel.cvut.cz"
__date__    = "2024/01/05" 


import os
import argparse
import shutil

import pandas as pd

from filter_data import process_all_csv_files
from data_normalization import process_files


def main() -> None:
    """The main function."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Process some CSV files.")

    # Add the arguments
    parser.add_argument('raw_data_path', type=str, help='The path to the raw data')
    parser.add_argument('labels_data_path', type=str, help='The path to the labels')
    parser.add_argument('processed_data_path', type=str, help='The path to store the processed data')

    # Parse the arguments
    args = parser.parse_args()

    # Process the CSV files
    tmp_dir = './tmp'
    process_all_csv_files(args.raw_data_path, tmp_dir)

    labels_df = pd.read_csv(f"{args.labels_data_path}/FT_dataset_labels.csv") # Load labels file

    # Create the output directory if it doesn't exist
    if not os.path.exists(args.processed_data_path):
        os.makedirs(args.processed_data_path)
    
    # Normalize data and save as torch tensor
    process_files(tmp_dir, labels_df, args.processed_data_path)

    # Remove the temporary directory
    shutil.rmtree(tmp_dir)


if __name__ == '__main__':
    main()




