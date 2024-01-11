#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import glob
import argparse

import pandas as pd

# Local imports
from utils import find_git_root_folder


"""normalize.py: Process filtered .csv files, normalize data and save as torch tensor."""

__author__  = "Vratislav Besta"
__group__   = "50"
__version__ = "1.0.1"
__email__   = "bestavra@fel.cvut.cz"
__date__    = "2024/01/10" 


def load_and_normalize_data(file_path: str) -> pd.DataFrame:
    """Load and normalize data from a CSV file.
    
    Parameters
    ----------
    file_path : str
        The path to the CSV file to load.
    Returns
    -------
    pd.DataFrame
        The normalized data.
    """
    df = pd.read_csv(file_path)
    normalized_df = (df - df.mean()) / df.std()
    # tensor = torch.tensor(normalized_df.values, dtype=torch.float32)
    # return tensor
    return normalized_df

def process_files(
        src_folder: str, 
        labels_path: str, 
        out_folder: str, 
        run_in_git: bool = True) -> list[str]:
    """Normalize all CSV files in a folder.

    Parameters
    ----------
    src_folder : str
        The path to the folder containing the CSV files.
    labels_path : str
        The path to the labels data.
    out_folder : str
        The path to the folder to store the normalized CSV files.
    run_in_git : bool, optional
        Set to True if the script is run in the git repository, by default True
    Returns
    -------
    list[str]
        The paths to the normalized CSV files.
    """
    # If running in git, set path relative to the root of the repository
    if run_in_git:
        git_root = find_git_root_folder()
        src_folder = os.path.join(git_root, src_folder)
        out_folder = os.path.join(git_root, out_folder)
        labels_path = os.path.join(git_root, labels_path)
        labels_df = pd.read_csv(labels_path)
    
    # Create the output directory if it doesn't exist
    if not os.path.exists(out_folder):
        os.makedirs(out_folder)

    processed_files = []

    for _, row in labels_df.iterrows():
        file_id = row['ID']
        label = row['LABEL']

        # Find the file with the matching ID
        file_pattern = os.path.join(src_folder, f"*_id_{file_id}.csv")
        matching_files = glob.glob(file_pattern)

        if matching_files:
            
            # Assuming the first matching file is the correct one
            file_name = matching_files[0]

            # data_tensor = load_and_normalize_data(file_name)
            normalized_df = load_and_normalize_data(file_name)


            # Save tensor as .pt file
            # tensor_file_name = f"{out_folder}/data_id_{file_id}_label_{label}.pt"
            # torch.save(data_tensor, tensor_file_name)
            normalized_df_file_name = f"{out_folder}/data_id_{file_id}_label_{label}.csv"
            normalized_df.to_csv(normalized_df_file_name, index=False)

            # processed_files.append(tensor_file_name)
            processed_files.append(normalized_df_file_name)

    return processed_files


def main() -> None:
    """The main function."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Normalize filtered CSV files.")

    # Add the arguments
    parser.add_argument('filtered_data_path', type=str, nargs='?', default='data/filtered', help='The path to the filtered data')
    parser.add_argument('labels_data_path', type=str, nargs='?', default='data/FT_dataset_labels.csv', help='The path to the labels data')
    parser.add_argument('normalized_data_path', type=str, nargs='?', default='data/normalized', help='The path to store the normalized data')

    # Parse the arguments
    args = parser.parse_args()

    # Process the CSV files
    _ = process_files(args.filtered_data_path, args.labels_data_path, args.normalized_data_path)

if __name__ == '__main__':
    main()
