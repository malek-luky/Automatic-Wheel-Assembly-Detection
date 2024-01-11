#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import annotations
import os
import argparse

import pandas as pd

# Local imports
from utils import find_git_root_folder


"""filter.py: Filter raw .csv files and preserve specific columns."""

__author__  = "Vratislav Besta"
__group__   = "50"
__version__ = "1.0.1"
__email__   = "bestavra@fel.cvut.cz"
__date__    = "2024/01/10" 


# Define the columns to be preserved
columns_to_preserve = [
    "Time",
    "\"DeltaPicker\".TcpInWcs.z.Position",
    "\"HMI_User\".FT_Data.Force_X",
    "\"HMI_User\".FT_Data.Force_Y",
    "\"HMI_User\".FT_Data.Force_Z",
    "\"HMI_User\".FT_Data.Torque_X",
    "\"HMI_User\".FT_Data.Torque_Y",
    "\"HMI_User\".FT_Data.Torque_Z",
]


def process_csv_file(file_path: str, output_dir: str) -> None:
    """Process a single CSV file.

    Parameters
    ----------
    file_path : str
        The path to the CSV file to process.
    output_dir : str
        The path to the directory to store the processed CSV files.
    """
    df = pd.read_csv(file_path)
    # Filter out rows where #Identifier value is 0
    filtered_df = df[df['#Identifier'] != 0]

    # Create the output directory if it doesn't exist
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # Split the DataFrame into four separate DataFrames based on the #Identifier value
    for identifier in filtered_df['#Identifier'].unique():
        df_subset = filtered_df[filtered_df['#Identifier'] == identifier]
        df_subset = df_subset[columns_to_preserve]
        base_file_name = os.path.splitext(os.path.basename(file_path))[0]
        output_file_path = f'./{output_dir}/{base_file_name}_id_{identifier}.csv'
        df_subset.to_csv(output_file_path, index=False)


def process_all_csv_files(folder_path: str, output_dir: str, run_in_git: bool = True) -> None:
    """Process all CSV files in a folder.

    Parameters
    ----------
    folder_path : str
        The path to the folder containing the CSV files to process.
    output_dir : str
        The path to the directory to store the processed CSV files.
    run_in_git : bool, optional
        Whether to run the script in a git repository, by default True
    """
    # If running in git, set path relative to the root of the repository
    if run_in_git:
        git_root = find_git_root_folder()
        folder_path = os.path.join(git_root, folder_path)
        output_dir = os.path.join(git_root, output_dir)

    for file_name in os.listdir(folder_path):
        if file_name.endswith('.csv'):
            file_path = os.path.join(folder_path, file_name)
            process_csv_file(file_path, output_dir)


def main() -> None:
    """The main function."""
    # Create the parser
    parser = argparse.ArgumentParser(description="Filter raw CSV files.")

    # Add the arguments
    parser.add_argument('raw_data_path', type=str, help='The path to the raw data')
    parser.add_argument('processed_data_path', type=str, help='The path to store the processed data')

    # Parse the arguments
    args = parser.parse_args()

    # Process the CSV files
    process_all_csv_files(args.raw_data_path, args.processed_data_path)


if __name__ == '__main__':
    main()
