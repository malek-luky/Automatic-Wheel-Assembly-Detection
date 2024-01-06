# dataset_2024-01-05

## Contents

### Files

-   `FT_dataset_labels.csv`: Contains pairs ID, label
-   `src/data/filter_data.py`: A script for processing the raw dataset.
    Columms that need to be preserved can be specified in the script.
    ex. $ python3 filter_data.py <path_to_src_dir> <path_to_output_dir>

### Folders

-   `/raw`: Contains raw data in .csv format.
-   `/processed`: Contains .csv files, where each file contains data specific to one identifier

## Data Format

Describe the format of the data, including details about any coding or encoding schemes. For example, explain CSV file structure, what each column represents, etc.

TODO:
-   [ ] Explain how to use DVC to pull the data
-   [ ] Explain why 0 is filtered out
-   [ ] Explain the data and the robot
