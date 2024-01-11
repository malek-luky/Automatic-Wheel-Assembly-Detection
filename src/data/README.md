# dataset_2024-01-05

## Contents

### Files
- `FT_dataset_labels.csv`: Contains pairs ID, label
- `filter.py`: A script for processing the raw dataset. 
                Columms that need to be preserved can be specified in the script.
                ex. $ python3 filter.py <path_to_src_dir> <path_to_output_dir> 
- `normalize.py`: A script for data normalization.
                  Loads filtered CSV files and saves the normalized data in CSV files
                  in separate folder
                  ex. $ python3 normalize.py <path_to_src_dir> <path_to_labels> <path_to_output_dir> 
- `process.py`: A script that loads normalized data files and saves them as torch tensors
                in separate folder.
                ex. $ python3 process.py <path_to_src_dir> <path_to_output_dir>
- `make_dataset.py`: Runs previous scripts in pipeline: filter -> normalize -> process.
                      Data from each step of the pipeline are preserved in separate folders.
                      ex. $ python3 make_dataset.py <path_to_raw_data> <path_to_filtered_data> <path_to_normalized_data> <path_to_labels> <path_to_processed_data>

### Folders (notation)
- `/raw`: Contains raw data in .csv format.
- `/filtered`: Contains filtered data in .csv format, each file ends with '_id_<num>.csv
- `/normalized`: Contains normalized data in .csv format. 
                  File names in format 'data_id_<num>_label_<True/False>.csv
- `/processed`: Contains pytorch tensor files derived from normalized data files.
                File names in format 'data_id_<num>_label_<True/False>.csv



### Commands (inside git folder)
- it finds the git root, should be runable from any directory inside the git folder
- must have raw data inside data/raw folder
- universal: `python make_dataset.py` (works on windows!)

### Commands (without git folder)
- change the flag in make_dataset.py run_in_git to False (line 37)
- go to the root folder
- ubuntu: `python make_dataset.py ./data/raw ./data/filtered ./data/normalized ./data/FT_dataset_labels.csv ./data/processed`
- windows: will not work most likely, since the absolute path needs to be added :()
- you can tryo to run, but no guarantee `python make_dataset.py data/raw data/filtered data/normalized data/FT_dataset_labels.csv data/processed`