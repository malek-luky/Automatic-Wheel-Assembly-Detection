import torch
import os
import git
from torch.utils.data import DataLoader, TensorDataset

def find_git_root_folder():
    """
    Finds the root folder of the git repository to automatically set the paths to the data folders.
    """
    try:
        repo = git.Repo(search_parent_directories=True)
        git_root = repo.git.rev_parse("--show-toplevel")
        return git_root
    except git.InvalidGitRepositoryError:
        return None

# Load data
root_foler = find_git_root_folder()
data_path = root_foler + '/data/processed/data_id_1_label_True.pt'
data = torch.load(data_path)
print(data.size()[1])

# Create PyTorch dataset
dataset = TensorDataset(*data)

# Create PyTorch dataloader
dataloader = DataLoader(dataset, batch_size=1)

# Print size of first item
print(next(iter(dataloader))[0].size())