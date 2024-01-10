import os
import torch
import pytest
from tests import _PATH_DATA

def test_folder_path_exists():
    assert os.path.exists(_PATH_DATA), "The _PATH_DATA folder does not exist, please run dvc pull"

def test_folder_contents():

    # Check if there are more than 90 items in the folder
    folder_path = _PATH_DATA+"/processed"
    assert len(os.listdir(folder_path)) > 90, "There are not enough items in the folder, did dvc pull run correctly?"

    # Check if there are more than 90 items in the folder
    folder_path = _PATH_DATA+"/raw"
    assert len(os.listdir(folder_path)) > 10, "The original meassurements are missing, maybe try to run dvc pull?"

    # Load one of the items
    item_path = os.path.join(_PATH_DATA, "processed/data_id_1_label_True.pt")
    item = torch.load(item_path)

    # Check if it is a torch file
    assert isinstance(item, torch.Tensor), "The item is not a torch file"

    # Check its dimensions
    assert item.size()[1] == 7, "The item does not have the expected dimensions, should include seven columns, but it does not match"