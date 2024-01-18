import os
import torch
import pytest

_TEST_ROOT = os.path.dirname(__file__)  # root of test folder
_PROJECT_ROOT = os.path.dirname(_TEST_ROOT)  # root of project
_PATH_DATA = os.path.join(_PROJECT_ROOT, "data")  # root of data

# Get the absolute path to the data directory
data_dir = os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'data'))


def test_folder_path_exists():
    assert os.path.exists(data_dir), "The data folder does not exist, please run dvc pull"


def test_folder_contents():

    # Check if there are more than 90 items in the folder
    folder_path = os.path.join(data_dir, "processed")
    assert len(os.listdir(folder_path)) > 90, "There are not enough items in the folder, did dvc pull run correctly?"

    # Check if there are more than 90 items in the folder
    folder_path = os.path.join(data_dir, "raw")
    assert len(os.listdir(folder_path)) > 10, "The original measurements are missing, maybe try to run dvc pull?"

    # Load one of the items
    item_path = os.path.join(data_dir, "processed/data_id_1_label_True.pt")
    item = torch.load(item_path)

    # Check if it is a torch file
    assert isinstance(item, torch.Tensor), "The item is not a torch file"

    # Check its dimensions
    assert item.size()[
        1] == 10, "The item does not have the expected dimensions, should include ten columns, but it does not match"
