from omegaconf import OmegaConf
import pandas as pd
import torch

from src.models.model import TireAssemblyLSTM
from src.models.train_model import create_sequences

# evaluate the output shape of the model
def test_output_shape():
    input_size  = 8
    output_size = 1
    hidden_layer_size = 50
    
    # construct model
    model = TireAssemblyLSTM(input_size, hidden_layer_size, output_size)

    # Define a sample input with shape X
    sample_input = torch.randn((10, 10, 8))  

    # Pass the input through the model
    output = model(sample_input)

    # Define the expected output shape 
    expected_shape = (10, 1)  
    
    print("Got shape: ", output.shape)
    print("Expected shape: ", expected_shape)

    # Check if the output shape matches the expected shape
    assert output.shape == expected_shape, "Output shape does not match the expected shape"
   

# test if values in config file have not been corrupted
def test_config_loads():
    config = OmegaConf.load('src/models/config/default_config.yaml')
    hparams = config
    
    assert hparams.max_epochs > 0, "Hyper parameter 'max_epochs' from default_config.yaml is corrupted. Expected value \
        should be greater than 0."
    assert hparams.batch_size > 0, "Hyper parameter 'batch_size' from default_config.yaml is corrupted. Expected value \
        should be greater than 0."
    assert hparams.hidden_layer_size > 0, "Hyper parameter 'hidden_layer_size' from default_config.yaml is corrupted. Expected value \
        should be greater than 0."
    assert hparams.output_size == 1,"Hyper parameter 'output_size' from default_config.yaml is corrupted. Expected value \
        should be 1."
    assert hparams.sequence_length > 0, "Hyper parameter 'sequence_length' from default_config.yaml is corrupted. Expected value \
        should be greater than 0."
    

# test that dataset is being correctly partitioned during training
def test_sequences():
    num_features = 8
    seq_length = 10
    
    #load dataset
    df = pd.read_csv('data/processed/dataset_concatenated.csv')
    
    # Create sequences for each experiment
    X_seq, Y_seq = create_sequences(df, seq_length)
    
    assert X_seq.shape[1]==seq_length, "Sequence for data samples is of incorrect shape.\
        shape[1] of datasamples should be equal to sequence length."
    assert X_seq.shape[2]==num_features, "Sequence for data samples is of incorrect shape.\
        shape[2] of datasamples should be equal to number of features."
    assert Y_seq.shape[0]==X_seq.shape[0], "Sequence for data samples is of incorrect shape.\
        Number of datasamples should be equal to number of labels.]"
    
    
if __name__ == '__main__':
    test_output_shape()
    