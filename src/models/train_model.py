import click
from omegaconf import DictConfig, OmegaConf
from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger

from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset
import wandb

from src.models.model import TireAssemblyLSTM
#from model import TireAssemblyLSTM

# WANDB SETUP
WANDB_DEFINED = False
WANDB_PROJECT = None
WANDB_ENTITY  = None
SWEEP_DEFINED = False

# Load dataset (call script from root directory -> python src/models/train_model.py)
df = pd.read_csv('data/processed/dataset_concatenated.csv')


def create_sequences(df, seq_length):
        sequences = []
        labels = []

        for _, group in df.groupby('#Identifier'):
            data = group.values
            for i in range(len(data) - seq_length):
                seq = data[i:(i + seq_length), :-2]  # All columns except Identifier and Label
                label = data[i + seq_length - 1, -1]  # Label of the last time step in the sequence
                sequences.append(seq)
                labels.append(label)

        return np.array(sequences), np.array(labels)
  

def sweep_hparams(iterations: int) -> None:
    global SWEEP_DEFINED
    SWEEP_DEFINED = True
    
    # current version only performs parameter sweeping on the number of hidden layers
    # by minimizing the training loss
    # choice of parameters can be added to the click interface
    sweep_config = {
        "method": "random",
        "name": "sweep",
        "metric": {"goal": "minimize", "name": "train_loss"},
        "parameters": {
            "max_epochs": {"values": [20]},
            "batch_size": {"values": [64]},
            "hidden_layer_size": {"max": 60, "min": 50},
            "output_size": {"values": [1]},
            "sequence_length": {"values": [10]},
        },
        "early_terminate":  {
            "type": "hyperband",
            "min_iter": 1,
            "max_iter": 10
        },
    }
    sweep_id = wandb.sweep(
        sweep=sweep_config, project=WANDB_PROJECT, entity=WANDB_ENTITY
    )
    wandb.agent(sweep_id, train_routine, count=iterations)
    

def train_routine(config=None) -> None:
    # if hyperparameters sweeping is defined, the config is loaded from wandb
    if SWEEP_DEFINED:
        wandb.init(config=config)
        hparams = wandb.config
        wandb_logger = WandbLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY)
    # otherwise the paranters are loaded from a local directory
    else:
        config = OmegaConf.load('src/models/config/default_config.yaml')
        hparams = config
        # in the only-training mode the connection to Wandb can be disabled
        if WANDB_DEFINED:
            wandb_logger = WandbLogger(project=WANDB_PROJECT, entity=WANDB_ENTITY)
        else:
            wandb_logger = None
        
    # Create sequences for each experiment
    X_seq, y_seq = create_sequences(df, hparams.sequence_length)

    # Split the data into training and test sets
    X_train, X_test, y_train, y_test = train_test_split(X_seq, y_seq, test_size=0.2, random_state=42)

    # Convert to PyTorch tensors
    X_train_tensor = torch.from_numpy(X_train).float()
    y_train_tensor = torch.from_numpy(y_train).float()
    X_test_tensor = torch.from_numpy(X_test).float()
    y_test_tensor = torch.from_numpy(y_test).float()

    # Create TensorDatasets
    train_data = TensorDataset(X_train_tensor, y_train_tensor)
    test_data = TensorDataset(X_test_tensor, y_test_tensor)

    # Define the model
    input_size = X_train.shape[2]
    hidden_layer_size = hparams.hidden_layer_size
    output_size = hparams.output_size

    model = TireAssemblyLSTM(input_size, hidden_layer_size, output_size)

    # Create DataLoaders
    train_loader = DataLoader(train_data, batch_size=hparams.batch_size, shuffle=True)
    test_loader = DataLoader(test_data, batch_size=hparams.batch_size)


    # Train the model
    trainer = Trainer(
        logger=wandb_logger,
        max_epochs=hparams.max_epochs,
    )

    trainer.fit(model, train_loader)

    # Test the model
    trainer.test(dataloaders=test_loader)

    # Save the model
    time = datetime.now().strftime("%Y%m%d-%H%M%S")
    torch.save(model.state_dict(), f'models/model_{time}.pth')

    # save the model in gcloud storage bucket
    # create a run and log a model artifact to it
    with wandb.init(project="automatic-wheel-assembly-detection") as run:
        model_artifact = wandb.Artifact(
        name="mlops_model", 
        type="model"
        )
        model_artifact.add_file(f'models/model_{time}.pth')
        run.log_artifact(model_artifact) # saves the model to wandb artifact registry"
        run.link_artifact(model_artifact, "model-registry/basic-LTSM") # links to model as the best model
        # TODO: implement the logic of comparing last vs new model and choose the better one and link that one

@click.command()
@click.option('--train', is_flag=True, default=True, help='Use to only train the model.')
@click.option('--sweep', is_flag=True, default=False, help='Use to sweep hyperparameters.')
@click.option('--sweep_iter', default=5, help='Number of iterations for hyperparameters sweeping.')
@click.option('--wandb_on', is_flag=True, default=True, help='Use to connect to Wandb service. Automatically set to True if --sweep is defined. Otherwise False')
@click.option('--wandb_project', default="automatic-wheel-assembly-detection", 
              help='Your wandb project name. Default is "automatic-wheel-assembly-detection"')
@click.option('--wandb_entity', default="02476mlops", 
              help='Your wandb entity name. Default is "02476mlops"')

def parse_input(train: bool, sweep:bool, sweep_iter:int, wandb_on: bool, wandb_project: str, wandb_entity: str) -> None:
    
    global WANDB_PROJECT, WANDB_ENTITY, WANDB_DEFINED
    WANDB_DEFINED = wandb_on
    WANDB_PROJECT = wandb_project
    WANDB_ENTITY = wandb_entity
    
    if sweep:
        print("Sweeping hyperparameters with", sweep_iter, "iteration(s).")
        sweep_hparams(sweep_iter)
    else:
        train_routine(config=None)


if __name__ == '__main__':
    parse_input()
    