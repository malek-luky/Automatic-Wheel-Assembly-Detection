from pytorch_lightning import Trainer
# from pytorch_lightning.loggers import WandbLogger

from datetime import datetime
import pandas as pd
from sklearn.model_selection import train_test_split
import numpy as np
import torch
from torch.utils.data import DataLoader, TensorDataset

from src.models.model import TireAssemblyLSTM

# HYPERPARAMETERS (consider using either hydra or load them from a config file or pass them as arguments)
MAX_EPOCHS = 2
BATCH_SIZE = 64
HIDDEN_LAYER_SIZE = 50  # Example size, adjust based on your needs
OUTPUT_SIZE = 1
SEQUENCE_LENGTH = 10  # number of time steps to consider for each sequence

# WANDB SETUP
# wandb_logger = WandbLogger(project="automatic-wheel-assembly-detection", entity="02476mlops")

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


# Create sequences for each experiment
X_seq, y_seq = create_sequences(df, SEQUENCE_LENGTH)

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
hidden_layer_size = HIDDEN_LAYER_SIZE
output_size = OUTPUT_SIZE

model = TireAssemblyLSTM(input_size, hidden_layer_size, output_size)

# Create DataLoaders
train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True)
test_loader = DataLoader(test_data, batch_size=BATCH_SIZE)


# Train the model
trainer = Trainer(
    #    logger=wandb_logger,
    max_epochs=MAX_EPOCHS,
)

trainer.fit(model, train_loader)

# Test the model
trainer.test(dataloaders=test_loader)

# Save the model
time = datetime.now().strftime("%Y%m%d-%H%M%S")
torch.save(model.state_dict(), f'models/model_{time}.pth')

print("DONE")

# save the model in gcloud storage bucket
