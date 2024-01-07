from pytorch_lightning import Trainer
from pytorch_lightning.loggers import WandbLogger
import lightning as L

import torch
from torch.utils.data import TensorDataset, DataLoader # these are needed for the training data
from src.models.model import LightningLSTM
from src.data.make_dataset import sensorDataset

## params to be removed to config later
num_epochs = 100
logging_step = 2
wandb_project_name = "02476mlops"

# params to be removed to config later ##

model = LightningLSTM() # First, make model from the class

## print out the name and value for each parameter
print("Before optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)
    
    
# retrieve data from dataset
## create the training data for the neural network.
# inputs = torch.tensor([[0., 0.5, 0.25, 1.], [1., 0.5, 0.25, 1.]])
# labels = torch.tensor([0., 1.])

inputs = ...
labels = ...

dataset = TensorDataset(inputs, labels) 
dataloader = DataLoader(dataset)


# set the trainer
trainer = L.Trainer(max_epochs=num_epochs, log_every_n_steps=logging_step, logger=WandbLogger(project=wandb_project_name))

trainer.fit(model, train_dataloaders=dataloader)

print("After optimization, the parameters are...")
for name, param in model.named_parameters():
    print(name, param.data)