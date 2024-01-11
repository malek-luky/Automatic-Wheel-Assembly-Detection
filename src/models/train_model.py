"""This is inital version of the training code for the project."""
import os
from pathlib import Path
import pickle
import torch
import warnings
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor
from lightning.pytorch.loggers import TensorBoardLogger
from lightning.pytorch.tuner import Tuner
import numpy as np
import pandas as pd
from pytorch_forecasting import GroupNormalizer, TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.data.examples import get_stallion_data
from pytorch_forecasting.metrics import MAE, RMSE, SMAPE, PoissonLoss, QuantileLoss
from pytorch_forecasting.models.temporal_fusion_transformer.tuning import optimize_hyperparameters
from pytorch_forecasting.utils import profile

# Suppress warnings
import warnings
from pandas.errors import SettingWithCopyWarning

warnings.simplefilter(action="ignore", category=SettingWithCopyWarning)

# Load data
file_path = os.path.abspath("src/full_dataset.csv")
data = pd.read_csv(file_path)
data["time_idx"] = data["Time"]

# Define training parameters
training_cutoff = data["time_idx"].max() - 0 # we may not want to cut off the training data
max_encoder_length = 1500 # how long the max encoder sequence should be
max_prediction_length = 1 # should be 1 since we are predictiong bool value
data['LABEL'] = data['LABEL'].astype(str) # convert to srting

# Create TimeSeriesDataSet
training = TimeSeriesDataSet(
    data[lambda x: x.time_idx <= training_cutoff],
    time_idx="time_idx",
    target="LABEL", # True or False
    group_ids=["#Identifier"], # column name of the identifier sequence id need to exclude 0
    min_encoder_length=10,
    max_encoder_length=max_encoder_length,
    min_prediction_length=1,
    max_prediction_length=max_prediction_length,
    static_categoricals=[],
    static_reals=[],
    time_varying_known_categoricals=[],
    variable_groups={},  # group of categorical variables can be treated as one variable
    time_varying_known_reals=["time_idx", 
                            "TcpInWcs_z_Position",
                            "FT_Data_Force_X",
                            "FT_Data_Force_Y",
                            "FT_Data_Force_Z",
                            "FT_Data_Torque_X",
                            "FT_Data_Torque_Y",
                            "FT_Data_Torque_Z",],
    time_varying_unknown_categoricals=["LABEL"],
    time_varying_unknown_reals=[],
    add_relative_time_idx=True,
    add_target_scales=True,
    add_encoder_length=True,
)

# Create validation dataset
validation = TimeSeriesDataSet.from_dataset(training, data, predict=True, stop_randomization=True)
batch_size = 128
train_dataloader = training.to_dataloader(train=True, batch_size=batch_size, num_workers=31)
val_dataloader = validation.to_dataloader(train=False, batch_size=batch_size, num_workers=31)


# save datasets
save_data = True
if (save_data):
    training.save("training.pkl")
    validation.save("validation.pkl")

# Define callbacks and logger
early_stop_callback = EarlyStopping(monitor="val_loss", min_delta=1e-4, patience=10, verbose=False, mode="min")
lr_logger = LearningRateMonitor()
logger = TensorBoardLogger("tb_logs", log_graph=True)

# configure network and trainer
pl.seed_everything(42)
trainer = pl.Trainer(
    max_epochs=2,
    accelerator="auto",
    gradient_clip_val=0.1,
    limit_train_batches=30,
    logger=logger,
    callbacks=[lr_logger, early_stop_callback],
)

# Create TemporalFusionTransformer model
tft = TemporalFusionTransformer.from_dataset(
    training,
    learning_rate=0.03,
    hidden_size=16,
    attention_head_size=1,
    dropout=0.1,
    hidden_continuous_size=8,
    output_size=7,
    loss=QuantileLoss(),
    log_interval=10,
    log_val_interval=1,
    reduce_on_plateau_patience=3,
)
print(f"Number of parameters in network: {tft.size()/1e3:.1f}k")

# find optimal learning rate
# remove logging and artificial epoch size
tft.hparams.log_interval = -1
tft.hparams.log_val_interval = -1
trainer.limit_train_batches = 1.0

# run learning rate finder
res = Tuner(trainer).lr_find(
    tft, train_dataloaders=train_dataloader, val_dataloaders=val_dataloader, min_lr=1e-5, max_lr=1e2
)
print(f"suggested learning rate: {res.suggestion()}")
fig = res.plot(show=True, suggest=True)
# fig.show()
tft.hparams.learning_rate = res.suggestion()

trainer.fit(
    tft,
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)

# make a prediction on entire validation set
# preds, index = tft.predict(val_dataloader, return_index=True, fast_dev_run=True)
preds= tft.predict(val_dataloader, return_index=True, fast_dev_run=True)


# Optimize hyperparameters
study = optimize_hyperparameters(
    train_dataloader,
    val_dataloader,
    model_path="optuna_test",
    n_trials=200,
    max_epochs=50,
    gradient_clip_val_range=(0.01, 1.0),
    hidden_size_range=(8, 128),
    hidden_continuous_size_range=(8, 128),
    attention_head_size_range=(1, 4),
    learning_rate_range=(0.001, 0.1),
    dropout_range=(0.1, 0.3),
    trainer_kwargs=dict(limit_train_batches=30),
    reduce_on_plateau_patience=4,
    use_learning_rate_finder=False,
)

# Save the study object
with open("test_study.pkl", "wb") as fout:
    pickle.dump(study, fout)


# Profile speed
profile(
    trainer.fit,
    profile_fname="profile.prof",
    model=tft,
    period=0.001,
    filter="pytorch_forecasting",
    train_dataloaders=train_dataloader,
    val_dataloaders=val_dataloader,
)
