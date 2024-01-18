import pytorch_lightning as pl
import torch
import torch.nn as nn


class TireAssemblyLSTM(pl.LightningModule):
    def __init__(self, input_size, hidden_layer_size, output_size):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_layer_size, batch_first=True)
        self.linear = nn.Linear(hidden_layer_size, output_size)
        self.hidden_layer_size = hidden_layer_size

    def forward(self, x):
        # Initializing hidden state for first input using method defined below
        h0, c0 = self.init_hidden(x.size(0))
        # Forward propagate LSTM
        out, _ = self.lstm(x, (h0, c0))  # out: tensor of shape (batch_size, seq_length, hidden_layer_size)
        # Decode the hidden state of the last time step
        out = self.linear(out[:, -1, :])
        return out

    def init_hidden(self, batch_size):
        # Generate the initial hidden state and cell state for the LSTM
        h0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(self.device)
        c0 = torch.zeros(1, batch_size, self.hidden_layer_size).to(self.device)
        return h0, c0

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.BCEWithLogitsLoss()(y_pred, y.unsqueeze(1))

        # y_pred are logits, so we need to convert them to probabilities
        y_pred = torch.sigmoid(y_pred)
        accuracy = (y == (y_pred > 0.5)).float().mean()

        self.log("train_loss", loss)
        self.log("train_accuracy", accuracy)
        return loss

    def test_step(self, batch, batch_idx):
        x, y = batch
        y_pred = self(x)
        loss = nn.BCEWithLogitsLoss()(y_pred, y.unsqueeze(1))

        y_pred = torch.sigmoid(y_pred)
        accuracy = (y == (y_pred > 0.5)).float().mean()

        self.log("test_loss", loss)
        self.log("test_accuracy", accuracy)
        return {"test_loss": loss}

    def configure_optimizers(self):
        return torch.optim.Adam(self.parameters(), lr=0.001)
