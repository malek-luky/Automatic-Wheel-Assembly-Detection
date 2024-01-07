import torch 
import torch.nn as nn # torch.nn allows us to create a neural network.
import torch.nn.functional as F # nn.functional give us access to the activation and loss functions.
from torch.optim import Adam # optim contains many optimizers. This time we're using Adam

import lightning as L # lightning has tons of cool tools that make neural networks easier


## params to be removed to config later
seed_value = 42
input_s = 6
hidden_s = 1

# params to be removed to config later ##


class LightningLSTM(L.LightningModule):

    def __init__(self): # __init__() is the class constructor function, and we use it to initialize the Weights and Biases.
        
        super().__init__() # initialize an instance of the parent class, LightningModule.

        L.seed_everything(seed=seed_value)
        
        ## input_size = number of features (or variables) in the data. In our example
        ##              we only have 6 values 
        ## hidden_size = this determines the dimension of the output
        ##               in other words, if we set hidden_size=1, then we have 1 output node
        ##               if we set hiddeen_size=50, then we hve 50 output nodes (that can then be 50 input
        ##               nodes to a subsequent fully connected neural network.
        self.lstm = nn.LSTM(input_size=input_s, hidden_size=hidden_s) 
         

    def forward(self, input):
        ## transpose the input vector
        input_trans = input.view(len(input), 1)
        
        lstm_out, temp = self.lstm(input_trans)
        
        ## lstm_out has the short-term memories for all inputs. We make our prediction with the last one
        prediction = lstm_out[-1] 
        return prediction
    
    
    def configure_optimizers(self): # this configures the optimizer we want to use for backpropagation.
        return Adam(self.parameters(), lr=0.1) ## we'll just go ahead and set the learning rate to 0.1

    
    def training_step(self, batch, batch_idx): # take a step during gradient descent.
        input_i, label_i = batch # collect input
        output_i = self.forward(input_i[0]) # run input through the neural network
        loss = (output_i - label_i)**2 ## loss = squared residual
        
        ###################
        ##
        ## Logging the loss and the predicted values so we can evaluate the training
        ##
        ###################
        self.log("train_loss", loss)
        
        if (label_i == 0):
            self.log("out_0", output_i)
        else:
            self.log("out_1", output_i)

        return loss