import torch
import torch.nn as nn
from torch.utils import data

class myDataset(data.Dataset):
    'My instantiation of the Dataset class'
    def __init__(self, feats, labels):
        'Initialiation'
        self.feats = feats
        self.labels = labels

    def __len__(self):
        'Returns the number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generate sample of data'
        x = torch.tensor(self.feats[index])
        y = torch.tensor(self.labels[index])
        return x, y

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        # Initializing LSTM with size (input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size)

        # Initializing output layer with size (hidden_size, output_size)
        self.linear = nn.Linear(self.hidden_size, output_size)

    def forward(self, inputs):
        '''Forward pass through the LSTM layer 
         LSTM inputs are of size (seq_len, batch, input_size)
         lstm_out size (input_size, batch_size, hidden_size)
         self.hidden size (num_layers, batch_size, hidden_size)'''
        lstm_out, self.hidden = self.lstm(inputs.view(len(inputs), self.batch_size, -1))
         
        # now only grab output from final set
        y_pred = self.linear(lstm_out[-1].view(self.batch_size, -1))
        return y_pred




        

