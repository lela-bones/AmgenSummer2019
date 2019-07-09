import torch
import torch.nn as nn
from torch.utils import data

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class myDataset(data.Dataset):
    'My instantiation of the Dataset class'
    def __init__(self, feats, weights, labels):
        'Initialiation'
        self.feats = feats
        self.weights = weights
        self.labels = labels

    def __len__(self):
        'Returns the number of samples'
        return len(self.labels)

    def __getitem__(self, index):
        'Generate sample of data'
        x = torch.tensor(self.feats[index])
        w = torch.tensor(self.weights[index])
        y = torch.tensor(self.labels[index])
        return x, w, y

class myLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, batch_size):
        super(myLSTM, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.batch_size = batch_size
        
        # Initializing LSTM with size (input_size, hidden_size)
        self.lstm = nn.LSTM(input_size, hidden_size, batch_first = True, bidirectional = True)

        # Initializing output layer with size (hidden_size, output_size)
        self.linear = nn.Linear(self.hidden_size *2, output_size)

    def forward(self, inputs):
        '''Forward pass through the LSTM layer 
         LSTM inputs are of size (seq_len, batch, input_size)
         lstm_out size (input_size, batch_size, hidden_size)
         self.hidden size (num_layers, batch_size, hidden_size)'''
        lstm_out, self.hidden = self.lstm(inputs)
         
        # now only grab output from final set
        outputs = self.linear(lstm_out)
        #outputs = nn.functional.dropout(outputs, p=.2)
        return outputs

def one_hot(num, nb_classes):
	data = np.zeros(nb_classes)
