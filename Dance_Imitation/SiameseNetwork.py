import torch
import torch.nn as nn

class mySiameseNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, output_size, batch_size):
		super(SiameseNetwork, slef).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.output_size = output_size
		self.batch_size = batch_size

		#initializing the LSTM with size (input_size, hidden_size)
		self.lstm1 = nn.LSTM(input_size, hidden_size, batch_first=True)
		#self.lstm2 = nn.LSTM(input_size, hidden_size, batch_first=True)

	def forward_per(self, x):
		output = self.lstm1(x)
		return output

	def forward(self, input1, input2):
		output1 = self.forward_per(input1)
		output2 = self.forward_per(input2)
		return output1, output2
		

class myContrastiveLoss(nn.Module):
	def __init__(self, margin=1.0):
		super(myContrastiveLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		distance = nn.F.pairwise_distance(output1, output2, keepdim = True)
		loss = torch.mean((1-label) * torch.pow(distance, 2) + (label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
		return loss
