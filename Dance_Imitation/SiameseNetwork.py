import torch
import statistics
import math
import numpy as np
import torch.nn as nn

class mySiameseNetwork(nn.Module):
	def __init__(self, input_size, hidden_size, batch_size):
		super(mySiameseNetwork, self).__init__()
		self.input_size = input_size
		self.hidden_size = hidden_size
		self.batch_size = batch_size

		#initializing the LSTM with size (input_size, hidden_size)
		self.lstm = nn.LSTM(input_size, hidden_size, batch_first=True, bidirectional = False)
	
	def forward(self, input1, input2):
		output1, h1 = self.lstm(input1)
		output2, h2 = self.lstm(input2)
		return output1, output2

class myContrastiveLoss(nn.Module):
	def __init__(self):
		super(myContrastiveLoss, self).__init__()

	def forward(self, output1, output2, label):
		distance = nn.functional.pairwise_distance(output1, output2, keepdim=True)
#		loss = torch.mean((label) * torch.pow(distance, 2) + (1-label) * torch.pow(torch.clamp(self.margin - distance, min=0.0), 2))
		loss = torch.mean(abs(label - (1 - distance)))
		return loss, distance

def sigmoid(x):
	x = 1 / (1+ math.exp(-1 * x))
	return x

class myBasicLoss(nn.Module):
	def __init__(self, margin=0):
		super(myBasicLoss, self).__init__()
		self.margin = margin

	def forward(self, output1, output2, label):
		distance = output1 - output2
		distance = np.linalg.norm(distance.cpu().detach().numpy(), 2)
		distance = 1 - sigmoid(distance)
		loss = statistics.mean(1 - (label - distance[0]))
		print(distance)

def newaccuracy(y_pred, y_true):
	count = 0.0
	for i in range(len(y_true)):
		if( abs(y_true[i] - (1 - y_pred[i])) < .2 ): 
			count += 1
	return count / len(y_pred)
	
def accuracy(y_pred, y_true):
	count = 0.0
	for i in range(len(y_true)):
		if(y_pred[i]!=y_true[i]):
			count += 1
	return count / len(y_pred)
	
