import numpy as np
import statistics
import math
import torch
import pandas as pd
from torch.utils.data import DataLoader
from  DataLoading import *
from Preprocessing import *
from SiameseNetwork import *

#trial = 3 # trial 0-3
#kid = 24  # kids from 0-44
#move = 0  # moves from 0-17 or 0-13
#seq = 1 #Sequence 1 or 3

mask_testGold = np.load('../../dance_results/mask_testGold.npy')
#mask_testChild = np.load('../../dance_results/mask_testChild.npy')
#mask_trainChild  = np.load('../../dance_results/mask_trainChild.npy')
mask_trainGold = np.load('../../dance_results/mask_trainGold.npy')
testChild = np.load('../../dance_results/padded_testChild.npy')
testGold = np.load('../../dance_results/padded_testGold.npy')
trainChild = np.load('../../dance_results/padded_trainChild.npy')
trainGold = np.load('../../dance_results/padded_trainGold.npy')
trainScores = np.load('../../dance_results/trainScores.npy')
testScores = np.load('../../dance_results/testScores.npy')

#revtestChild = np.flip(testChild, 1)
#newtestChild = np.concatenate((testChild, revtestChild))

#revtestGold = np.flip(testGold, 1)
#newtestGold = np.concatenate((testGold, revtestGold))

#revtrainChild = np.flip(trainChild, 1)
#newtrainChild = np.concatenate((trainChild, revtrainChild))

#revtrainGold = np.flip(trainGold, 1)
#newtrainGold = np.concatenate((trainGold, revtrainGold))

#revmask_trainGold = np.flip(mask_trainGold, 1)
#newtrainGold = np.concatenate((mask_trainGold, revmask_trainGold))

#revmask_testGold = np.flip(mask_testGold, 1)
#newtestGold = np.concatenate((mask_testGold, revmask_testGold))

#revtestScores = np.flip(testScores, 0)
#revtrainScores = np.flip(trainScores, 0)
#newtestScores = np.concatenate((testScores, revtestScores))
#newtrainScores = np.concatenate((trainScores, revtrainScores))


#train_data = myDataset(newtrainChild, newtrainGold, newmask_trainGold, newtrainScores)
#test_data = myDataset(newtestChild, newtestGold, newmask_testGold, newtestScores)

train_data = myDataset(trainChild, trainGold, mask_trainGold, trainScores)
test_data = myDataset(testChild, testGold, mask_testGold, testScores)

###################
# Hyper Parameters#
###################
batch_size = 25
hidden_size = 64
lr_rate = .001

# train and testing parameters for batching
train_params = {'batch_size': batch_size,
		'shuffle': True,
		'num_workers': 0}
test_params = {'batch_size': batch_size,
		'shuffle': False,
		'num_workers': 0}

#batch generators
train_gen = DataLoader(train_data, **train_params)
test_gen = DataLoader(test_data, **test_params)

hyper_params = {'input_size': 60,
		'hidden_size': hidden_size,
		'batch_size': batch_size}

#device configuration
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

#intializing my model of Siamese NN
mymodel = mySiameseNetwork(**hyper_params)
mymodel = mymodel.to(device)

# defining metric loss and optimizer
criterion = myContrastiveLoss()
#criterion = myBasicLoss()
optimizer = torch.optim.Adam(mymodel.parameters(), lr=lr_rate)

num_epochs = 50

#saving intermediate infromation
losses = []
train_accuracies = []
test_accuracies = []

for i in range(num_epochs):
	mymodel.train(True)
	for child, gold, weights, scores in train_gen:
		child = child.to(device)
		gold = gold.to(device)
		scores = scores.to(device)
		#clearing gradients
		optimizer.zero_grad()
		#forward pass
		output1, output2= mymodel(child, gold)
		output1 = torch.stack([output1[j, w-1, :] for j, w in enumerate(weights)])
		output2 = torch.stack([output2[j, w-1, :] for j, w in enumerate(weights)])
		# Use the weight matrix
		loss, y_pred = criterion(output1, output2, scores)
		acc = newaccuracy(y_pred, scores)
		loss.backward()
		optimizer.step()
	print(y_pred.t(), scores)	
	mymodel.train(False)
	for child, gold, weights, scores in train_gen:
		child = child.to(device)
		gold = gold.to(device)
		scores = scores.to(device)
		output1, output2 = mymodel(child, gold)
		output1 = torch.stack([output1[j, w-1, :] for j, w in enumerate(weights)])
		output2 = torch.stack([output2[j, w-1, :] for j, w in enumerate(weights)])
		loss, y_pred = criterion(output1, output2, scores)
		test_acc = newaccuracy(y_pred, scores)

	print("Epoch number {}\n Current loss {}\n Current accuracy {}\n Test accuracy {}".format(i+1, loss.item(), acc, test_acc))
	train_accuracies.append(acc)
	losses.append(loss.item())
	test_accuracies.append(test_acc)
test = 1
np.save('../../dance_results/losses{}'.format(test), losses)	
np.save('../../dance_results/acc{}'.format(test), train_accuracies)
np.save('../../dance_results/testacc{}'.format(test), test_accuracies)
torch.save(mymodel.state_dict(), 'noAugmentDataSNN{}'.format(test))
