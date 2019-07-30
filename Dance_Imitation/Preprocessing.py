import numpy as np
import torch
import torch.nn as nn
from torch.utils import data
import pandas as pd
from DataLoading import *

maxl = 325

def batch_scores(nb_trials=4):
	trainScores = []
	testScores = []
	childIDs = sequence_data['PIDs']
	for i in range(nb_trials):
		if(i == 0):
			scores = read_scores(i)
			childID = np.delete(childIDs, 2, 0)
			for j, ID in enumerate(childID):
				for k in range(18):
					# testing set, but accounting for deleting the 2
					if( j == 9 or j == 16 or j == 28 or j == 39 or j == 43 ):
						testScores.append(scores.loc[j, k+1])
					else:
						trainScores.append(scores.loc[j, k+1])
		if(i == 1):
			scores = read_scores(i)
			childID = np.delete(childIDs, [4, 18, 26, 30], 0)
			for j, ID in enumerate(childID):
				for k in range(18):
					# testing set, accounting for deleted children
					if( j == 9 or j == 16 or j == 26 or j == 36 or j == 40 ):
						testScores.append(scores.loc[j, k+1])
					else:
						trainScores.append(scores.loc[j, k+1])
		if(i == 2):
			scores = read_scores(i)
			childID = np.delete(childIDs, 2, 0)
			for j, ID in enumerate(childID):
				for k in range(14):
					# testing set, accounting for deleted children
					if( j == 9 or j == 16 or j == 28 or j == 39 or j == 43 ):
						testScores.append(scores.loc[j, k+1])
					else:
						trainScores.append(scores.loc[j, k+1])
		if(i == 3):
			scores = read_scores(i)
			childID = np.delete(childIDs, 6, 0)
			for j, ID in enumerate(childID):
				for k in range(14):
					# testing set, accounting for deleted children
					if( j == 9 or j == 16 or j == 28 or j == 39 or j == 43 ):
						testScores.append(scores.loc[j, k+1])
					else:
						trainScores.append(scores.loc[j, k+1])
	return testScores, trainScores

def batch_sequences(nb_trials=4, nb_kids=45):
	trainChild = []
	trainGold = []
	testChild = []
	testGold = []
	for i in range(nb_trials):
		if(i == 0):
			for j in range(nb_kids):
				for k in range(18):
					if( j!=2 ):
						child, gold = read_moves(j, i, 1, k)
						# testing set, but accounting for deleting the 2
						if( j == 10 or j == 17 or j == 29 or j == 40 or j == 44 ):
							testChild.append(child.drop(0, axis=1).values)
							testGold.append(gold.drop(0, axis=1).values)
						else:
							trainChild.append(child.drop(0, axis=1).values)
							trainGold.append(gold.drop(0, axis=1).values)
		if(i == 1):
			for j in range(nb_kids):
				for k in range(18):
					if( j != 4 and j!=18 and j!=26 and j!=30 ):
						child, gold = read_moves(j, i, 1, k)
						# testing set, accounting for deleted children
						if( j == 10 or j == 17 or j == 29 or j == 40 or j == 44 ):
							testChild.append(child.drop(0, axis=1).values)
							testGold.append(gold.drop(0, axis=1).values)
						else:
							trainChild.append(child.drop(0, axis=1).values)
							trainGold.append(gold.drop(0, axis=1).values)
		if(i == 2):
			for j in range(nb_kids):
				for k in range(14):
					if( j != 2):
						child, gold = read_moves(j, i, 3, k)
						# testing set, accounting for deleted children
						if( j == 10 or j == 17 or j == 29 or j == 40 or j == 44 ):
							testChild.append(child.drop(0, axis=1).values)
							testGold.append(gold.drop(0, axis=1).values)
						else:
							trainChild.append(child.drop(0, axis=1).values)
							trainGold.append(gold.drop(0, axis=1).values)
		if(i == 3):
			for j in range(nb_kids):
				for k in range(14):
					if( j!= 6):
						child, gold = read_moves(j, i, 3, k)
						# testing set, accounting for deleted children
						if( j == 10 or j == 17 or j == 29 or j == 40 or j == 44 ):
							testChild.append(child.drop(0, axis=1).values)
							testGold.append(gold.drop(0, axis=1).values)
						else:
							trainChild.append(child.drop(0, axis=1).values)
							trainGold.append(gold.drop(0, axis=1).values)
	return trainGold, trainChild, testGold, testChild



def pad_sequences(sequences, max_len):
	''' This function takes a sequence and pads the data out to the max length '''
	nb_seq = len(sequences)
	data_per = 60
	len_per_seq = [s.shape[0] for s in sequences] 

	new_seq = np.ones((nb_seq, max_len, data_per))
	padding_mask = []
	for i, s in enumerate(sequences):
		new_seq[i, :len_per_seq[i], :] = s
		padding_mask.append(len_per_seq[i])
	return new_seq, padding_mask

class myDataset(data.Dataset):
	def __init__(self, child, gold, weight, scores):
		self.child = child
		self.gold = gold
		self.weight = weight
		self.scores = scores

	def __len__(self):
		return len(self.scores)

	def __getitem__(self, index):
		x1 = torch.tensor(self.child[index], dtype=torch.float32)
		x2 = torch.tensor(self.gold[index], dtype = torch.float32)
		w = self.weight[index]
		y = torch.tensor(self.scores[index], dtype=torch.float32)
		return x1, x2, w, y


