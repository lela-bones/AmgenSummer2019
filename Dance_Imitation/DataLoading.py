import numpy as np
import pandas as pd
from scipy.io import loadmat
import torch

#location of data
filepath = '/cis/project/vision_sequences/dance_sequences/dance_imitation/'

#files containing specific data
sequences = 'Data_sequences.mat'
moveTypes = 'MoveTypeFrames.mat'
scores = 'Score_per_movetype.mat'

#loading in the mat files
sequence_data = loadmat(filepath+sequences)
move_data = loadmat(filepath+moveTypes)
score_data = loadmat(filepath+scores)


move_data['MoveTypeFrames_seq1'][17][1] = 1803

def read_scores(trial):
	''' Accepts a trial 0: PT, 1: Test, 2: PT1, 3: TR60 
	    and returns a dataframe of that data 
            df.loc[kidID] is how to search it       '''
	
	df_scores = pd.DataFrame(score_data['Data'].flat[trial])
	return df_scores

def read_gold_seq(trial):
	''' Accepts the trial 0: PT, 1: Test, 2: PT1. 3: TR60
	    and returns a dataframe of the data 
            searchable by frame df.loc[start:end] '''

	df_gold_seq = pd.DataFrame(sequence_data['Gold'][0][0][trial])
	return df_gold_seq

def read_kid_sequence(kid, trial):
	df_kid_seq = pd.DataFrame(sequence_data['Kids'][kid][trial])
	return df_kid_seq

def read_sequences(kid, trial):
	''' Given the kid and the trial 
	this function will return the 
	correctly aligned kid, with the 
	correctly aligned adult data '''

	gold_seq = read_gold_seq(trial)
	kid_seq = read_kid_sequence(kid, trial)
	gold_index = sequence_data['IY'][kid][trial].reshape(-1)
	gold_index = gold_index - 1
	kid_index = sequence_data['IX'][kid][trial].reshape(-1)
	kid_index = kid_index - 1
	gold = gold_seq.loc[gold_index]
	# reset the frame index
	gold[0] = gold.iloc[0][0] + gold.index
	return gold, kid_seq.loc[kid_index]

''' Grabs beginning and end grame for the sequence and move type '''
def read_moves(kid, trial, seq, move):
	''' Picks the aligned gold move and seq with the kids '''
	move_frames = move_data['MoveTypeFrames_seq{}'.format(seq)][move]
	gold, kid = read_sequences(kid, trial)
	# reset the index so there are no repeats
	gold = gold.reset_index(drop=True)
	kid = kid.reset_index(drop=True)
	#gold[0] = gold.index + gold[0][0]
	# get the indices that we need from the specific frames
	start = gold.index[gold[0] == move_frames[0]].tolist()
	start = start[0]
	end = gold.index[gold[0] == move_frames[1]].tolist()
	end = end[-1]
	
	# create a list of the indices we need
	frames = [i for i in range(start, end+1)]
	
	# select proper indices from gold and kid
	newGold = gold.loc[frames]
	newKid = kid.loc[frames]
	return newGold, newKid


