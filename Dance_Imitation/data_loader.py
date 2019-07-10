from scipy.io import loadmat
import numpy as np
import torch
import pandas as pd

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

trial = 0
kid = 44

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
	this function will return the correctly
	aligned kid, with the correctly aligned
	adult data '''

	gold_seq = read_gold_seq(trial)
	kid_seq = read_kid_sequence(kid, trial)
	gold_index = sequence_data['IY'][kid][trial].reshape(-1)
	gold_index = gold_index - 1
	kid_index = sequence_data['IX'][kid][trial].reshape(-1)
	kid_index = kid_index - 1
	return gold_seq.loc[gold_index], kid_seq.loc[kid_index]

''' this grabs the aligned kid and adult '''
#print(read_sequences(kid, trial))

''' Gets the speed of the video for the trial '''
#print(sequence_data['speed'])

''' Gets the sequence supposed to be used for the trial '''
#print(sequence_data['seq'])

''' Use dataframe.drop[_, _, _] to get rid of the test date '''
''' use df.loc[_, _, _] to select the test data '''

''' Finds the index number given the kids ID Number '''
#test = df_PT_scores[0][(df_PT_scores[0] == kid)]
#print(test)

print(move_data.keys())

