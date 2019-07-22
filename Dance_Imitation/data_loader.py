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

trial = 0 # trial 0-3
kid = 1 # kids from 0-44
move = 0 # moves from 0-17 or 0-13
seq = 1 #Sequence 1 or 3

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
	return gold_seq.loc[gold_index], kid_seq.loc[kid_index]

print(read_gold_seq(0))

''' this grabs the aligned kid and adult '''
print(read_sequences(kid, trial))

''' Gets the speed of the video for the trial '''
#print(sequence_data['speed'])

''' Gets the sequence supposed to be used for the trial '''
#print(sequence_data['seq'])

''' Use dataframe.drop[_, _, _] to get rid of the test date '''
''' use df.loc[_, _, _] to select the test data '''

''' Finds the id number given the kid index number: use with the pid data'''
#test = df_PT_scores[0][(df_PT_scores[0] == kid)]
#print(test)

''' Grabs beginning and end grame for the sequence and move type '''
def read_moves(kid, trial, seq, move):
	''' Picks the aligned gold move and seq with the kids '''
	move_frames = move_data['MoveTypeFrames_seq{}'.format(seq)][move]
	gold, kid = read_sequences(kid, trial)
	newGold = gold[gold[0].between(move_frames[0], move_frames[1], inclusive = True)]
	# grab indexes of gold start and end frames and grab kid from that
	#newKid = kid[kid[0].between(move_frames[0], move_frames[1], inclusive = True)]
	return newGold, newKid

#gold, kid = read_moves(kid, trial, seq, move)
#print(gold.head())
#print(kid.head())

# be able to find which kids dont have scores and not include them, per trial 
