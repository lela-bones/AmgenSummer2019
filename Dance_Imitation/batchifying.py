import numpy as np
import torch
import pandas as pd
from  DataLoading import *
from Preprocessing import *


#scores = read_scores(1)
#scoreIDs = scores[0].values
#childIDs = sequence_data['PIDs']

#print(np.setdiff1d(scoreIDs, childIDs))
#print(np.setdiff1d(childIDs, scoreIDs))

#print(scores)

testScores, trainScores = batch_scores()
testScores = np.array(testScores)
testScores[np.where(testScores>1)] = 1
testScores[np.where(testScores<0)] = 0

trainScores = np.array(trainScores)
trainScores[np.where(trainScores>1)] = 1
trainScores[np.where(trainScores<0)] = 0


np.save('../../dance_results/testScores', testScores)
np.save('../../dance_results/trainScores', trainScores)
'''
testScores = np.load('../../dance_results/testScores.npy')
trainScores = np.load('../../dance_results/trainScores.npy')

for i, score in enumerate(trainScores):
	if(score >= .5):
		trainScores[i] = 1
	else:
		trainScores[i] = 0

for i, score in enumerate(testScores):
	if(score >= .5):
		testScores[i] = 1
	else:
		testScores[i] = 0

np.save('../../dance_results/testScores', testScores)
np.save('../../dance_results/trainScores', trainScores)

#trial = 3 # trial 0-3
#kid = 24  # kids from 0-44
#move = 0  # moves from 0-17 or 0-13
#seq = 1 #Sequence 1 or 3

#the maxlength of all of the sequences
maxl = 325

trainGold, trainChild, testGold, testChild = batch_sequences()
np.save('../../dance_results/trainGold', trainGold)
np.save('../../dance_results/trainChild', trainChild)
np.save('../../dance_results/testGold', testGold)
np.save('../../dance_results/testChild', testChild)

trainGold = np.load('../../dance_results/trainGold.npy')
testGold = np.load('../../dance_results/testGold.npy')
trainChild = np.load('../../dance_results/trainChild.npy')
testChild = np.load('../../dance_results/testChild.npy')


padded_testChild, mask_testChild = pad_sequences(testChild, maxl)
padded_testGold, mask_testGold= pad_sequences(testGold, maxl)
padded_trainChild, mask_trainChild = pad_sequences(trainChild, maxl)
padded_trainGold, mask_trainGold = pad_sequences(trainGold, maxl)


np.save('../../dance_results/padded_testChild', padded_testChild)
np.save('../../dance_results/mask_testChild', mask_testChild)
np.save('../../dance_results/padded_testGold', padded_testGold)
np.save('../../dance_results/mask_testGold', mask_testGold)
np.save('../../dance_results/padded_trainChild', padded_trainChild)
np.save('../../dance_results/mask_trainChild', mask_trainChild)
np.save('../../dance_results/padded_trainGold', padded_trainGold)
np.save('../../dance_results/mask_trainGold', mask_trainGold) 
'''
''' Gets the speed of the video for the trial '''
#print(sequence_data['speed'])

''' Gets the sequence supposed to be used for the trial '''
#print(sequence_data['seq'])

''' Use dataframe.drop[_, _, _] to get rid of the test data '''
''' use df.loc[_, _, _] to select the test data '''

''' Finds the id number given the kid index number: use with the pid data'''
#test = df_PT_scores[0][(df_PT_scores[0] == kid)]
#print(test)

#gold, kid = read_moves(kid, trial, seq, move)
#print(gold)
#print(kid)

# be able to find which kids dont have scores and not include them, per trial 
