from preprocessing import *
from feat_readers import * 
import os
import numpy as np

# setting the path for the data
path = '../SpatialCNN_mid/'

# grabbing the different split folders
splits = os.listdir(path)
split = 0 #pick split 0-4

# grabbing the files in that split
file_feats = os.listdir(path + '{}'.format(splits[split]))

# defining the train and test files
train_files = np.loadtxt('train.txt', dtype = str)
test_files = np.loadtxt('test.txt', dtype = str)

# creating instance of feat read class
feats = LeaSpatialCNNFeatReader()

X_train = []
Y_train = []
X_test = []
Y_test = []

for files in train_files:
    x = feats.read_feat(path + '{}/rgb-{}.avi.mat'.format(splits[split], files))   
