from preprocessing import *
from feat_readers import * 
import os
import numpy as np

# setting the path for the data
path = '../SpatialCNN_mid/'

# grabbing the different split folders
splits = os.listdir(path)
split = 0 #pick split 0-4
nb_classes = 18

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

# loading train data
for files in train_files:
    X_train.append(feats.read_feat(path + '{}/rgb-{}.avi.mat'.format(splits[split], files)))
    Y_train.append(feats.read_labels(path + '{}/rgb-{}.avi.mat'.format(splits[split], files), nb_classes))

# loading test data
for files in test_files:
    X_test.append(feats.read_feat(path + '{}/rgb-{}.avi.mat'.format(splits[split], files)))
    Y_test.append(feats.read_labels(path + '{}/rgb-{}.avi.mat'.format(splits[split], files), nb_classes))

# find the min and max video length
minl = 100000000
maxl = 0
for i in range(len(X_test)):
    temp = len(X_test[i])
    if(temp < minl):
        minl = temp
    if(temp > maxl):
        maxl = temp
#print(minl, maxl)

# padding all of the data
X_train, x_mask = pad_sequences(X_train, maxl)
X_test, x_mask = pad_sequences(X_test, maxl)

# starting with all -1 for all 128 features
Y = np.repeat(-1, maxl)
Y = np.tile(Y, (len(Y_train), 1))
for i in range(len(Y_train)):
    Y[i, :len(Y_train[i])] = Y_train[i]
Y_train = Y

Y = np.repeat(-1, maxl)
Y = np.tile(Y, (len(Y_test), 1))
for i in range(len(Y_test)):
    Y[i, :len(Y_test[i])] = Y_test[i]
Y_test = Y


