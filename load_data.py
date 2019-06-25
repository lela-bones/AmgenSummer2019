from metrics import *
from  myLSTM import *
from preprocessing import *
from feat_readers import *
import os
import numpy as np
import matplotlib.pyplot as plt
import pdb

# setting the path for the data
#splits_path = '/cis/project/diva/dataset/50Salads/lea_splits/'
#feat_path = '/cis/project/diva/dataset/50Salads/SpatialCNN_feat/SpatialCNN_mid/'
splits_path ='../SpatialCNN_mid/'
# grabbing the different split folders
# Make sure they are sorted
splits = sorted(os.listdir(splits_path))
#print(splits)
split = 3 #pick split 0-4
nb_classes = 18

# defining the train and test files
print(splits_path +'{}/train.txt'.format(splits[split]))
print(splits_path +'{}/text.txt'.format(splits[split]))
train_files = np.loadtxt(splits_path +'{}/train.txt'.format(splits[split]), dtype = str)
test_files = np.loadtxt(splits_path + '{}/test.txt'.format(splits[split]), dtype = str)

# creating instance of feat read class
feats = LeaSpatialCNNFeatReader()

X_train = []
Y_train = []
X_test = []
Y_test = []

# loading train data
for files in train_files:
    X_train.append(feats.read_feat(splits_path + '{}/rgb-{}.avi.mat'.format(splits[split], files)))
    Y_train.append(feats.read_labels(splits_path + '{}/rgb-{}.avi.mat'.format(splits[split], files), nb_classes))

# loading test data
for files in test_files:
    X_test.append(feats.read_feat(splits_path + '{}/rgb-{}.avi.mat'.format(splits[split], files)))
    Y_test.append(feats.read_labels(splits_path + '{}/rgb-{}.avi.mat'.format(splits[split], files), nb_classes))

# find the min and max video length
#minl = 100000000
#maxl = 0
#for i in range(len(X_test)):
#    temp = len(X_test[i])
#    if(temp < minl):
#        minl = temp
#    if(temp > maxl):
#        maxl = temp
#print(minl, maxl)

# padding all of the data
maxl = 1655
X_train, x_train_mask = pad_sequences(X_train, maxl)
X_test, x_test_mask = pad_sequences(X_test, maxl)

#print(np.count_nonzero(x_test_mask))
#print(x_train_mask)

# starting with all -1 for all 128 features
Y = np.repeat(-1, maxl)
Y = np.tile(Y, (len(Y_train), 1))
for i in range(len(Y_train)):
    Y[i, :len(Y_train[i])] = Y_train[i]
Y_train = Y
#print(Y_train)

Y = np.repeat(-1, maxl)
Y = np.tile(Y, (len(Y_test), 1))
for i in range(len(Y_test)):
    Y[i, :len(Y_test[i])] = Y_test[i]
Y_test = Y
#print(Y_test)
#print(Y_train)
#print(np.array(X_train).shape)
#print(np.array(Y_train).shape)


# putting data into my dataset
train_data = myDataset(X_train, x_train_mask, Y_train)
test_data = myDataset(X_test, x_test_mask, Y_test)

batch_size = 10

# defining params for the data
train_params = {'batch_size': batch_size,
                'shuffle': True,
                'num_workers': 0}

test_params = {'batch_size': batch_size,
               'shuffle': False,
               'num_workers': 0}

# batch generators
train_gen = data.DataLoader(train_data, **train_params)
test_gen = data.DataLoader(test_data, **test_params)

hyper_params = {'input_size': 128,
                'hidden_size': 64,
                'output_size': nb_classes,
                'batch_size': 10}

#device config
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# initializing my model of the LSTM
mymodel = myLSTM(**hyper_params)
mymmodel = mymodel.to(device)

'''optional for loading'''
#mymodel.load_state_dict(torch.load('nnparams'))

# defining my loss and optimizer
criterion = nn.CrossEntropyLoss(ignore_index = -1)
optimizer = torch.optim.Adam(mymodel.parameters(), lr=.001)

#defining max epochs
num_epochs = 100

#saving intermediate information
losses = []
epochs = []
train_accuracies = []
test_accuracies = []
#training model
for i in range(num_epochs):
    mymodel.train(True)
    for feats, weights, labels in train_gen:
        feats = feats.to(device)
        labels = labels.to(device)

        #clearing gradient
        optimizer.zero_grad()

        #forward pass
        outputs = mymodel(feats)
	#reformatting data to fit into loss
        outputs = outputs.view((batch_size * maxl, -1))
        weights = weights.view(-1)
        labels = labels.view(-1)
        pred = torch.max(outputs, 1)[1]
        train_accuracy = per_frame_accuracy(labels.cpu().data.numpy(), pred.cpu().data.numpy(), weights)
        loss = criterion(outputs, labels)

        #backward optimize
        loss.backward()
        optimizer.step()

        del outputs, weights, labels, pred, feats

    mymodel.train(False)
    #calculating accuracy
    for feats, weights, labels in test_gen:
        feats = feats.to(device)
        labels = labels.to(device)

        #forward pass
        outputs = mymodel(feats)

        #reformatting data to fit into loss
        #outputs = outputs.view((batch_size * maxl, -1))
        #weights = weights.view(-1)
        #labels = labels.view(-1)
        pred = torch.max(outputs, -1)[1]
  #      print(pred)
  #      print(labels)
  #      pdb.set_trace()
	#use metrics here
        test_accuracy = per_frame_accuracy(labels.cpu().data.numpy(), pred.cpu().data.numpy(), weights)
    print('Epoch: {} Loss: {} Train Accuracy: {} Test Accuracy: {}'.format(i, loss.item(), train_accuracy, test_accuracy))
    #storing data
    losses.append(loss.cpu())
    train_accuracies.append(train_accuracy)
    test_accuracies.append(test_accuracy)
    #train_accuracies.append(accuracy)

torch.save(mymodel.state_dict(), 'split{}nnparams'.format(split+1))
np.save('losses', losses)
np.save('trainaccuracy', train_accuracies)
np.save('testaccuracy', test_accuracies)
'''
# visualization loss
plt.plot(epochs, losses)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of epochs in training")
plt.savefig('Lossgraph.png')
plt.show()
'''
'''
# visualization train accuracy
plt.plot([range(num_epochs)], train_accuracies, color = "red")
plt.xlabel("Number of iteration")
plt.ylabel("Accuracy")
plt.title("RNN: Accuracy vs Number of iteration")
plt.savefig('graph.png')
plt.show()

with torch.no_grad():
    for feats, weights, labels in test_gen:
        feats = feats.to(device)
        labels = labels.to(device)
        outputs = mymodel(feats)

        pred = torch.max(outputs, -1)[1]
        #use metrics here
        test_accuracy = per_frame_accuracy(labels.cpu().data.numpy(), pred.cpu().data.numpy(), weights)
        print(test_accuracy)
'''
