import matplotlib.pyplot as plt
import numpy as np
import torch

losses = np.load('losses.npy', allow_pickle=True)
test = np.load('testaccuracy.npy', allow_pickle=True)
train = np.load('trainaccuracy.npy', allow_pickle=True)

# visualization loss 
plt.plot(losses)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of epochs in training")
plt.savefig('Results/Lossgraph.png')
plt.show()

# visualization training accuracy
plt.plot(train)
plt.xlabel("Number of iteration")
plt.ylabel("Train Accuracy")
plt.title("RNN: Train Accuracy vs Number of epochs in training")
plt.savefig('Results/TrainAccuracy.png')
plt.show()

# visualization test accuracy 
plt.plot(test)
plt.xlabel("Number of iteration")
plt.ylabel("Test Accuracy")
plt.title("RNN: Test Accuracy vs Number of epochs in training")
plt.savefig('Results/TestAccuracy.png')
plt.show()

