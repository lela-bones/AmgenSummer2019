import matplotlib.pyplot as plt
import numpy as np
import torch

splitz = [0, 1, 2, 3, 4]
batch = 10
hidden = 64
lr = .005
drop = 0
for split in splitz:
    losses = np.loadtxt('../Results/lossBi{}_{}_{}_{}_Drop{}'.format(split, batch, hidden, lr, drop))
#    test = np.loadtxt('../Results/test{}_{}_{}_{}_Drop{}.csv'.format(split, batch, hidden, lr, drop))
#    train = np.loadtxt('../Results/train{}_{}_{}_{}_Drop{}.csv'.format(split, batch, hidden, lr, drop))

    # visualization loss 
    #plt.ylim(0, 2.5)
    plt.plot(losses, label = 'Split{}'.format(split), linewidth = 1)
    plt.xlabel("Number of iteration")
    plt.ylabel("Loss")
    plt.legend()
    plt.title("LSTM: Loss vs Number of epochs in training")
    plt.savefig('../Graphs/lossBi{}_{}_{}_{}_{}.png'.format(split, batch, hidden, lr, drop))
    plt.show()

#for split in splitz:
#    losses = np.loadtxt('../Results/loss{}_{}_{}_{}_Drop{}'.format(split, batch, hidden, lr, drop))
    train = np.loadtxt('../Results/trainBi{}_{}_{}_{}_Drop{}.csv'.format(split, batch, hidden, lr, drop))

    # visualization training accuracy
    #plt.ylim(0, 1)
    plt.plot(train, label = 'Split{}'.format(split), linewidth = 1)
    plt.xlabel("Number of iteration")
    plt.ylabel("Train Accuracy")
    plt.legend()
    plt.title("LSTM: Train Accuracy vs Number of epochs in training")
    plt.savefig('../Graphs/trainBi{}_{}_{}_{}_{}.png'.format(split, batch, hidden, lr, drop))
    plt.show()


#for split in splitz:
    test = np.loadtxt('../Results/testBi{}_{}_{}_{}_Drop{}.csv'.format(split, batch, hidden, lr, drop))
    # visualization test accuracy 
    #plt.ylim(0, 1)
    plt.plot(test, label = 'Split{}'.format(split), linewidth=1)
    plt.xlabel("Number of iteration")
    plt.ylabel("Test Accuracy")
    plt.legend()
    plt.title("LSTM: Test Accuracy vs Number of epochs in training")
    plt.savefig('../Graphs/testBi{}_{}_{}_{}_{}.png'.format(split, batch, hidden, lr, drop))
    plt.show()

