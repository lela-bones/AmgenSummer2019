import matplotlib.pyplot as plt
import numpy as np
import torch

# visualization loss 
plt.plot(losses)
plt.xlabel("Number of iteration")
plt.ylabel("Loss")
plt.title("RNN: Loss vs Number of epochs in training")
plt.savefig('Lossgraph.png')
plt.show()

