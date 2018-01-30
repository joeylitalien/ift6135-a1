# IFT 6135: Representation Learning
# Assignment 1: Multilayer Perceptron
# Authors: Samuel Laferrière & Joey Litalien

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt


# Model parameters
batch_size = 100
h0, h1, h2, h3 = 784, 512, 256, 10
learning_rate = 1e-2
nb_epochs = 10
root = './data'
cuda = False

# torch.manual_seed(17)

# Load MNIST dataset (normalized)
mnist = transforms.Compose([transforms.ToTensor(),
                            transforms.Normalize((0.1307,), (0.3081,))])

train_loader = torch.utils.data.DataLoader(
    data.MNIST(root, train=True, download=True, transform=mnist),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    data.MNIST(root, train=False, transform=mnist),
    batch_size=batch_size, shuffle=True)

#print(iter(train_loader).next()[0][0])

def plot(losses):
    """ Plot loss/epoch """
    plt.plot(losses, 'ro')
    plt.show()

def init_weights(m):
    """ Initialize weights """
    if isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
        # m.bias.data.normal_(0,1)
        # m.weight.data.fill_(0)
        # m.weight.data.normal_(0,1)
        # m.weight.data.uniform_(0,1)
        nn.init.xavier_uniform(m.weight.data)


# MLP with 2 hidden layers
model = nn.Sequential(
            nn.Linear(h0, h1), nn.ReLU(),
            nn.Linear(h1, h2), nn.ReLU(),
            nn.Linear(h2, h3)
        )

# Initialize weights
model.apply(init_weights)
#print(list(model.parameters()))

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Cuda support
if cuda:
    model = model.cuda()
    criterion = criterion.cuda()

losses = []

# Training
for t in range(nb_epochs):
    total_loss = 0

    # Mini-batch SGD
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass
        x, y = Variable(x).view(batch_size, -1), Variable(y)
        if cuda:
            x = x.cuda()
            y = y.cuda()

        # Predict
        y_pred = model(x)

        # Compute and print loss
        loss = criterion(y_pred, y)
        total_loss += loss.data[0]

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

    losses.append(total_loss / (batch_idx + 1))
    print("Epoch %d -- Avg Loss: %f" % (t, losses[t]))

plot(losses)
