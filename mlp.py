# IFT 6135: Representation Learning
# Assignment 1
# Authors: Samuel Laferrièe & Joey Litalien

import torch
from torch.autograd import Variable
import torch.nn as nn
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms

# Model parameters
batch_size = 100
D_in, H1, H2, D_out = 784, 512, 256, 10
learning_rate = 1e-4
nb_epochs = 10
root = './data'

# Load MNIST dataset
norm = transforms.Compose([transforms.ToTensor(), 
    transforms.Normalize((0.5,), (1.0,))])

train_loader = torch.utils.data.DataLoader(
    data.MNIST(root, train=True, download=True, transform=norm),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    data.MNIST(root, train=False, transform=norm),
    batch_size=batch_size, shuffle=True)

# MLP with 2 hidden layers
model = nn.Sequential(
    nn.Linear(D_in, H1),
    nn.ReLU(),
    nn.Linear(H1, H2),
    nn.ReLU(),
    nn.Linear(H2, D_out)
    )

# Loss function
criterion = nn.BCELoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

# Training
for t in range(nb_epochs):
    # Mini-batch
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass
        x = x.view(batch_size, -1)
        x, y = Variable(x), Variable(y)
        y_pred = model(x)
        print(x.size(), y.size(), y_pred.size())

        # Compute and print loss
        loss = criterion(y_pred, y)
        print(t, loss.data[0])

        # Zero gradients, perform a backward pass, and update the weights
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
