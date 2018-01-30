# IFT 6135: Representation Learning
# Assignment 1
# Authors: Samuel Laferrièe & Joey Litalien

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

# Model parameters
batch_size = 10
D_in, H1, H2, D_out = 784, 512, 256, 10
learning_rate = 1e-2
nb_epochs = 5
root = './data'

# Load MNIST dataset
mnist = transforms.Compose([transforms.ToTensor()])

train_loader = torch.utils.data.DataLoader(
    data.MNIST(root, train=True, download=True, transform=mnist),
    batch_size=batch_size, shuffle=True)

test_loader = torch.utils.data.DataLoader(
    data.MNIST(root, train=False, transform=mnist),
    batch_size=batch_size, shuffle=True)

# Plot loss/epoch
def plot(losses):
    plt.plot(losses, 'ro')
    plt.show()

# Weight initializer
def init_weights(m):
    if isinstance(m, nn.Linear):
        m.bias.data.fill_(0)
        # m.weight.data.fill_(0)
        # m.weight.data.normal_(0,1)
        # m.weight.data.uniform_(-1,1)
        # nn.init.xavier_uniform(m.weight.data)


# MLP with 2 hidden layers
model = nn.Sequential(
    nn.Linear(D_in, H1),
    nn.ReLU(),
    nn.Linear(H1, H2),
    nn.ReLU(),
    nn.Linear(H2, D_out)
    )

model.apply(init_weights)

# Loss function
criterion = nn.CrossEntropyLoss()
optimizer = optim.SGD(model.parameters(), lr=learning_rate)

losses = []

# Training
for t in range(nb_epochs):
    total_loss = 0

    # Mini-batch
    for batch_idx, (x, y) in enumerate(train_loader):
        # Forward pass
        x = x.view(batch_size, -1)
        x, y = Variable(x), Variable(y)
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
