# IFT 6135: Representation Learning
# Assignment 1: Multilayer Perceptron
# Authors: Samuel Laferriere & Joey Litalien

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
from utils import *


# Model parameters
batch_size = 50
h0, h1, h2, h3 = 784, 512, 512, 10
learning_rate = 1e-2
init = "glorot"
nb_epochs = 5
data_dir = "./data"
cuda = False


# MNIST dataset normalization
normalize = transforms.Compose([transforms.ToTensor(),
                transforms.Normalize((0.1307,), (0.3081,))])

# Training set
train_data = data.MNIST(root=data_dir, train=True, 
                download=True, transform=normalize)

# Training set loader
train_loader = torch.utils.data.DataLoader(
                    train_data, 
                    batch_size=batch_size, 
                    shuffle=True)

# Test set
test_data = data.MNIST(root=data_dir, train=False, 
                       download=False, transform=normalize)

# Test set loader
test_loader = torch.utils.data.DataLoader(
                    test_data,
                    batch_size=batch_size,
                    shuffle=True)


def plot(pts):
    """ Plot _ per epoch """

    plt.plot(pts, 'ro')
    plt.show()


def init_weights(tensor):
    """ Weight initialization methods (default: Xavier) """

    def weights(tensor, init="glorot"):
        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(0)
            if init == "zeros":
                tensor.weight.data.fill_(0)
            elif init == "uniform":
                tensor.weight.data.uniform_(0,1)
            else:
                nn.init.xavier_uniform(tensor.weight.data)

    return weights(tensor, init)


def predict(data_loader, batch_size):
    """ Evaluate model on dataset """

    correct = 0.
    for i, (x, y) in enumerate(data_loader):
        # Forward pass
        x, y = Variable(x).view(batch_size, -1), Variable(y)
        if cuda:
            x = x.cuda()
            y = y.cuda()
        
        # Predict
        y_pred = model(x)
        correct += (y_pred.max(1)[1] == y).sum().data[0] / batch_size 

    # Compute accuracy
    acc = correct / len(data_loader)
    return acc 


def split(data_loader, n):
    """ Split dataset into subset of size n """


def build_model():
    """ Initialize model parameters """

    # MLP with 2 hidden layers
    model = nn.Sequential(
                nn.Linear(h0, h1), 
                nn.ReLU(),
                nn.Linear(h1, h2), 
                nn.ReLU(),
                nn.Linear(h2, h3)
            )

    # Initialize weights
    model.apply(init_weights)

    # Set loss function and gradient-descend optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # CUDA support
    if cuda:
        model = model.cuda()
        criterion = criterion.cuda()

    return model, criterion, optimizer


def train(model, criterion, optimizer):
    """ Train model on data """

    # Initialize tracked quantities
    train_loss, train_acc, test_acc = [], [], []

    # Train
    for epoch in range(nb_epochs):
        print("Epoch %d/%d" % (epoch + 1, nb_epochs))
        total_loss = 0

        # Mini-batch SGD
        for i, (x, y) in enumerate(train_loader):
            # Print progress bar
            progress(i, len(train_loader))

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

        # Save losses and accuracies
        train_loss.append(total_loss / (i + 1))
        train_acc.append(predict(train_loader, batch_size))
        test_acc.append(predict(test_loader, batch_size))
        
        print("Avg loss: %.4f -- Train acc: %.4f -- Test acc: %.4f" % \
                (train_loss[epoch], train_acc[epoch], test_acc[epoch]))



if __name__ == "__main__":
    model, criterion, optimizer = build_model()
    train(model, criterion, optimizer)
