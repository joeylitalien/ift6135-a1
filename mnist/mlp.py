# IFT6135: Representation Learning
# Assignment 1: Multilayer Perceptron
# Authors: Samuel Laferriere & Joey Litalien

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
import torchvision.datasets as data
import torchvision.transforms as transforms
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
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


def init_weights(tensor):
    """ Weight initialization methods (default: Xavier) """

    def init_schemes(tensor, init="glorot"):
        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(0)
            if init == "zeros":
                tensor.weight.data.fill_(0)
            elif init == "uniform":
                tensor.weight.data.uniform_(0,1)
            else:
                nn.init.xavier_uniform(tensor.weight.data)

    return init_schemes(tensor, init)


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


def split(data, n, shuffle=True):
    """ Split dataset into two subsets (A,B)
        where |A| = n, |B| = |data| - n 
    """

    data_size = len(data)
    if n > data_size:
        print("Error: Cannot split dataset since n is too large")
        return -1
    else:
        indices = list(range(data_size))
        if shuffle:
            np.random.shuffle(indices)
    
        A_idx, B_idx = indices[:n], indices[n:]
        return A_idx, B_idx



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
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate)

    # CUDA support
    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    return model, loss_fn, optimizer


def train(model, loss_fn, optimizer, train_data_loader, 
            valid_data_loader, test_data_loader):
    """ Train model on data """

    # Initialize tracked quantities
    train_loss, train_acc, valid_acc, test_acc = [], [], [], []

    # Train
    for epoch in range(nb_epochs):
        print("Epoch %d/%d" % (epoch + 1, nb_epochs))
        total_loss = 0

        # Mini-batch SGD
        for i, (x, y) in enumerate(train_data_loader):
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
            loss = loss_fn(y_pred, y)
            total_loss += loss.data[0]

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save losses and accuracies
        train_loss.append(total_loss / (i + 1))
        train_acc.append(predict(train_data_loader, batch_size))
        if valid_data_loader:
            valid_acc.append(predict(valid_data_loader, batch_size))
        else:
            valid_acc.append(0)
        test_acc.append(predict(test_data_loader, batch_size))
        
        print("Avg loss: %.4f -- Train acc: %.4f -- Val acc: %.4f -- Test acc: %.4f" % \
                (train_loss[epoch], train_acc[epoch], valid_acc[epoch], test_acc[epoch]))


def train_subsample(model, loss_fn, optimizer, train_size, ratios):
    """ Train by subsampling original training set """
    
    # Create validation set with sampler
    train_idx, valid_idx = split(train_data, train_size)
    valid_sampler = SubsetRandomSampler(valid_idx)
    valid_data_loader = torch.utils.data.DataLoader(
                                    train_data,
                                    sampler=valid_sampler,
                                    batch_size=batch_size)

    for a in ratios:
        # Subsample a training set
        Na = int(a * len(train_idx))
        sub_train_idx = [train_idx[i] for i in 
                            np.random.choice(len(train_idx), 
                            size=Na, replace=False)]

        # Create sampler/loader for subsampled training set
        sub_train_sampler = SubsetRandomSampler(sub_train_idx)
        sub_train_data_loader = torch.utils.data.DataLoader(
                                    train_data,
                                    sampler=sub_train_sampler,
                                    batch_size=batch_size)

        # Train
        print("\na = %.2f, Na = %d" % (a, Na))
        train(model, loss_fn, optimizer, 
                sub_train_data_loader, valid_data_loader, test_loader)


if __name__ == "__main__":
    model, loss_fn, optimizer = build_model()

    train_size = 50000
    ratios = [0.01, 0.02, 0.05, 0.1, 1.0]
    train_subsample(model, loss_fn, optimizer, train_size, ratios)
