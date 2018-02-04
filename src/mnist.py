# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 1: Multilayer Perceptrons (Problem 1)

Authors: 
    Samuel Laferriere <samlaf92@gmail.com>
    Joey Litalien <joey.litalien@mail.mcgill.ca>
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data.sampler import SubsetRandomSampler
import numpy as np
import datetime
from utils import *


# Model parameters
batch_size = 64
h0, h1, h2, h3 = 784, 512, 512, 10
learning_rate = 1e-2
#init = "glorot"
data_filename = "../data/mnist/mnist.pkl"


def init_weights(tensor, init):
    """ Weight initialization methods (default: Xavier) """

    if isinstance(tensor, nn.Linear):
        tensor.bias.data.fill_(0)
        if init == "zeros":
            tensor.weight.data.fill_(0)
        elif init == "normal":
            tensor.weight.data.normal_(0,1)
        else:
            nn.init.xavier_uniform(tensor.weight.data)


def predict(model, batch_size, data_loader):
    """ Evaluate model on dataset """

    correct = 0.
    for batch_idx, (x, y) in enumerate(data_loader):
        # Forward pass
        x, y = Variable(x).view(len(x), -1), Variable(y)
        if torch.cuda.is_available(): 
            x = x.cuda()
            y = y.cuda()
        
        # Predict
        y_pred = model(x)
        correct += (y_pred.max(1)[1] == y).sum().data[0] / batch_size 

    # Compute accuracy
    acc = correct / len(data_loader)
    return acc 


def build_model(input_dim, h1, h2, output_dim, init, lr):
    """ Initialize model parameters """

    # MLP with 2 hidden layers
    model = nn.Sequential(
                nn.Linear(input_dim, h1), 
                nn.ReLU(),
                nn.Linear(h1, h2), 
                nn.ReLU(),
                nn.Linear(h2, output_dim)
            )

    # Initialize weights
    weights = lambda tensor : init_weights(tensor, init)
    model.apply(weights)

    # Set loss function and gradient-descend optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr)

    # CUDA support
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    return model, loss_fn, optimizer


def train(model, loss_fn, optimizer, nb_epochs, Na, 
            train_loader, valid_loader, test_loader, gen_gap=False):
    """ Train model on data """

    # Initialize tracked quantities
    train_loss, train_acc, valid_acc, test_acc = [], [], [], []

    # Train
    start = datetime.datetime.now()
    for epoch in range(nb_epochs):
        print("Epoch %d/%d" % (epoch + 1, nb_epochs))
        total_loss = 0

        # Mini-batch SGD
        for batch_idx, (x, y) in enumerate(train_loader):
            # Print progress bar
            progress_bar(batch_idx, Na / batch_size)

            # Forward pass
            x, y = Variable(x).view(len(x), -1), Variable(y)
            if torch.cuda.is_available():
                x = x.cuda()
                y = y.cuda()

            # Predict
            y_pred = model(x)

            # Compute loss
            loss = loss_fn(y_pred, y)
            total_loss += loss.data[0]

            # Zero gradients, perform a backward pass, and update the weights
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # Save losses and accuracies
        train_loss.append(total_loss / (batch_idx + 1))
        train_acc.append(predict(model, train_loader))
        if valid_loader:
            valid_acc.append(predict(model, valid_loader))
        else:
            valid_acc.append(-1)
        if test_loader:
            test_acc.append(predict(model, test_loader))
        else:
            test_acc.append(-1)
      
        # Format printing depending on tracked quantities
        if valid_loader and test_loader and gen_gap:
            gen_gap = train_acc[epoch] - test_acc[epoch]
            print("Avg loss: %.4f -- Train acc: %.4f -- Val acc: %.4f -- Test acc: %.4f -- Gen gap %.4f" % (train_loss[epoch], train_acc[epoch], valid_acc[epoch], test_acc[epoch], gen_gap))

        if valid_loader and test_loader and not gen_gap:
            print("Avg loss: %.4f -- Train acc: %.4f -- Val acc: %.4f -- Test acc: %.4f" % 
                (train_loss[epoch], train_acc[epoch], valid_acc[epoch], test_acc[epoch]))
    
        if valid_loader and not test_loader:
            print("Avg loss: %.4f -- Train acc: %.4f -- Val acc: %.4f" % 
                (train_loss[epoch], train_acc[epoch], valid_acc[epoch]))

        if not valid_loader and not test_loader:
            print("Avg loss: %.4f -- Train acc: %.4f" % 
                (train_loss[epoch], train_acc[epoch]))

    # Print elapsed time
    end = datetime.datetime.now()
    elapsed = str(end - start)[:-7]
    print("Training done! Elapsed time: %s\n" % elapsed)

    return train_loss, train_acc, valid_acc, test_acc


def subsample_train(model, loss_fn, optimizer, ratio, train_loader):
    """ Train by subsampling training set """

    # Get random indices from training set
    train_size = len(train_loader.dataset)
    indices = list(range(train_size))
    
    # Subsample a training set
    sub_train_size = int(ratio * train_size)
    np.random.shuffle(indices)
    sub_train_idx = indices[:sub_train_size]

    # Create sampler/loader for subsampled training set
    sub_train_sampler = SubsetRandomSampler(sub_train_idx)
    sub_train_loader = torch.utils.data.DataLoader(
                            train_loader.dataset,
                            sampler=sub_train_sampler,
                            batch_size=batch_size)

    return sub_train_size, sub_train_loader


"""
if __name__ == "__main__":
    
    # Load datasets and create Torch loaders
    train_data, valid_data, test_data = unpickle(data_filename)

    train_loader = torch.utils.data.DataLoader(
                        train_data, 
                        batch_size=batch_size, 
                        shuffle=True)

    valid_loader = torch.utils.data.DataLoader(
                        valid_data,
                        batch_size=batch_size,
                        shuffle=True)

    test_loader = torch.utils.data.DataLoader(
                        test_data,
                        batch_size=batch_size,
                        shuffle=True)


    # Compile model
    model, loss_fn, optimizer = build_model(h0, h1, h2, h3, init="normal")
    train(model, loss_fn, optimizer, len(train_data), train_loader, valid_loader, [])


    # Train for different reduced-size training sets
    # ratios = [0.01, 0.02, 0.05, 0.1, 1.0]
"""
