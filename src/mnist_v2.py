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


def get_data_loaders(data_filename, batch_size):
    """ Load data from pickled file """

    train_data, valid_data, test_data = unpickle_mnist(data_filename)

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

    return train_loader, valid_loader, test_loader


def subsample_train(ratio, train_loader, batch_size):
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


class MNIST():

    def __init__(self, layers, learning_rate, init):
        """ Initialize multilayer perceptron """

        self.layers = layers
        self.learning_rate = learning_rate
        self.init = init
        self.compile()


    def init_weights(self, tensor, init):
        """ Weight initialization methods (default: Xavier) """

        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(0)
            if init == "zeros":
                tensor.weight.data.fill_(0)
            elif init == "normal":
                tensor.weight.data.normal_(0,1)
            else:
                nn.init.xavier_uniform(tensor.weight.data)


    def compile(self):
        """ Initialize model parameters """

        # MLP with 2 hidden layers
        self.model = nn.Sequential(
                        nn.Linear(self.layers[0], self.layers[1]), 
                        nn.ReLU(),
                        nn.Linear(self.layers[1], self.layers[2]), 
                        nn.ReLU(),
                        nn.Linear(self.layers[2], self.layers[3])
                    )

        # Initialize weights
        weights = lambda tensor : self.init_weights(tensor, self.init)
        self.model.apply(weights)

        # Set loss function and gradient-descend optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), 
                            lr=self.learning_rate)

        # CUDA support
        if torch.cuda.is_available():
            self.model = self.model.cuda()
            self.loss_fn = self.loss_fn.cuda()


    def predict(self, data_loader):
        """ Evaluate model on dataset """

        correct = 0.
        for batch_idx, (x, y) in enumerate(data_loader):
            # Forward pass
            x, y = Variable(x).view(len(x), -1), Variable(y)
            if torch.cuda.is_available(): 
                x = x.cuda()
                y = y.cuda()
            
            # Predict
            y_pred = self.model(x)
            correct += (y_pred.max(1)[1] == y).sum().data[0] / data_loader.batch_size 

        # Compute accuracy
        acc = correct / len(data_loader)
        return acc 


    def train(self, nb_epochs, train_loader, valid_loader, test_loader,
            Na, gen_gap=False):
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
                progress_bar(batch_idx, Na / train_loader.batch_size)

                # Forward pass
                x, y = Variable(x).view(len(x), -1), Variable(y)
                if torch.cuda.is_available():
                    x = x.cuda()
                    y = y.cuda()

                # Predict
                y_pred = self.model(x)

                # Compute loss
                loss = self.loss_fn(y_pred, y)
                total_loss += loss.data[0]

                # Zero gradients, perform a backward pass, and update the weights
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()

            # Save losses and accuracies
            train_loss.append(total_loss / (batch_idx + 1))
            train_acc.append(self.predict(train_loader))
            if valid_loader:
                valid_acc.append(self.predict(valid_loader))
            else:
                valid_acc.append(-1)
            if test_loader:
                test_acc.append(self.predict(test_loader))
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


if __name__ == "__main__":

    # Model parameters
    batch_size = 64
    layers = [784, 512, 512, 10]
    learning_rate = 1e-2
    nb_epochs = 3
    data_filename = "../data/mnist/mnist.pkl"

    # Load data
    train_loader, valid_loader, test_loader = get_data_loaders(data_filename, batch_size)

    # Build MLP and train
    mlp = MNIST(layers, learning_rate, "normal")
    mlp.train(10, train_loader, valid_loader, test_loader, len(train_loader.dataset))
