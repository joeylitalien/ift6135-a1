# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 1: Multilayer Perceptrons (Problem 2)

Authors: 
    Samuel Laferriere <samlaf92@gmail.com>
    Joey Litalien <joey.litalien@mail.mcgill.ca>
"""

from __future__ import print_function
import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import datetime
import argparse
from utils import *


class Newsgroups():

    def __init__(self, layers, learning_rate, momentum):
        """ Initialize multilayer perceptron """

        self.layers = layers
        self.learning_rate = learning_rate
        self.momentum = momentum
        self.compile()


    def init_weights(self, tensor):
        """ Weight initialization methods (default: Xavier) """

        if isinstance(tensor, nn.Linear):
            tensor.bias.data.fill_(0)
            nn.init.xavier_uniform(tensor.weight.data)


    def compile(self):
        """ Initialize model parameters """

        # MLP with 2 hidden layers
        self.model = nn.Sequential(
                        nn.Linear(self.layers[0], self.layers[1]), 
                        nn.ReLU(),
                        nn.Linear(self.layers[1], self.layers[2]) 
                    )

        # Initialize weights
        self.model.apply(self.init_weights)

        # Set loss function and gradient-descend optimizer
        self.loss_fn = nn.CrossEntropyLoss()
        self.optimizer = optim.SGD(self.model.parameters(), 
                            lr=self.learning_rate, momentum=self.momentum)

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
            correct += float((y_pred.max(1)[1] == y).sum().data[0]) / data_loader.batch_size 

        # Compute accuracy
        acc = correct / len(data_loader)
        return acc 


    def train(self, nb_epochs, train_loader, valid_loader, nb_max_updates=0):
        """ Train model on data """

        # Initialize tracked quantities
        train_loss, train_acc, valid_acc = [], [], []
        nb_updates = 0

        # Train
        break_switch = False
        start = datetime.datetime.now()
        for epoch in range(nb_epochs):
	   
            # Hack to get out if max number of updates reached
            if break_switch:
                break

            print("Epoch {:d}/{:d}".format(epoch + 1, nb_epochs))
            total_loss = 0

            # Mini-batch SGD
            for batch_idx, (x, y) in enumerate(train_loader):

                # Print progress bar only when ran normally
                if not nb_max_updates:
                    progress_bar(batch_idx, len(train_loader.dataset) / train_loader.batch_size)

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

                if nb_max_updates > 0:
                    train_loss.append(loss.data[0])
                    if nb_updates % 100 == 0:
                        print("Update {:4d}/{:d} -- Cur loss: {:.4f}".format(
                            nb_updates, nb_max_updates, loss.data[0]))
                    nb_updates += 1
                    if (nb_updates == nb_max_updates):
                        break_switch = True
                        break

            # Save losses and accuracies
            if not nb_max_updates:
                train_loss.append(total_loss / (batch_idx + 1))
                train_acc.append(self.predict(train_loader))
                valid_acc.append(self.predict(valid_loader))

                # Print stats
                print("Avg loss: {:.4f} -- Train acc: {:.4f} -- Val acc: {:.4f}".format( 
                    train_loss[epoch], train_acc[epoch], valid_acc[epoch]))
        
        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Elapsed time: %s\n" % elapsed)

        return train_loss, train_acc, valid_acc


if __name__ == "__main__":

    # Model parameters
    batch_size = 8
    layers = [61188, 100, 20]
    learning_rate = 0.2
    momentum = 0.9
    eps = 1e-5
    preprocess_scheme = "count"
    train_filename = "../data/newsgroups/matlab/train"
    test_filename = "../data/newsgroups/matlab/test"
    saved = "../data/newsgroups/saved/"
    train_size = 11269
    test_size = 7505

    # Create argument parser
    parser = argparse.ArgumentParser(description="Newsgroups MLP")
    parser.add_argument("--load", help="unpickle preparsed tensors", 
                action="store_true")
    args = parser.parse_args()

    # Load pre-parsed data
    if args.load:
        print("Loading saved training/test sets...", 
                sep=" ", end="")
        train_data = torch.load(saved + "train_data.pt")
        train_idf = torch.load(saved + "train_tfidf.pt")
        test_data = torch.load(saved + "test_data.pt")
        test_idf = torch.load(saved + "test_tfidf.pt")
        print(" done.\n")

    else:
        print("Loading training/test sets for the first time...",
                sep=" ", end="")
        # Load .data and .label files
        train_data, train_idf, test_data, test_idf = load_newsgroups(
            train_filename, test_filename, layers[0], train_size, test_size)

        print(" done.\nSaving training/test sets...", 
                sep=" ", end="")
        torch.save(train_data, saved + "train_data.pt")
        torch.save(train_idf, saved + "train_tfidf.pt")
        torch.save(test_data, saved + "test_data.pt")
        torch.save(test_idf, saved + "test_tfidf.pt")
        print(" done.\n")

    # Load datasets and create Torch loaders
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    train_stand = standardize(train_data, eps)
    test_stand = standardize(test_data, eps)
    train_loader_s = DataLoader(train_stand, batch_size=batch_size)
    test_loader_s = DataLoader(test_stand, batch_size=batch_size)
 
    # Build and train model
    mlp_s = Newsgroups(layers, learning_rate, momentum)
    mlp_s.train(1, train_loader_s, test_loader_s, len(train_loader_s.dataset))

    # Build MLP and train
    # mlp = Newsgroups(layers, learning_rate, momentum)
    # mlp.train(10, train_loader, test_loader, len(train_loader.dataset))
