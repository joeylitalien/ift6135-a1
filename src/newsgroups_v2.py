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


    def train(self, nb_epochs, train_loader, test_loader, gen_gap=False):
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

            # Save losses and accuracies
            train_loss.append(total_loss / (batch_idx + 1))
            train_acc.append(self.predict(train_loader))
            test_acc.append(self.predict(test_loader))

            # Print stats
            print("Avg loss: %.4f -- Train acc: %.4f -- Test acc: %.4f" % 
                (train_loss[epoch], train_acc[epoch], test_acc[epoch]))
        
        # Print elapsed time
        end = datetime.datetime.now()
        elapsed = str(end - start)[:-7]
        print("Training done! Elapsed time: %s\n" % elapsed)

        return train_loss, train_acc, test_acc


if __name__ == "__main__":

    # Model parameters
    batch_size = 64
    layers = [61188, 100, 20]
    learning_rate = 0.2
    momentum = 0.9
    epsilon = 1e-5
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
                sep=" ", end="", flush=True)
        train_data = torch.load(saved + "train_data.pt")
        train_idf = torch.load(saved + "train_tfidf.pt")
        test_data = torch.load(saved + "test_data.pt")
        test_idf = torch.load(saved + "test_tfidf.pt")
        print(" done.\n")

    else:
        print("Loading training/test sets for the first time...",
                sep=" ", end="", flush=True)
        # Load .data and .label files
        train_data, train_idf, test_data, test_idf = load_newsgroups(
            train_filename, test_filename, layers[0], train_size, test_size)

        print(" done.\nSaving training/test sets...", 
                sep=" ", end="", flush=True)
        torch.save(train_data, saved + "train_data.pt")
        torch.save(train_idf, saved + "train_tfidf.pt")
        torch.save(test_data, saved + "test_data.pt")
        torch.save(test_idf, saved + "test_tfidf.pt")
        print(" done.\n")

    # Load datasets and create Torch loaders
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Build MLP and train
    mlp = Newsgroups(layers, learning_rate, momentum)
    mlp.train(10, train_loader, test_loader, len(train_loader.dataset))
