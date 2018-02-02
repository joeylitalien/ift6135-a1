# -*- coding: utf-8 -*-
"""
IFT6135: Representation Learning
Assignment 1: Multilayer Perceptron (Problem 2)

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
import argparse
from utils import *


# Model parameters
batch_size = 100
h0, h1, h2 = 61188, 100, 20
learning_rate = 0.2
momentum = 0.9
nb_epochs = 20
epsilon = 1e-5
preprocess_scheme = "count"
train_filename = "../data/newsgroups/matlab/train"
test_filename = "../data/newsgroups/matlab/test"
saved = "../data/newsgroups/saved/"
train_size = 11269
test_size = 7505


def preprocess(corpus, scheme="tfidf"):
    """ Process data according to different schemes """

    if scheme == "count":
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(corpus.data)

    elif scheme == "tfidf":
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus.data)

    else:
        ValueError("Standardization not implemented yet!")
    
    return vectors


def init_weights(tensor):
    """ Glorot weight initialization """

    if isinstance(tensor, nn.Linear):
        tensor.bias.data.fill_(0)
        nn.init.xavier_uniform(tensor.weight.data)


def predict(data_loader):
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


def build_model():
    """ Initialize model parameters """

    # MLP with a single hidden layer
    model = nn.Sequential(
                nn.Linear(h0, h1), 
                nn.ReLU(),
                nn.Linear(h1, h2)
            )

    # Initialize weights
    model.apply(init_weights)

    # Set loss function and gradient-descend optimizer
    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, 
                    momentum=momentum)

    # CUDA support
    if torch.cuda.is_available():
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    return model, loss_fn, optimizer


def train(model, loss_fn, optimizer, train_loader, test_loader):
    """ Train model on data """

    # Initialize tracked quantities
    train_loss, train_acc, test_acc = [], [], []

    # Train
    for epoch in range(nb_epochs):
        print("Epoch %d/%d" % (epoch + 1, nb_epochs))
        total_loss = 0

        # Mini-batch SGD
        for batch_idx, (x, y) in enumerate(train_loader):
            # Print progress bar
            progress_bar(batch_idx, len(train_loader.dataset) / batch_size)

            # Forward pass
            x, y = Variable(x).view(len(x), -1), Variable(y)
            if torch.cuda.is_available():
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
        train_loss.append(total_loss / (batch_idx + 1))
        train_acc.append(predict(train_loader))
        test_acc.append(predict(test_loader))
        
        print("Avg loss: %.4f -- Train acc: %.4f -- Test acc: %.4f" % \
            (train_loss[epoch], train_acc[epoch], test_acc[epoch]))


if __name__ == "__main__":

    # Create argument parser
    parser = argparse.ArgumentParser(description="Newsgroups MLP")
    parser.add_argument("--load", help="unpickle preparsed tensors", 
                action="store_true")
    args = parser.parse_args()

    # Load pre-parsed data
    if args.load:
        print("Loading saved training/test datasets...", 
                sep=" ", end="", flush=True)
        train_data = torch.load(saved + "train_data.pt")
        train_idf = torch.load(saved + "train_idf.pt")
        test_data = torch.load(saved + "test_data.pt")
        test_idf = torch.load(saved + "test_idf.pt")
        print(" done.\n")

    else:
        # Load .data and .label files
        train_data, train_idf, test_data, test_idf = load_data(
            train_filename, test_filename, h0, train_size, test_size)

        print("Saving training/test datasets...", 
                sep=" ", end="", flush=True)
        torch.save(train_data, saved + "train_data.pt")
        torch.save(train_idf, saved + "train_idf.pt")
        torch.save(test_data, saved + "test_data.pt")
        torch.save(test_idf, saved + "test_idf.pt")
        print(" done.\n")

    # Load datasets and create Torch loaders
    train_loader = DataLoader(train_data, batch_size=batch_size)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    # Compile model
    model, loss_fn, optimizer = build_model()

    # Train
    train(model, loss_fn, optimizer, train_loader, test_loader)
