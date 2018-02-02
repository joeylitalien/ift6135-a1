# IFT6135: Representation Learning
# Assignment 1: Multilayer Perceptron (Problem 2)
# Authors: Samuel Laferriere & Joey Litalien

import torch
import torch.nn as nn
from torch.autograd import Variable
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset
from sklearn.datasets import fetch_20newsgroups
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
from utils import *


# Model parameters
batch_size = 1
h0, h1, h2 = 61188, 100, 20
learning_rate = 0.2
momentum = 0.9
nb_epochs = 20
epsilon = 1e-5
preprocess_scheme = "count"
data_dir = "./data"
cuda = False


def preprocess(corpus, scheme="tfidf"):
    """ Process data according to different schemes """

    if scheme == "count":
        vectorizer = CountVectorizer()
        vectors = vectorizer.fit_transform(corpus.data)

    elif scheme == "tfidf":
        vectorizer = TfidfVectorizer()
        vectors = vectorizer.fit_transform(corpus.data)

    else:
        print("Error!")
    
    return vectors


def load_data_sam(process):
    n_vocab = 61188
    n_train = 11269
    n_test = 7505

    X_train = torch.zeros(n_train, n_vocab).float()
    y_train = torch.zeros(n_train).long()

    X_test = torch.zeros(n_test, n_vocab).float()
    y_test = torch.zeros(n_test).long()

    with open("./20news-bydate/matlab/train.data", "rb") as f:
        for line in f:
            i,j,c = line.split()
            i,j,c = int(i), int(j), int(c)
            X_train[i-1][j-1] = c

    with open("./20news-bydate/matlab/train.label") as f:
        for i, line in enumerate(f):
            y_train[i] = int(line) - 1

    with open("./20news-bydate/matlab/test.data", "rb") as f:
        for line in f:
            i,j,c = line.split()
            i,j,c = int(i), int(j), int(c)
            X_test[i-1][j-1] = c

    with open("./20news-bydate/matlab/test.label") as f:
        for i, line in enumerate(f):
            y_test[i] = int(line) - 1


    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size)

    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size)

    return train_loader, test_loader


def load_data(process):
    """ Fetch newsgroup data and convert to tensor format """

    # Download training and test sets
    fetch_train = fetch_20newsgroups(data_home=data_dir, subset="train")
    fetch_test = fetch_20newsgroups(data_home=data_dir, subset="test")
    
    # Convert sets to tensors
    X_train = to_torch_sparse_tensor(preprocess(fetch_train, process)).to_dense()
    y_train = torch.Tensor(fetch_train.target) 

    X_test = to_torch_sparse_tensor(preprocess(fetch_test, process)).to_dense()
    y_test = torch.Tensor(fetch_test.target) 
  
    # Create training set data loaders
    train_data = TensorDataset(X_train, y_train)
    train_loader = DataLoader(train_data, batch_size=batch_size, shuffle=True)
   
    test_data = TensorDataset(X_test, y_test)
    test_loader = DataLoader(test_data, batch_size=batch_size, shuffle=True)

    return train_loader, test_loader


def init_weights(tensor):
    """ Glorot weight initialization """

    if isinstance(tensor, nn.Linear):
        tensor.bias.data.fill_(0)
        nn.init.xavier_uniform(tensor.weight.data)


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
    if cuda:
        model = model.cuda()
        loss_fn = loss_fn.cuda()

    return model, loss_fn, optimizer


def train(model, loss_fn, optimizer, train_data_loader, test_data_loader):
    """ Train model on data """

    # Initialize tracked quantities
    train_loss, train_acc, test_acc = [], [], []

    # Train
    for epoch in range(nb_epochs):
        print("Epoch %d/%d" % (epoch + 1, nb_epochs))
        total_loss = 0

        # Mini-batch SGD
        for i, (x, y) in enumerate(train_data_loader):
            # Print progress bar
            progress(i, 11269)

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
        test_acc.append(predict(test_data_loader, batch_size))
        
        print("Avg loss: %.4f -- Train acc: %.4f -- Test acc: %.4f" % \
                (train_loss[epoch], train_acc[epoch], test_acc[epoch]))


if __name__ == "__main__":
    model, loss_fn, optimizer = build_model()
    train_loader, test_loader = load_data_sam(preprocess_scheme)
    train(model, loss_fn, optimizer, train_loader, test_loader)
