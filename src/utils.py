import sys
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle


def progress_bar(count, total, status=""):
    """ Neat progress bar to track training """

    bar_size = 20
    filled = int(round(bar_size * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = u"\u25A0" * filled + " " * (bar_size - filled)
    sys.stdout.write("Training [%s] %s%s %s\r" % \
            (bar, percents, "%", status))
    sys.stdout.flush()


def plot_error_bars(x1, x2, err, xlabel, ylabel, title):
    """ Plot error bar graph """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.errorbar(x1, x2, err, linestyle="None", marker="o", 
        ecolor="g", c="b", elinewidth=1, capsize=3)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title)
    plt.show()


def plot_per_epoch(d, d_label, title):
    """ Plot graph; only takes a single list """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.scatter(range(1,len(d)+1), d, c="b", s=6, marker="o", label=d_label)
    ax.set_xlabel("Epoch")
    ax.set_ylabel(d_label)
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_title(title)
    plt.show()


def plots_per_epoch(d, d_labels, tracked_label, title):
    """ Plot graph; takes multiple sets of points """

    fig = plt.figure()
    ax = fig.add_subplot(111)
    colors = ["b", "r", "g", "m", "c", "y"]
    for i in range(len(d)):
        ax.scatter(range(1,len(d[i])+1), d[i], c=colors[i], s=6, 
            marker="o", label=d_labels[i])
    plt.legend(loc="lower right")
    ax.set_xlabel("Epoch")
    ax.xaxis.set_major_locator(MaxNLocator(integer=True))
    ax.set_ylabel(tracked_label)
    ax.set_title(title)
    plt.show()


def unpickle_mnist(filename):
    """ Load data into training/valid/test sets """

    # Unpickle files (uses latin switch for py2.x to py3.x compatibility)
    if sys.version_info[0] < 3:
        train, valid, test = pickle.load(open(filename, "rb"))
    else:
        train, valid, test = pickle.load(open(filename, "rb"), encoding="latin1")
    X_train, y_train = map(torch.from_numpy, train)
    X_valid, y_valid = map(torch.from_numpy, valid)
    X_test, y_test = map(torch.from_numpy, test)

    # Convert to tensors
    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)
    test_data = TensorDataset(X_test, y_test)
    
    return train_data, valid_data, test_data


def standardize(X, eps):
    """ Standardize tensor """

    means = X.mean(0)
    stds = X.std(0) + eps
    # stds = torch.zeros(61188)
    stand = (X - means) / stds
    # print((stds.sum(0) == 0).any())

    return stand


def load_newsgroups(train_filename, test_filename, vocab_size, train_size, test_size, pp, eps=1e-5):
    """ Load .data and .label files to retrieve dataset """

    def parse_data_file(fp, n, m):
        X = torch.zeros(n, m).float()
        idf = torch.ones(vocab_size)
        for line in fp:
            i, j, c = line.split()
            i, j, c = map(int, (i, j, c))
            X[i - 1][j - 1] = c
            idf[i - 1] += 1
        idf = (train_size / idf).log()
        return X, idf

    def parse_label_file(fp, n):
        y = torch.zeros(n).long()
        for i, line in enumerate(fp):
            y[i] = int(line) - 1
        return y

    # Build training/test sets, with idf matrix
    X_fp, y_fp = train_filename + ".data", train_filename + ".label"
    X_train, X_train_idf = parse_data_file(open(X_fp, "rb"), train_size, vocab_size)
    y_train = parse_label_file(open(y_fp, "rb"), train_size)
    
    X_fp, y_fp = test_filename + ".data", test_filename + ".label"
    X_test, X_test_idf = parse_data_file(open(X_fp, "rb"), test_size, vocab_size)
    y_test = parse_label_file(open(y_fp, "rb"), test_size)

    # Split training into train/valid sets
    N = len(X_train)
    indices = range(N)
    np.random.shuffle(indices)
    idx_train = indices[int(N/5):]
    idx_valid = indices[:int(N/5)]
    X_valid = X_train[idx_valid]
    X_valid_idf = X_train_idf[idx_valid]
    y_valid = y_train[idx_valid]
    X_train = X_train[idx_train]
    X_train_idf = X_train_idf[idx_train]
    y_train = y_train[idx_train]

    # Convert to tensors
    if pp == "count":
        train_data = TensorDataset(X_train, y_train)
        valid_data = TensorDataset(X_valid, y_valid)
        test_data = TensorDataset(X_test, y_test)

    elif pp == "tfidf":
        X_train_tfidf = torch.mul(X_train, X_train_idf.view(-1,1))
        X_valid_tfidf = torch.mul(X_valid, X_valid_idf.view(-1,1))
        X_test_tfidf = torch.mul(X_test, X_test_idf)
        train_data = TensorDataset(X_train_tfidf, y_train)
        valid_data = TensorDataset(X_valid_tfidf, y_valid)
        test_data = TensorDataset(X_test_tfidf, y_test)

    elif pp == "stand":
        X_train_stand = standardize(X_train, eps)
        X_valid_stand = standardize(X_valid, eps)
        X_test_stand = standardize(X_test, eps)
        train_data = TensorDataset(X_train_stand, y_train)
        valid_data = TensorDataset(X_valid_stand, y_valid)
        test_data = TensorDataset(X_test_stand, y_test)

    else:
      print("Processing step undefined.")
    
    return train_data, valid_data, test_data
