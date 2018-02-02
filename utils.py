import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle


def progress_bar(count, total, status=''):
    """ Neat progress bar to track training """

    bar_len = 30
    filled_len = int(round(bar_len * count / float(total)))
    percents = round(100.0 * count / float(total), 1)
    bar = u"\u25A0" * filled_len + " " * (bar_len - filled_len)
    sys.stdout.write("Training [%s] %s%s %s\r" % (bar, percents, "%", status))
    sys.stdout.flush()


def plot(pts):
    """ Plot _ per epoch """

    plt.plot(pts, 'ro')
    plt.show()


def unpickle(filename):
    """ Load data into training/valid/test sets """

    # Unpickle files (uses latin switch for py2.x to py3.x compatibility)
    train, valid, test = pickle.load(open(filename, "rb"), encoding="latin1")
    X_train, y_train = map(torch.from_numpy, train)
    X_valid, y_valid = map(torch.from_numpy, valid)
    X_test, y_test = map(torch.from_numpy, test)

    # Convert to tensors
    train_data = TensorDataset(X_train, y_train)
    valid_data = TensorDataset(X_valid, y_valid)
    test_data = TensorDataset(X_test, y_test)
    
    return train_data, valid_data, test_data


def load_data(train_filename, test_filename, vocab_size, train_size, test_size):
    """ Load .data and .label files to retrieve dataset """

    def parse_data_file(fp, n, m):
        X = torch.zeros(n, m).float()
        idf = torch.ones(1, vocab_size)
        for line in fp:
            i, j, c = line.split()
            i, j, c = map(int, (i, j, c))
            X[i - 1][j - 1] = c
            idf[0][j - 1] += 1
        return X, idf

    def parse_label_file(fp, n):
        y = torch.zeros(n).long()
        for i, line in enumerate(fp):
            y[i] = int(line) - 1
        return y

    # Build training/test sets, with idf matrix
    X_fp, y_fp = train_filename + ".data", train_filename + ".label"
    X_train, train_idf = parse_data_file(open(X_fp, "rb"), train_size, vocab_size)
    y_train = parse_label_file(open(y_fp, "rb"), train_size)
    
    X_fp, y_fp = test_filename + ".data", test_filename + ".label"
    X_test, test_idf = parse_data_file(open(X_fp, "rb"), test_size, vocab_size)
    y_test = parse_label_file(open(y_fp, "rb"), test_size)

    # Convert to tensors
    train_data = TensorDataset(X_train, y_train)
    test_data = TensorDataset(X_test, y_test)

    return train_data, train_idf, test_data, test_idf
