import sys
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch.utils.data import TensorDataset
import pickle


def progress(count, total, status=''):
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


def load_data(filename):
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


def to_torch_sparse_tensor(m):
    """ Convert SciPy sparse matrix to Torch sparse tensor """

    m = m.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((m.row, m.col))).long()
    values = torch.from_numpy(m.data)
    shape = torch.Size(m.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
