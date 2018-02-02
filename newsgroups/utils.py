import sys
import matplotlib.pyplot as plt
import numpy as np
import torch

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


def to_torch_sparse_tensor(m):
    """ Convert SciPy sparse matrix to Torch sparse tensor """

    m = m.tocoo().astype(np.float32)
    indices = torch.from_numpy(np.vstack((m.row, m.col))).long()
    values = torch.from_numpy(m.data)
    shape = torch.Size(m.shape)
    
    return torch.sparse.FloatTensor(indices, values, shape)
