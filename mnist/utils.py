import sys
import matplotlib.pyplot as plt


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


