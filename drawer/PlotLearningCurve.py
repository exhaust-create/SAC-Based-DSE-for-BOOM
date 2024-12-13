import os
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import pandas as pd
import numpy as np
 
def plot_result(episodes, records, title, ylabel, figure_file, label=None, xlabel='episode', ylim_bottom=None, ylim_top=None, dot=True):
    """
    Args
    -------
    episodes:
        The length of it must be the max num of "records[i]", where "i" is the num of lines that should be plotted. 
    records:
        Can be a list or ndarray.
    """
    plt.figure()
    if dot:
        plt.scatter(episodes, records, marker='.', color='r')
    else:
        if label is None:
            plt.plot(episodes, records, linestyle='-', color='r')
        else:
            for i in range(len(label)):
                plt.plot(episodes[:len(records[i])], records[i], label=label[i], linestyle='-')
            plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.title(title)
    plt.xlabel(xlabel, fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    # plt.gca().yaxis.set_major_formatter(ticker.FormatStrFormatter('%.4f'))
    plt.ylim(ylim_bottom, ylim_top)
    plt.tight_layout()
 
    # plt.show()
    plt.savefig(figure_file)
 
 
def create_directory(path: str, sub_dirs: list):
    for sub_dir in sub_dirs:
        if os.path.exists(path + sub_dir):
            print(path + sub_dir + ' is already exist!')
        else:
            os.makedirs(path + sub_dir, exist_ok=True)
            print(path + sub_dir + ' create successfully!')

def ema_plotting(episodes:np.ndarray, records:np.ndarray, span:int, title, ylabel, figure_file, label, std=None, ylim_bottom=None, ylim_top=None):
    """
    Description
    -------
    Calculate the Exponential Moving Average of the `records`.

    Args
    -------
    episodes: ndarray
        xaxis.
    records: ndarray(num_lines, num_epidoses)
        yaxis.
    span: int
        the span of ema.
    title: str
        the title of the figure.
    ylabel: str
        the ylabel of the figure.
    figure_file: str
        the path of the figure.
    label: list
        the label of each line.
    std:
        The data of standard deviation. If it's not None, then use the standard error to plot the figure.
    ylim_bottom: int
        the bottom limit of yaxis.
    ylim_top: int
        the top limit of yaxis.

    Returns
    -------
    None
    """
    records = records.reshape(-1, len(episodes))
    ema = pd.DataFrame(records).ewm(span=span, axis=1).mean().values.reshape(-1, len(episodes))
    reverse_records = records[::-1]
    reverse_ema = pd.DataFrame(reverse_records).ewm(span=span, axis=1).mean().values.reshape(-1, len(episodes))
    ema = (ema + reverse_ema[::-1]) / 2
    plt.figure()
    for i in range(len(ema)):
        plt.plot(episodes, ema[i], label=label[i], linestyle='-')
        if std is not None:
            std_error = std/np.sqrt(records.shape[0])
            plt.fill_between(episodes, ema[i] - std_error[i], ema[i] + std_error[i], alpha=0.2)
    plt.title(title)
    plt.xlabel('episode', fontsize=18)
    plt.ylabel(ylabel, fontsize=18)
    plt.legend(frameon=False, bbox_to_anchor=(1.05, 1), loc=2, borderaxespad=0)
    plt.ylim(ylim_bottom, ylim_top)
    plt.xticks(fontsize=16)
    plt.yticks(fontsize=16)
    plt.tight_layout()
 
    # plt.show()
    plt.savefig(figure_file)
