import argparse
import os
from typing import List

from stable_baselines.results_plotter import load_results, X_EPISODES, X_WALLTIME, X_TIMESTEPS
import matplotlib.pyplot as plt
import numpy as np
from scipy.interpolate import make_interp_spline

import training

Y_REWARDS = 'r'
Y_EPISODE_LENGTH = 'l'
Y_TIME_ELAPSED = 't'

lines = ["-", "--", ":", "-."]
markers = ['o', 'x', '+', '^']
colors = ['#000000', '#222222', '#444444', '#666666']

CURRENT_ALG = None
LINE_NUMBER = None
HANDLES = None
MAX_LENGTH = None


def plot_average_results(dirs, num_timesteps, xaxis, yaxis, task_name):
    global MAX_LENGTH

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)

    xy_list = [ts2xy(timesteps_item, xaxis, yaxis) for timesteps_item in tslist]

    max_length = len(xy_list[0][1])
    for l in xy_list:
        max_length = min(max_length, len(l[1]))

    MAX_LENGTH = min(max_length, MAX_LENGTH)

    average_y = np.zeros(max_length)

    for (i, (x, y)) in enumerate(xy_list):
        average_y = np.add(average_y, y[:max_length])

    average_y = np.true_divide(average_y, len(xy_list))

    x_new, y_new = smooth_moving_average(range(len(average_y)), average_y, 10)

    line, = plt.plot(x_new, y_new, linewidth=1, label=CURRENT_ALG)

    HANDLES.append(line)


def plot_results(dirs, num_timesteps, xaxis, yaxis, task_name):
    """
    plot the results

    :param dirs: ([str]) the save location of the results to plot
    :param num_timesteps: (int or None) only plot the points below this value
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param task_name: (str) the title of the task to plot
    """

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        if num_timesteps is not None:
            timesteps = timesteps[timesteps.l.cumsum() <= num_timesteps]
        tslist.append(timesteps)
    xy_list = [ts2xy(timesteps_item, xaxis, yaxis) for timesteps_item in tslist]
    plot_curves(xy_list, xaxis, yaxis, task_name)


def smooth_exponential_moving_average(scalars: List[float], weight: float) -> List[float]:  # Weight between 0 and 1
    """
    From here: https://stackoverflow.com/a/49357445
    :param scalars:
    :param weight:
    :return:
    """
    last = scalars[0]  # First value in the plot (first timestep)
    smoothed = list()
    for point in scalars:
        smoothed_val = last * weight + (1 - weight) * point  # Calculate smoothed value
        smoothed.append(smoothed_val)  # Save it
        last = smoothed_val  # Anchor the last smoothed value

    return smoothed


def smooth_moving_average(x, y, window_size):
    # Understanding convolution with window size: https://stackoverflow.com/a/20036959/7308982
    y_new = np.convolve(y, np.ones((window_size,)) / window_size, mode='valid')

    # We need to trim the last values, because the "valid" mode returns a list with size max(M, N) - min(M, N) + 1.
    # See here: https://docs.scipy.org/doc/numpy/reference/generated/numpy.convolve.html
    x_trimmed = x[:-(window_size - 1)]

    return x_trimmed, y_new


def plot_curves(xy_list, xaxis, yaxis, title):
    """
    plot the curves

    :param yaxis:
    :param xy_list: ([(np.ndarray, np.ndarray)]) the x and y coordinates to plot
    :param xaxis: (str) the axis for the x and y output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param title: (str) the title of the plot
    """

    plt.figure(figsize=(8, 2))

    minx = 0

    for (i, (x, y)) in enumerate(xy_list):
        # window_size = 10  # window size
        # x_new, y_new = smooth_moving_average(x, y, window_size)
        # y_new = smooth_exponential_moving_average(y, 0.8)

        plt.plot(x, y, linewidth=1, linestyle=lines[0], color=colors[0])

    plt.xlim(left=minx, right=100)
    plt.title(title)
    plt.xlabel(xaxis)
    plt.ylabel(yaxis)
    plt.tight_layout()


def ts2xy(timesteps, xaxis, y_axis):
    """
    Decompose a timesteps variable to x ans ys

    :param timesteps: (Pandas DataFrame) the input data
    :param xaxis: (str) the axis for the x output
        (can be X_TIMESTEPS='timesteps', X_EPISODES='episodes' or X_WALLTIME='walltime_hrs')
    :param y_axis: (str) the axis for the y output

    :return: (np.ndarray, np.ndarray) the x and y output
    """
    if y_axis == Y_REWARDS:
        y_var = timesteps.r.values
    elif y_axis == Y_EPISODE_LENGTH:
        y_var = timesteps.l.values
    elif y_axis == Y_TIME_ELAPSED:
        y_var = timesteps.t.values
    else:
        raise NotImplementedError

    if xaxis == X_TIMESTEPS:
        x_var = np.cumsum(timesteps.l.values)
    elif xaxis == X_EPISODES:
        x_var = np.arange(len(timesteps))
    elif xaxis == X_WALLTIME:
        x_var = timesteps.t.values / 3600.
    else:
        raise NotImplementedError

    return x_var, y_var


if __name__ == '__main__':
    TRAINING_INFO_DIR = "training_info"

    plt.style.use('ggplot')

    for env in training.AVAILABLE_ENVIRONMENTS:
        LINE_NUMBER = 0
        HANDLES = []
        MAX_LENGTH = 100000

        for alg in training.AVAILABLE_ALGORITHMS:
            CURRENT_ALG = alg
            search_term = '-' + alg + '-' + env

            log_dirs = []

            for dirpath, dirnames, filenames in os.walk(TRAINING_INFO_DIR):
                for dirname in dirnames:
                    if dirname.endswith(search_term):
                        log_dirs.append(os.path.join(TRAINING_INFO_DIR, dirname))

            plot_average_results(log_dirs, None, X_EPISODES, Y_REWARDS, "Generic title")

            LINE_NUMBER += 1

        plt.xlim(left=0, right=MAX_LENGTH)
        plt.title(env)
        plt.xlabel("Episodes")
        plt.ylabel("Average Reward")
        plt.tight_layout()
        plt.legend(handles=HANDLES)
        plt.show()
