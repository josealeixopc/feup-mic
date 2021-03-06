import argparse
import csv
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
MIN_NUM_TIMESTEPS = None


def calculate_average_time_per_timestep(dirs):
    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        tslist.append(timesteps)

    xy_list = [ts2xy(timesteps_item, X_TIMESTEPS, Y_TIME_ELAPSED) for timesteps_item in tslist]

    min_num_timesteps = np.Inf
    for (i, (x, y)) in enumerate(xy_list):
        # Number of timesteps is cumulative
        min_num_timesteps = min(min_num_timesteps, x[-1])

    total_time = 0
    total_num_timesteps = 0

    for (i, (x, y)) in enumerate(xy_list):
        total_num_timesteps += x[-1]
        total_time += y[-1]

    average_time_per_timestep = total_time / total_num_timesteps

    return average_time_per_timestep


def plot_average_reward_per_number_of_timesteps(dirs):
    """
    Plots the average reward for cumulative timesteps of several runs.
    :param dirs: The list of directories with monitors for the runs that should be averaged.
    :return:
    """
    global MIN_NUM_TIMESTEPS

    tslist = []
    for folder in dirs:
        timesteps = load_results(folder)
        tslist.append(timesteps)

    xy_list = [ts2xy(timesteps_item, X_TIMESTEPS, Y_REWARDS) for timesteps_item in tslist]

    min_num_timesteps = np.Inf
    for (i, (x, y)) in enumerate(xy_list):
        min_num_timesteps = min(min_num_timesteps, x[-1])

    # Determine the dataset with less timesteps, so that we do not plot timesteps beyond that point
    MIN_NUM_TIMESTEPS = min(min_num_timesteps, MIN_NUM_TIMESTEPS)

    # Create new timestep axis and average reward axis
    x_timesteps = np.arange(0, MIN_NUM_TIMESTEPS, 5)
    y_average_rewards = np.zeros(len(x_timesteps))

    # Because the timesteps are different in each dataset, we need to
    for (i, (x, y)) in enumerate(xy_list):
        y_interpolated = np.interp(x_timesteps, x, y)
        y_average_rewards = np.add(y_average_rewards, y_interpolated)

    num_datasets = len(xy_list)
    average_y = np.true_divide(y_average_rewards, num_datasets)

    x_new, y_new = smooth_moving_average(x_timesteps, average_y, 100)

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
        plt.plot(x, y, linewidth=1, label='Run {}'.format(i))

    plt.xlim(left=minx)
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
        MIN_NUM_TIMESTEPS = 100000

        found_dirs = True

        average_time_per_timestep = {}

        for alg in training.AVAILABLE_ALGORITHMS:
            CURRENT_ALG = alg
            search_term = '-' + alg + '-' + env

            log_dirs = []

            for dirpath, dirnames, filenames in os.walk(TRAINING_INFO_DIR):
                for dirname in dirnames:
                    if dirname.endswith(search_term):
                        log_dirs.append(os.path.join(TRAINING_INFO_DIR, dirname))

            if len(log_dirs) == 0:
                found_dirs = False
                continue

            plot_average_reward_per_number_of_timesteps(log_dirs)

            average_time_per_timestep[alg] = calculate_average_time_per_timestep(log_dirs)

            LINE_NUMBER += 1

        if not found_dirs:
            continue

        if env == 'cpa_dense':
            plt.xlim(right=6000)
        plt.xlim(left=0)
        plt.title(env)
        plt.xlabel("Number of Timesteps")
        plt.ylabel("Average Reward")
        plt.tight_layout()
        plt.legend(handles=HANDLES)

        figure_file_name = 'figures' + os.path.sep + '{}.eps'.format(env)

        training.create_dir(figure_file_name)
        plt.savefig(figure_file_name, format='eps')

        plt.show()

        average_timestep_file = 'figures' + os.path.sep + '{}-rewards.csv'.format(env)

        training.create_dir(average_timestep_file)

        # Save average time per timestep file
        with open(average_timestep_file, 'w') as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=average_time_per_timestep.keys())
            writer.writeheader()
            writer.writerow(average_time_per_timestep)
