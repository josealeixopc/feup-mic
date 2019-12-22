import argparse

from stable_baselines.results_plotter import load_results, X_EPISODES, X_WALLTIME, X_TIMESTEPS
import matplotlib.pyplot as plt
import numpy as np

Y_REWARDS = 'r'
Y_EPISODE_LENGTH = 'l'
Y_TIME_ELAPSED = 't'


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
    maxx = max(xy[0][-1] for xy in xy_list)
    minx = 0
    for (i, (x, y)) in enumerate(xy_list):
        plt.scatter(x, y, s=2)
    plt.xlim(minx, maxx)
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


parser = argparse.ArgumentParser(description='Plot results from a trained model.')
parser.add_argument('log_dir',
                    help='The relative path of the monitor.csv dir.')

args = parser.parse_args()

plot_results([args.log_dir], None, X_EPISODES, Y_EPISODE_LENGTH, "Generic title")
plt.show()
