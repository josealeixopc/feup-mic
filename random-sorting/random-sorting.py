import random
import sys
import time

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from tqdm import tqdm

MAX_ITERATIONS = 1e100
TRIAL_COUNT = 10

def is_ordered_ascending(some_list):
    for x, y in zip(some_list[:-1], some_list[1:]):
        if x > y:
            return False
    return True

class Sorter:
  def __init__(self, name):
    self.name = name
    self.number_iter = 0
    self.number_operations = 0  # An operation is a single swap between two elements of the array (exchanging position of two elements)
    self.time_elapsed = 0

  def sort():
    pass

class RandomSorter(Sorter):

  def __init__(self):
    super().__init__("Random Sort")

  def sort(self, arr):
    self.number_iter = 0
    self.number_operations = 0
    self.time_elapsed = 0

    start_time = time.time()

    while self.number_iter < MAX_ITERATIONS:
      random.shuffle(arr)
      self.number_operations += len(arr) 
      # The shuffle operations iterates every element, swapping it with an element in a lower index
      # Therefore, it does N operations (where N is the length of the array)
      
      ordered = is_ordered_ascending(arr)
      self.number_operations += len(arr) - 1

      if(ordered):
        self.time_elapsed = time.time() - start_time
        break

class SelectionSorter(Sorter):
  def __init__(self):
    super().__init__("Selection Sort")

  def sort(self, arr):
    self.number_iter = 0
    self.number_operations = 0
    self.time_elapsed = 0

    start_time = time.time()
    # This value of i corresponds to how many values were sorted
    for i in range(len(arr)):
      # We assume that the first item of the unsorted segment is the smallest
      lowest_value_index = i
      # This loop iterates over the unsorted items
      for j in range(i + 1, len(arr)):
        self.number_operations += 1
        if arr[j] < arr[lowest_value_index]:
          lowest_value_index = j
      # Swap values of the lowest unsorted element with the first unsorted
      # element
      arr[i], arr[lowest_value_index] = arr[lowest_value_index], arr[i]
      self.number_operations += 1

    self.time_elapsed = time.time() - start_time

def run_trial(sorter, list_len):
  trials = {"name": sorter.name, "time": [], "operations": []}

  for _ in tqdm(range(TRIAL_COUNT)):
  # for _ in range(TRIAL_COUNT):
    random_array = random.sample(list(range(list_len)), k=list_len)
    sorter.sort(random_array)

    trials["time"].append(sorter.time_elapsed)
    trials["operations"].append(sorter.number_operations)

  return trials

def plot_chart(sorter, list_lengths, results):
    # init chart
    fig, axarr = plt.subplots(1, 1, figsize=(8, 3))

    # # Average time graph
    # # plot runtime to the graph
    # for i, (length, trial) in enumerate(zip(list_lengths, results)):
    #     trial_time = np.ones(TRIAL_COUNT) * length
    #     axarr[0].plot(trial_time, np.log(trial["time"]), "rx", alpha=0.4)

    # # plot average result
    # avg_result = [np.log(sum(t["time"]) / len(t["time"])) for t in results]
    # axarr[0].plot(list_lengths, avg_result, label="Average Result")

    # # chart labels
    # axarr[0].legend(loc=0)
    # axarr[0].set_xlabel("Length of Initial List")
    # axarr[0].set_ylabel("Average Time Elapsed - ln(seconds)")

    # average operation graph
    # plot cycles to graph
    for i, (length, trial) in enumerate(zip(list_lengths, results)):
        trial_ops = np.ones(TRIAL_COUNT) * length
        axarr.plot(trial_ops, np.log(trial["operations"]), "rx", alpha=0.4)

    # plot average result
    avg_result = [np.log(sum(t["operations"]) / len(t["operations"])) for t in results]
    axarr.plot(list_lengths, avg_result, label="Average Result", color='black')

    # plot n . n!
    n_dot = np.log([(2*n - 1) * factorial(n) for n in list_lengths])
    axarr.plot(list_lengths, n_dot, label=r"$(2n-1) \cdot n!$", color='black', linestyle='dashed')

    # chart labels
    axarr.legend(loc=0)
    axarr.set_xlabel("Length of Initial List")
    axarr.set_ylabel("Average Processing Time - ln(Operations)")

    # Headings and set layout
    fig.suptitle(sorter.name)
    plt.tight_layout()

    # Save plot

    file_name = sorter.name
    file_name = file_name.lower()
    file_name = file_name.replace(" ","-")
    plt.savefig(file_name + ".png")

def main():
    list_sorters = []

    list_sorters.append(RandomSorter())
    list_sorters.append(SelectionSorter())

    for sorter in list_sorters:
      list_lengths = range(2, 10)  # random length for the unsorted lists
      # results = {"sorter": sorter.name, }
      trial_results = []

      print ("Running trial with {}.".format(sorter.name))

      # run trials and add results to array
      for list_len in list_lengths:
          trial_info = run_trial(sorter, list_len)
          trial_results.append(trial_info)

      # print(trial_results)

      # plot info to chart and save as png
      plot_chart(sorter, list_lengths, trial_results)


if __name__ == "__main__":
    sys.exit(main())
