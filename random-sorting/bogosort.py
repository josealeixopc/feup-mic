# SOURCE: https://en.wikipedia.org/wiki/Bogosort

import sys
import time
import random
from multiprocessing import Process, Queue

import numpy as np
import matplotlib.pyplot as plt
from scipy.special import factorial
from tqdm import tqdm

WORKER_COUNT = 8
TRIAL_COUNT = 10


def is_sorted(some_list):
    for x, y in zip(some_list[:-1], some_list[1:]):
        if x > y:
            return False
    return True


class Sorter(Process):
    def __init__(self, array, output, counts, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.array = array
        self.length = len(array)
        self.output = output
        self.count = 0
        self.counts = counts

    def run(self):
        while True:
            if self.output.empty():
                new_list = random.sample(self.array, k=len(self.array))
                self.count += self.length  # to check all items
                if is_sorted(new_list):
                    self.counts.put(self.count)
                    self.output.put(new_list)
                    break
            else:
                self.counts.put(self.count)
                break
                

def run_trial(list_len):
    trials = {"time": [], "cycles": []}
    
    for _ in tqdm(range(TRIAL_COUNT)):
        start_time = time.time()
        array = random.sample(list(range(list_len)), k=list_len)

        workers = []
        output = Queue()
        counts = Queue()
        for _ in range(WORKER_COUNT):
            w = Sorter(array, output, counts)
            workers.append(w)
            w.start()

        _result = output.get()

        total_count = 0
        for _ in range(WORKER_COUNT):
            total_count += counts.get()

        for _ in range(WORKER_COUNT):
            output.put("DEATH")

        for w in workers:
            w.join()

        end_time = time.time()
        trials["time"].append(end_time - start_time)
        trials["cycles"].append(total_count)

    return trials


def plot_chart(list_lengths, results):
    # init chart
    fig, axarr = plt.subplots(2, 1, figsize=(8, 6))

    # Average time graph
    # plot runtime to the graph
    for i, (length, trial) in enumerate(zip(list_lengths, results)):
        trial_time = np.ones(TRIAL_COUNT) * length
        axarr[0].plot(trial_time, np.log(trial["time"]), "rx", alpha=0.4)

    # plot average result
    avg_result = [np.log(sum(t["time"]) / len(t["time"])) for t in results]
    axarr[0].plot(list_lengths, avg_result, label="Average Result")

    # chart labels
    axarr[0].legend(loc=0)
    axarr[0].set_xlabel("Length of Initial List")
    axarr[0].set_ylabel("Average Time Elapsed - ln(seconds)")

    # average operation graph
    # plot cycles to graph
    for i, (length, trial) in enumerate(zip(list_lengths, results)):
        trial_ops = np.ones(TRIAL_COUNT) * length
        axarr[1].plot(trial_ops, np.log(trial["cycles"]), "rx", alpha=0.4)

    # plot average result
    avg_result = [np.log(sum(t["cycles"]) / len(t["cycles"])) for t in results]
    axarr[1].plot(list_lengths, avg_result, label="Average Result")

    # plot n . n!
    n_dot = np.log([n * factorial(n) for n in list_lengths])
    axarr[1].plot(list_lengths, n_dot, label=r"$n \cdot n!$",)

    # chart labels
    axarr[1].legend(loc=0)
    axarr[1].set_xlabel("Length of Initial List")
    axarr[1].set_ylabel("Average Time Elapsed - ln(Operations)")

    # Headings and set layout
    fig.suptitle("Parallel Bogosort")
    plt.tight_layout()

    # Save plot
    plt.savefig("bogosort.png")
    
    
def main():
    list_lengths = range(2, 11)  # random length for the unsorted lists
    trial_results = []

    # run trials and add results to array
    for list_len in list_lengths:
        trial_info = run_trial(list_len)
        trial_results.append(trial_info)

    # plot info to chart and save as png
    plot_chart(list_lengths, trial_results)


if __name__ == "__main__":
    sys.exit(main())