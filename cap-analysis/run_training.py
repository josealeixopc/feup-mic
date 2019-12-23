import sys

import training

if __name__ == '__main__':
    NUM_TIMESTEPS = 100000
    NUM_RUNS = 10

    # Run every combination of algorithm and environment 10 times
    for alg in training.AVAILABLE_ALGORITHMS:
        for env in training.AVAILABLE_ENVIRONMENTS:
            for _ in range(10):
                training.train(env, alg, NUM_TIMESTEPS)
