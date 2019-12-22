import argparse
import errno
import os
from datetime import datetime

AVAILABLE_ALGORITHMS = ['acktr', 'ppo', 'a2c', 'dqn']
AVAILABLE_ENVIRONMENTS = ['cpa_sparse', 'cpa_dense']
DEFAULT_TIMESTEPS = 100000

TENSORBOARD_DIR_NAME = 'tensorboard'


def create_dir(directory):
    try:
        os.makedirs(os.path.dirname(directory))
    except OSError as exc:  # Guard against race condition
        if exc.errno != errno.EEXIST:
            raise


def train(environment, algorithm, timesteps):
    from envs import cpa

    from stable_baselines.common.vec_env import DummyVecEnv
    from stable_baselines.bench import Monitor
    from stable_baselines import PPO2, ACKTR, DQN, A2C

    now = datetime.now()
    current_time = now.strftime("%Y-%m-%d-%H-%M-%S")

    training_info_dir = "training_info" + os.path.sep
    current_training_info = "{}-{}-{}".format(current_time, algorithm, environment)
    current_training_info_dir = training_info_dir + current_training_info + os.path.sep

    model_file_path = current_training_info_dir + "model"
    log_file_path = current_training_info_dir + "monitor.csv"

    tensorboard_dir = training_info_dir + TENSORBOARD_DIR_NAME + os.path.sep

    dirs_to_create = [model_file_path, tensorboard_dir, model_file_path]

    for directory in dirs_to_create:
        create_dir(directory)

    env = None

    if environment == 'cpa_sparse':
        env = cpa.CPAEnvSparse()
    elif environment == 'cpa_dense':
        env = cpa.CPAEnvDense()
    else:
        raise Exception("Environment '{}' is unknown.".format(environment))

    # Optional: PPO2 requires a vectorized environment to run
    # the env is now wrapped automatically when passing it to the constructor
    env = Monitor(env, filename=log_file_path, allow_early_resets=True)
    env = DummyVecEnv([lambda: env])

    model = None

    if algorithm == 'acktr':
        model = ACKTR('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
    elif algorithm == 'ppo':
        model = PPO2('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
    elif algorithm == 'a2c':
        model = A2C('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
    elif algorithm == 'dqn':
        model = DQN('MlpPolicy', env, verbose=1, tensorboard_log=tensorboard_dir)
    else:
        raise Exception("Algorithm '{}' is unknown.".format(algorithm))

    # Train the agent
    model.learn(total_timesteps=timesteps, tb_log_name=current_training_info)

    model.save(model_file_path)

    print("Finished training model: {}. Saved training info in: {}".format(model, current_training_info_dir))

    # # Test the trained agent
    # obs = env.reset()
    # n_steps = 20
    # for step in range(n_steps):
    #     action, _ = model.predict(obs, deterministic=True)
    #     obs, reward, done, info = env.step(action)
    #     env.render(mode='console')
    #     if done:
    #         # Note that the VecEnv resets automatically
    #         # when a done signal is encountered
    #         print("Goal reached!", "reward=", reward)
    #         break


def check_arguments(args):
    if args.environment not in AVAILABLE_ENVIRONMENTS:
        raise argparse.ArgumentError(
            "Environment '{}' does not belong to available environments ({}).".format(args.environment,
                                                                                      AVAILABLE_ENVIRONMENTS))

    if args.algorithm not in AVAILABLE_ALGORITHMS:
        raise argparse.ArgumentError(
            "Algorithm '{}' does not belong to available algorithms ({}).".format(args.algorithm, AVAILABLE_ALGORITHMS))

    if (args.timesteps <= 0) or (not isinstance(args.timesteps, int)):
        raise argparse.ArgumentTypeError("Number of timesteps must be a positive integer and different than zero.")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train a DRL model in a CPA env.')
    parser.add_argument('environment',
                        help='The environment to run. One of: '.format(AVAILABLE_ENVIRONMENTS))
    parser.add_argument('algorithm', help='The DRL algorithm. One of: '.format(AVAILABLE_ALGORITHMS))
    parser.add_argument('--timesteps', type=int, default=DEFAULT_TIMESTEPS,
                        help='Number of training timesteps (default: {})'.format(DEFAULT_TIMESTEPS))

    args = parser.parse_args()

    check_arguments(args)

    train(args.environment, args.algorithm, args.timesteps)
