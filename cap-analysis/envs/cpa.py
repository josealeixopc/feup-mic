import gym
from gym import spaces


class CPAEnv(gym.Env):
    """Custom Environment that follows gym interface.

    In this CPA environment, the observation is a single number.

    The agent must determine whether the number is 0, odd or even.

    Every time it gets it right, it gets a positive reward.

    After getting it right a number of times, the episode ends.
    """
    metadata = {'render.modes': ['console']}

    N_DISCRETE_ACTIONS = 3
    EVEN = 0
    ODD = 1

    N_DISCRETE_OBS = 10

    NEEDED_CORRECT_ANSWERS = 100
    MAX_NUM_STEPS = 3*NEEDED_CORRECT_ANSWERS

    def __init__(self):
        super(CPAEnv, self).__init__()
        # Define action and observation space
        # They must be gym.spaces objects
        # Example when using discrete actions:
        self.action_space = spaces.Discrete(self.N_DISCRETE_ACTIONS)
        self.observation_space = spaces.Discrete(self.N_DISCRETE_OBS)

        self.current_score = None
        self.current_observation = None
        self.last_observation = None

        self.last_action = None
        self.last_reward = None

        self.current_step_num = None

    def step(self, action):
        pass

    def reset(self):
        # Reset score and round
        self.current_score = 0
        self.current_step_num = 0

        # Current number if randomly decided from the observation space
        self.current_observation = self.observation_space.sample()
        observation = self.current_observation

        return observation  # reward, done, info can't be included

    def render(self, mode='console'):
        # Don't print in first round. No need for that.
        if self.current_step_num != 0:
            print("ROUND {}".format(self.current_step_num - 1))
            print("Observation: {}".format(self.last_observation))
            print("Action: {}".format(self.last_action))
            print("Score: {}".format(self.current_score))
            print("--------")

    def close(self):
        pass


class CPAEnvDense(CPAEnv):
    def step(self, action):
        self.current_step_num += 1

        reward = 0
        done = False
        info = {}

        if (self.current_observation % 2 == 0 and action == self.EVEN) or (
                self.current_observation % 2 == 1 and action == self.ODD):
            reward = 1
            self.current_score += 1

        if self.current_score >= self.NEEDED_CORRECT_ANSWERS or self.current_step_num >= self.MAX_NUM_STEPS:
            done = True

        # Save previous observation and generate new one
        self.last_observation = self.current_observation
        self.current_observation = self.observation_space.sample()
        observation = self.current_observation

        self.last_action = action
        self.last_reward = reward

        return observation, reward, done, info


class CPAEnvSparse(CPAEnv):
    def step(self, action):
        self.current_step_num += 1

        reward = 0
        done = False
        info = {}

        if (self.current_observation % 2 == 0 and action == self.EVEN) or (
                self.current_observation % 2 == 1 and action == self.ODD):
            self.current_score += 1

        if self.current_score >= self.NEEDED_CORRECT_ANSWERS:
            reward = self.NEEDED_CORRECT_ANSWERS
            done = True

        if self.current_step_num >= self.MAX_NUM_STEPS:
            done = True

        # Save previous observation and generate new one
        self.last_observation = self.current_observation
        self.current_observation = self.observation_space.sample()
        observation = self.current_observation

        self.last_action = action
        self.last_reward = reward

        return observation, reward, done, info
