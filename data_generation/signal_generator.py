import pickle
import random
from multiprocessing import Pool

import numpy as np
from stable_baselines3 import DQN
import gymnasium as gym


import common.utils as utils

NUM_CPUS = 1
NUM_EVAL_EPISODES = 100


class EpisodePlayer:
    """Generates data from and environment with a trained policy. Data is
    generated for a single episode upon request."""
    def __init__(self, env: gym.Env, model: DQN, explore_strat: str) -> None:
        self.env = env
        self.model = model
        self.explore_strat = explore_strat

        self.explore_steps_left = None
        self.obs = None

    def play_episode(self) -> tuple:
        """Plays an episode and returns list of modes, actions, and rewards
        from the episode."""
        modes, acts, rews = [], [], []
        self.obs, _ = self.env.reset()
        done = False
        self.explore_steps_left = 0
        while not done:
            act, mode, self.explore_steps_left = self.det_action()
            self.obs, reward, terminated, truncated, info = self.env.step(act)
            modes.append(mode)
            acts.append(act)
            rews.append(reward)
            if terminated or truncated:
                return modes, acts, rews

    def det_action(self) -> tuple:
        """Determines action based on mode. Returns action and mode signals.
        Also returns number of consecutive exploration steps remaining."""
        if self.explore_steps_left != 0:
            act = np.random.randint(self.env.action_space.n)
            self.explore_steps_left -= 1
            mode = 'explore'
        elif self.explore_strat == 'switching':
            act, mode, self.explore_steps_left = self.det_switching_act()
        else:
            act, mode, self.explore_steps_left = self.det_monolithic_act()
        return act, mode, self.explore_steps_left

    def det_switching_act(self) -> tuple:
        """Returns action from switching policy."""
        if np.random.uniform(0, 1) < 0.007:
            explore_steps_left = random.choices([5, 10, 15, 20, 25],
                                                [0.05, 0.20, 0.50, 0.20, 0.05],
                                                k=1)[0] - 1
            action = np.random.randint(self.env.action_space.n)
            step_mode = 'explore'
        else:
            action = self.greedy_action()
            step_mode = 'exploit'
            explore_steps_left = 0
        return action, step_mode, explore_steps_left

    def det_monolithic_act(self) -> tuple:
        """Returns action from monolithic policy."""
        if np.random.uniform(0, 1) < 0.10:
            action = np.random.randint(self.env.action_space.n)
            step_mode = 'explore'
        else:
            action = self.greedy_action()
            step_mode = 'exploit'
        return action, step_mode, 0

    def greedy_action(self) -> int:
        """Returns greedy action of trained model."""
        action, _ = self.model.predict(np.array(self.obs),
                                       deterministic=True)
        return action.item()


def generate_one_exp_signals(experiment: tuple) -> None:
    """Generates signals for one experiment. Stores signals in pickle files."""
    game_name, explore_strat, step_number, model_path = experiment
    all_modes, all_acts, all_rews = [], [], []
    env = utils.env_creator(game_name)
    model = DQN.load(model_path)
    episode_player = EpisodePlayer(env, model, explore_strat)
    for i in range(NUM_EVAL_EPISODES):
        modes, acts, rews = episode_player.play_episode()
        all_modes.append(modes)
        all_acts.append(acts)
        all_rews.append(rews)
    signal_dict = {'modes': all_modes,
                   'actions': all_acts,
                   'rewards': all_rews}
    signal_path = ('../data/' + game_name +
                   '/scores/' + explore_strat + '_' +
                   str(step_number) + '.p')
    pickle.dump(signal_dict, open(signal_path, 'wb'))
    print(game_name + ' ' + str(step_number) + ' ' + 'finished')


def generate_all_data() -> None:
    """Generates signals from a specified number of games of running behavior
    policies with monolithic and switching exploration across all games and
    all epochs. Stores signals in pickle files."""
    all_experiments = []
    for game_name in utils.GAME_NAMES:
        for explore_strat in utils.EXPLORE_STRATS:
            for step_number in utils.EPOCH_NUMBERS:
                model_path = ('../data/' + game_name +
                              '/runs/' + game_name + '_' +
                              str(step_number) + '_steps.zip')
                all_experiments.append((game_name,
                                        explore_strat,
                                        step_number,
                                        model_path))
    with Pool(NUM_CPUS) as p:
        p.map(generate_one_exp_signals, all_experiments)


if __name__ == "__main__":
    generate_all_data()
