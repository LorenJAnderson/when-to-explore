import pickle
from collections import deque

import numpy as np

import common.utils as utils


NUM_EPOCHS = 25
NUM_GAMES = 10
EXPLOIT_WINDOW = 10
NUM_EPISODES = 100


def exp_3_analyzer() -> None:
    """Quantifies the extent to which score was influenced by taking actions
    in exploit mode for each game and each epoch. A score for each time step
    is calculated by determining the reward accrued at that time step and
    then multiplying by the proportion of exploit actions taken in a window
    of previous steps. The score for an episode is the average score across
    time steps, and the score for an epoch and game is the average score
    across episodes."""
    monolithic_exploit = np.zeros((NUM_EPOCHS, NUM_GAMES))
    switching_exploit = np.zeros((NUM_EPOCHS, NUM_GAMES))

    for game_idx, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for iter_idx, step_number in enumerate(utils.EPOCH_NUMBERS):
                data_path = ('../../data/' + game_name + '/scores/'
                             + explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(data_path, 'rb'))
                experiment_modes = data['modes']
                experiment_rewards = data['rewards']
                experiment_score = 0
                for episode_stats in zip(experiment_modes, experiment_rewards):
                    episode_modes, episode_rewards = episode_stats
                    episode_score = 0
                    last_modes = deque([], maxlen=EXPLOIT_WINDOW)
                    for mode_idx in range(len(episode_modes)):
                        last_modes.append(episode_modes[mode_idx])
                        all_exploit = 0
                        for mode in last_modes:
                            if mode == 'exploit':
                                all_exploit += 1
                        episode_score += (episode_rewards[mode_idx] *
                                         (all_exploit / len(last_modes)))
                    if sum(episode_rewards) == 0:
                        episode_score = 0
                    else:
                        episode_score = episode_score / sum(episode_rewards)
                    experiment_score += episode_score / NUM_EPISODES
                if explore_strat == 'monolithic':
                    monolithic_exploit[iter_idx, game_idx] += experiment_score
                else:
                    switching_exploit[iter_idx, game_idx] += experiment_score

    final_data = {'monolithic_exploit': monolithic_exploit,
                  'switching_exploit': switching_exploit}
    pickle.dump(final_data, open('exp_3_data.p', 'wb'))


if __name__ == "__main__":
    exp_3_analyzer()
