import pickle
from collections import deque

import numpy as np

import common.utils as utils

NUM_GAMES = 10
NUM_EPOCHS = 25
NUM_EPISODES = 100
TERM_WINDOW = 10


def exp_1_analyzer() -> None:
    """Quantifies the extent to which terminal states occurred as a
    result of taking actions in explore mode for each game and each
    epoch. A score is calculated by determining the number of actions taken
    in explore mode for each episode and dividing by the length of the
    terminal window and number of episodes."""
    monolithic_terms = np.zeros((NUM_EPOCHS, NUM_GAMES))
    switching_terms = np.zeros((NUM_EPOCHS, NUM_GAMES))

    for game_idx, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for iter_idx, step_number in enumerate(utils.EPOCH_NUMBERS):
                data_path = ('../../data/' + game_name + '/scores/' +
                             explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(data_path, 'rb'))
                experiment_modes = data['modes']
                for _, episode_modes in enumerate(experiment_modes):
                    all_explore = 0
                    last_modes = deque([], maxlen=TERM_WINDOW)
                    for mode in episode_modes:
                        last_modes.append(mode)
                    for mode in last_modes:
                        if mode == 'explore':
                            all_explore += 1
                    if explore_strat == 'monolithic':
                        monolithic_terms[iter_idx, game_idx] += \
                            all_explore / (TERM_WINDOW * NUM_EPISODES)
                    else:
                        switching_terms[iter_idx, game_idx] += \
                            all_explore / (TERM_WINDOW * NUM_EPISODES)

    final_data = {'monolithic_terms': monolithic_terms,
                  'switching_terms': switching_terms}
    pickle.dump(final_data, open('exp_1_data.p', 'wb'))


if __name__ == "__main__":
    exp_1_analyzer()
