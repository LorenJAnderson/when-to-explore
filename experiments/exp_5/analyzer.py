import pickle

import numpy as np

import common.utils as utils

NUM_EPOCHS = 25
TOP_EPISODES = 10
NUM_GAMES = 10


def exp_5_analyzer() -> None:
    """Determines the top returns across all episodes of each game and
    epoch number."""
    monolithic_explore = np.zeros((NUM_EPOCHS, NUM_GAMES))
    switching_explore = np.zeros((NUM_EPOCHS, NUM_GAMES))

    for game_idx, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for iter_idx, step_number in enumerate(utils.EPOCH_NUMBERS):
                model_path = ('../../data/' + game_name + '/scores/' +
                              explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(model_path, 'rb'))
                exploit_reward_list = []
                for episode_data in zip(data['modes'], data['actions']):
                    episode_modes, episode_rewards = episode_data
                    exploit_percent = (episode_modes.count('exploit') /
                                       len(episode_modes))
                    exploit_reward_list.append((exploit_percent,
                                                sum(episode_rewards)))
                exploit_reward_list.sort(reverse=True)
                top_epi_sum = 0
                for _, episode_rewards in exploit_reward_list[:10]:
                    top_epi_sum += episode_rewards
                if explore_strat == 'monolithic':
                    monolithic_explore[iter_idx, game_idx] += (top_epi_sum /
                                                               TOP_EPISODES)
                else:
                    switching_explore[iter_idx, game_idx] += (top_epi_sum /
                                                              TOP_EPISODES)
    final_data = {'monolithic': monolithic_explore,
                  'switching': switching_explore}
    pickle.dump(final_data, open('exp_5_data.p', 'wb'))


if __name__ == "__main__":
    exp_5_analyzer()
