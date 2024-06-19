import pickle

import numpy as np

import common.utils as utils

BINS = 20
TOTAL_STEPS = 2_500_000
NUM_EPISODES = 100
NUM_GAMES = 10


def exp_3_illustrative_analyzer() -> None:
    """Generates the distribution of return proportion throughout the
    episode. The return proportion is binned across the percent of the
    episode that is complete. Results use the final trained policy and are
    averaged over all games."""
    monolithic_explore = np.zeros(BINS)
    switching_explore = np.zeros(BINS)

    for game_idx, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for step_number in [TOTAL_STEPS]:
                data_path = ('../../data/' + game_name + '/scores/'
                             + explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(data_path, 'rb'))
                episode_modes = data['modes']
                episode_rewards = data['rewards']
                non_zero_episodes = 0
                game_bins = np.zeros(BINS)
                for episode_stats in zip(episode_modes, episode_rewards):
                    episode_modes, episode_rewards = episode_stats
                    episode_bins = np.zeros(BINS)
                    for step_idx, _ in enumerate(episode_rewards):
                        bin_index = int(step_idx / len(episode_rewards) * BINS)
                        episode_bins[bin_index] += episode_rewards[step_idx]
                    if sum(episode_rewards) != 0:
                        episode_bins /= sum(episode_rewards)
                        non_zero_episodes += 1
                    game_bins += episode_bins
                if explore_strat == 'monolithic':
                    monolithic_explore += (game_bins /
                                           (NUM_GAMES * non_zero_episodes))
                else:
                    switching_explore += (game_bins /
                                          (NUM_GAMES * non_zero_episodes))

    final_data = {'monolithic_exploit': monolithic_explore,
                  'switching_exploit': switching_explore}
    pickle.dump(final_data, open('exp_3_illustration_data.p', 'wb'))


if __name__ == "__main__":
    exp_3_illustrative_analyzer()
