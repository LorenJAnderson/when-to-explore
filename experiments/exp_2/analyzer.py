import pickle

import numpy as np

import common.utils as utils

TOTAL_STEPS = 2_500_000
INITIAL_STEPS = 20
TOP_THRES = 5
NUM_EPISODES = 100
NUM_GAMES = 10
LAST_EXPLORE = 20


def exp_2_analyzer() -> None:
    """Determines number of steps taken to have x steps in exploration mode
    for x in {1, 2, ..., 20}. Then determines episodes that reached y steps
    in exploration mode the fastest for y in {1, 2, ..., 20} across all
    episodes for a single game. Due to blind switching implemented in the
    paper, only the final iteration is used to generate data."""
    monolithic_counts = np.zeros(INITIAL_STEPS)
    switching_counts = np.zeros(INITIAL_STEPS)

    for game_idx, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for iter_idx, step_number in enumerate([TOTAL_STEPS]):
                data_path = ('../../data/' + game_name + '/scores/' +
                             explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(data_path, 'rb'))

                experiment_steps = np.zeros((LAST_EXPLORE, NUM_EPISODES))
                for epi_idx, episode in enumerate(data['modes']):
                    explore_count = 0
                    for step_idx, mode in enumerate(episode):
                        if mode == 'explore' and explore_count < LAST_EXPLORE:
                            experiment_steps[explore_count, epi_idx] = step_idx
                            explore_count += 1

                for step_idx in range(INITIAL_STEPS):
                    top_list = []
                    for epi_idx in range(NUM_EPISODES):
                        top_list.append(experiment_steps[step_idx, epi_idx])
                    top_list.sort()
                    if explore_strat == 'monolithic':
                        monolithic_counts[step_idx] += \
                            (sum(top_list[0:TOP_THRES]) /
                             (NUM_GAMES * TOP_THRES))
                    else:
                        switching_counts[step_idx] += \
                            (sum(top_list[0:TOP_THRES]) /
                             (NUM_GAMES * TOP_THRES))

    final_data = {'monolithic_explore': monolithic_counts,
                  'switching_explore': switching_counts}
    pickle.dump(final_data, open('exp_2_data.p', 'wb'))


if __name__ == "__main__":
    exp_2_analyzer()
