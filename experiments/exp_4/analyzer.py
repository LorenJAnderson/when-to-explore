import pickle

import numpy as np

import common.utils as utils


NUM_EPOCHS = 25
NUM_GAMES = 10
MAX_ACTIONS = 18


def exp_4_analyzer() -> None:
    """Determines entropy of empirical action distribution of exploit
    actions that are taken within 5 steps after an exploration action.
    Entropies are conditioned on the epoch number and averaged across games."""
    monolithic_explore = np.zeros(NUM_EPOCHS)
    switching_explore = np.zeros(NUM_EPOCHS)

    for _, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for iter_idx, step_number in enumerate(utils.EPOCH_NUMBERS):
                model_path = ('../../data/' + game_name + '/scores/'
                              + explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(model_path, 'rb'))
                experiment_modes = data['modes']
                experiment_actions = data['actions']
                actions = np.zeros(MAX_ACTIONS)
                for episode_data in zip(experiment_modes, experiment_actions):
                    episode_modes, episode_actions = episode_data
                    explore_countdown = 0
                    for mode_idx, mode in enumerate(episode_modes):
                        if mode == 'explore':
                            explore_countdown = 5
                        elif mode == 'exploit' and explore_countdown > 0:
                            actions[episode_actions[mode_idx]] += 1
                            explore_countdown -= 1
                        else:
                            pass
                action_probs = actions / np.sum(actions)
                total_entropy = -1 * np.dot(action_probs, np.log(
                    action_probs, where=actions != 0))
                if explore_strat == 'monolithic':
                    monolithic_explore[iter_idx] += (total_entropy / NUM_GAMES)
                else:
                    switching_explore[iter_idx] += (total_entropy / NUM_GAMES)

    final_data = {'monolithic_entropy': monolithic_explore,
                  'switching_entropy': switching_explore}
    pickle.dump(final_data, open('exp_4_data.p', 'wb'))


if __name__ == "__main__":
    exp_4_analyzer()
