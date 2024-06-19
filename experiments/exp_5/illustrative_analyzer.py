import pickle

import common.utils as utils


TOTAL_STEPS = 2_500_000


def exp_5_illustrative_analyzer() -> None:
    """Determines percentage of exploit actions taken per episode
    conditioned on game and final trained policy for switching and
    monolithic policies."""
    monolithic_exploit = []
    switching_exploit = []

    for game_idx, game_name in enumerate(utils.GAME_NAMES):
        for explore_strat in utils.EXPLORE_STRATS:
            for _, step_number in enumerate([TOTAL_STEPS]):
                model_path = ('../../data/' + game_name + '/scores/' +
                              explore_strat + '_' + str(step_number) + '.p')
                data = pickle.load(open(model_path, 'rb'))
                for episode_data in data['modes']:
                    if explore_strat == 'monolithic':
                        monolithic_exploit.append(episode_data.count(
                            'exploit') / len(episode_data))
                    else:
                        switching_exploit.append(episode_data.count(
                            'exploit') / len(episode_data))
    final_data = {'monolithic': monolithic_exploit,
                  'switching': switching_exploit}
    pickle.dump(final_data, open('exp_5_illustration_data.p', 'wb'))


if __name__ == "__main__":
    exp_5_illustrative_analyzer()
