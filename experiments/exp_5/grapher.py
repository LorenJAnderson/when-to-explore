import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


NUM_EPOCHS = 25
NUM_GAMES = 10


def exp_5_grapher() -> None:
    """Plots ratio of the average top returns across all episodes of each
    game and epoch number for switching versus monolithic policies."""
    data_path = 'exp_5_data.p'
    data = pickle.load(open(data_path, 'rb'))
    mono = data['monolithic']
    switch = data['switching']
    sns.set(font_scale=1.2)

    iter_ratios = []
    for iter_idx in range(NUM_EPOCHS):
        ratios = []
        for game_idx in range(NUM_GAMES):
            ratios.append(switch[iter_idx, game_idx] /
                          mono[iter_idx, game_idx])
        iter_ratios.append(np.mean(ratios))
    sns.lineplot(iter_ratios, color=(0.1, 0.5, 0.9), linewidth=2.5)
    plt.xlabel('Frame')
    plt.ylabel('Return Ratio')
    plt.title('Average Return Ratio of Top Exploitation Episodes')
    plt.xticks([-1, 4, 9, 14, 19, 24],
               ['0e6', '2e6', '4e6', '6e6', '8e6', '10e6'])
    plt.show()


if __name__ == "__main__":
    exp_5_grapher()
