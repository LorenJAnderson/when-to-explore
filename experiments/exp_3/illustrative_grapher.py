import pickle

import matplotlib.pyplot as plt
import seaborn as sns


def exp_3_illustrative_grapher() -> None:
    """Displays the distribution of return proportion throughout the
    episode."""
    data_path = 'exp_3_illustration_data.p'
    data = pickle.load(open(data_path, 'rb'))
    switch = data['switching_exploit']
    sns.set_style("darkgrid")
    sns.set(font_scale=1.2)

    sns.lineplot(switch, color=(0.1, 0.5, 0.9), linewidth=2.5)
    plt.xlabel('Episode Proportion Complete')
    plt.ylabel('Return Proportion')
    plt.title('Return Concentration Throughout Episode')
    plt.xticks([-1, 3, 7, 11, 15, 19],
               ['0.00', '0.20', '0.40', '0.60', '0.80', '1.00'])
    plt.show()


if __name__ == "__main__":
    exp_3_illustrative_grapher()
