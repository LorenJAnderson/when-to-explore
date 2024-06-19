import pickle

import matplotlib.pyplot as plt
import seaborn as sns


def exp_5_illustrative_grapher() -> None:
    """Graphs percentage of exploit actions per episode for switching and
    monolithic policies."""
    data_path = 'exp_5_illustration_data.p'
    data = pickle.load(open(data_path, 'rb'))
    mono = data['monolithic']
    switch = data['switching']
    mono.sort(reverse=True)
    switch.sort(reverse=True)
    sns.set_style("darkgrid")
    sns.set_palette("Paired")
    sns.set(font_scale=1.2)

    sns.lineplot(switch, label='switching',
                 color=(0.1, 0.5, 0.9), linewidth=2.5)
    sns.lineplot(mono, label='monolithic',
                 color=(0.5, 0.9, 0.7), linewidth=2.5)
    plt.xlabel('Frame')
    plt.legend(title='Behavior Policy')
    plt.xlabel('Rank')
    plt.ylabel('Exploit Proportion')
    plt.title('Ranked Exploit Proportion per Episode')
    plt.xticks([-1, 199, 399, 599, 799, 999],
               ['0', '200', '400', '600', '800', '1000'])
    plt.show()


if __name__ == "__main__":
    exp_5_illustrative_grapher()
