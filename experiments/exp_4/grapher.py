import pickle

import matplotlib.pyplot as plt
import seaborn as sns


def exp_4_grapher() -> None:
    """Graphs entropy of empirical action distribution of exploit
    actions that are taken within 5 steps after an exploration action."""
    data_path = 'exp_4_data.p'
    data = pickle.load(open(data_path, 'rb'))
    mono = data['monolithic_entropy']
    switch = data['switching_entropy']
    sns.set_style("darkgrid")
    sns.set_palette("Paired")
    sns.set(font_scale=1.2)

    sns.lineplot(switch, label='switching',
                 color=(0.1, 0.5, 0.9), linewidth=2.5)
    sns.lineplot(mono, label='monolithic',
                 color=(0.5, 0.9, 0.7), linewidth=2.5)
    plt.xlabel('Frame')
    plt.ylabel('Entropy')
    plt.legend(title='Behavior Policy')
    plt.xticks([-1, 4, 9, 14, 19, 24],
               ['0e6', '2e6', '4e6', '6e6', '8e6', '10e6'])
    plt.title('Average Action Entropy After Exploration')
    plt.show()


if __name__ == "__main__":
    exp_4_grapher()
