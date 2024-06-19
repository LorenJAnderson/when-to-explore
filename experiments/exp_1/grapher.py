import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def exp_1_grapher() -> None:
    """Graphs the concentration of terminal states in exploration mode for
    monolithic and switching policies."""
    data_path = 'exp_1_data.p'
    data = pickle.load(open(data_path, 'rb'))
    mono = data['monolithic_terms']
    switch = data['switching_terms']
    sns.set_style("darkgrid")
    sns.set_palette("Paired")
    sns.set(font_scale=1.2)
    sns.lineplot(np.mean(switch, axis=1), label='switching',
                 color=(0.1, 0.5, 0.9), linewidth=2.5)
    sns.lineplot(np.mean(mono, axis=1), label='monolithic',
                 color=(0.5, 0.9, 0.7), linewidth=2.5)
    plt.xlabel('Frame')
    plt.ylabel('Weighted Exploration Score')
    plt.title('Concentration of Terminal States During Exploration')
    plt.legend(title='Behavior Policy')
    plt.xticks([-1, 4, 9, 14, 19, 24],
               ['0e6', '2e6', '4e6', '6e6', '8e6', '10e6'])
    plt.show()


if __name__ == "__main__":
    exp_1_grapher()
