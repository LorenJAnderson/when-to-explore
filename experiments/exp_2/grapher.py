import pickle

import matplotlib.pyplot as plt
import seaborn as sns


def exp_2_grapher() -> None:
    """Graphs average number of time steps to reach y steps
    in exploration mode the fastest for y in {1, 2, ..., 20} across all
    games at last iteration."""
    data_path = 'exp_2_data.p'
    data = pickle.load(open(data_path, 'rb'))
    mono = data['monolithic_explore']
    switch = data['switching_explore']
    sns.set_style("darkgrid")
    sns.set_palette("Paired")
    sns.set(font_scale=1.2)
    sns.lineplot(switch, label='switching',
                 color=(0.1, 0.5, 0.9), linewidth=2.5)
    sns.lineplot(mono, label='monolithic',
                 color=(0.5, 0.9, 0.7), linewidth=2.5)
    plt.legend(title='Behavior Policy')
    plt.xlabel('Number of Exploration Actions')
    plt.ylabel('Time Step')
    plt.title('Average Time to Take Early Exploration Actions')
    plt.xticks([-1, 3, 7, 11, 15, 19],
               ['0', '4', '8', '12', '16', '20'])
    plt.show()


if __name__ == "__main__":
    exp_2_grapher()
