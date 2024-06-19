import pickle

import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns


def exp_2_illustrative_grapher() -> None:
    """Displays downwalk environment results in grid form for switching and
    monolithic behavior policies."""
    data_path = 'exp_2_illustration_data.pkl'
    data = pickle.load(open(data_path, 'rb'))
    switch_grid, mono_grid = data[0], data[1]
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.set(font_scale=1.2)

    plt.figure(figsize=(4, 6))
    sns.heatmap(np.transpose(switch_grid), cmap=cmap,
                cbar_kws={'label': 'Visitation Frequency'})
    plt.scatter(5.5, 2.5, marker="o", s=25, c='white')
    plt.scatter(5.5, 21.5, marker="x", s=25, c='white')
    plt.title('Switching Downwalk')
    plt.yticks([0, 5, 10, 15, 21],
               ['0', '5', '10', '15', '21'])
    plt.show()

    plt.figure(figsize=(4, 6))
    sns.heatmap(np.transpose(mono_grid), cmap=cmap,
               cbar_kws={'label': 'Visitation Frequency'})
    plt.scatter(5.5, 2.5, marker="o", s=25, c='white')
    plt.scatter(5.5, 21.5, marker="x", s=25, c='white')
    plt.title('Monolithic Downwalk')
    plt.yticks([0, 5, 10, 15, 21],
               ['0', '5', '10', '15', '21'])
    plt.show()


if __name__ == "__main__":
    exp_2_illustrative_grapher()
