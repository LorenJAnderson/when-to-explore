import pickle

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np


def exp_1_illustrative_grapher() -> None:
    """Displays cliffwalk environment results in grid form for switching and
    monolithic behavior policies."""
    data_path = 'exp_1_illustration_data.pkl'
    data = pickle.load(open(data_path, 'rb'))
    switch_grid, mono_grid = data[0], data[1]
    cmap = sns.color_palette("mako", as_cmap=True)
    sns.set(font_scale=1.2)

    plt.figure(figsize=(4, 6))
    sns.heatmap(np.transpose(switch_grid), cmap=cmap,
                cbar_kws={'label': 'Visitation Frequency'})
    plt.scatter(5.5, 1, marker="o", s=25, c='black')
    plt.scatter(5.5, 100, marker="x", s=25, c='white')
    plt.title('Switching Cliffwalk')
    plt.show()

    plt.figure(figsize=(4, 6))
    sns.heatmap(np.transpose(mono_grid), cmap=cmap,
                cbar_kws={'label': 'Visitation Frequency'})
    plt.scatter(5.5, 1, marker="o", s=25, c='black')
    plt.scatter(5.5, 100, marker="x", s=25, c='white')
    plt.title('Monolithic Cliffwalk')
    plt.show()


if __name__ == "__main__":
    exp_1_illustrative_grapher()
