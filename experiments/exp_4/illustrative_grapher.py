import pickle

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns


def exp_4_illustrative_grapher() -> None:
    """Plots greedy action of learned policy in gridworld environment at
    each state."""
    data_path = 'exp_4_illustration_data.p'
    policy = pickle.load(open(data_path, 'rb'))
    action_grid = np.zeros((11, 22))
    cmap = sns.color_palette("mako", as_cmap=True)
    for x in range(11):
        for y in range(22):
            action_grid[x, y] = sorted([(policy[x, y, act], act) for act in
                                        range(4)], reverse=True)[0][1]
    plt.figure(figsize=(6, 6))
    sns.set(font_scale=1.2)
    sns.heatmap(np.transpose(action_grid), cmap=cmap, cbar=False)
    for x in range(11):
        for y in range(22):
            best_act = sorted([(policy[x, y, act], act) for act in
                               range(4)], reverse=True)[0][1]
            if best_act == 0:
                marker = ">"
            elif best_act == 1:
                marker = "^"
            elif best_act == 2:
                marker = "<"
            else:
                marker = "v"
            plt.scatter(x + 0.5, y + 0.5, marker=marker, s=30,
                        color=(0.1, 0.9, 0.7))
    plt.yticks(list(range(0, 22, 2)),
               [str(x) for x in range(0, 22, 2)])
    plt.title('Trained Q-learning Exploit Actions on Gridworld')
    plt.show()


if __name__ == "__main__":
    exp_4_illustrative_grapher()
