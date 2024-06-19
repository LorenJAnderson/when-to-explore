import random
import pickle

import numpy as np


GRID_LENGTH = 101
GRID_WIDTH = 11


class EpisodePlayer:
    """Generates data from a cliffwalk environment and hard-coded policy.
    Data is generated for a single episode upon request."""
    def __init__(self, explore_strat: str) -> None:
        self.explore_strat = explore_strat

        self.state_visitations = None
        self.agent_loc = None
        self.explore_steps_left = None

    def one_episode(self) -> np.ndarray:
        """Provides state visitations for one episode in cliffwalk
        environment."""
        self.state_visitations = np.zeros((GRID_WIDTH, GRID_LENGTH))
        self.agent_loc = (GRID_WIDTH // 2, 0)
        self.explore_steps_left = 0
        done = False
        while not done:
            old_x, old_y = self.agent_loc
            self.state_visitations[old_x, old_y] += 1
            action = self.get_action()
            x, y = self.action_decoder(action)
            delta_x, delta_y = self.agent_loc
            new_x, new_y = x + delta_x, y + delta_y
            done = ((new_x < (GRID_WIDTH // 2) or new_x > (GRID_WIDTH // 2)) or
                    (new_y < 0 or new_y > GRID_LENGTH - 1))
            self.agent_loc = (new_x, new_y)
        return self.state_visitations

    def get_action(self) -> int:
        """Returns policy action and updates the number of remaining
        exploration steps remaining if in switching exploration mode."""
        if self.explore_steps_left > 0:
            self.explore_steps_left -= 1
            action = np.random.randint(4)
        elif self.explore_strat == 'switching':
            if np.random.uniform(0, 1) < 0.007:
                self.explore_steps_left = random.choices(
                    [5, 10, 15, 20, 25], [0.05, 0.20, 0.50, 0.20, 0.05],
                    k=1)[0] - 1
                action = np.random.randint(4)
            else:
                action = self.exploit_action()
        else:
            if np.random.uniform(0, 1) < 0.1:
                action = np.random.randint(4)
            else:
                action = self.exploit_action()
        return action

    @staticmethod
    def exploit_action() -> int:
        """Hard-coded exploitation action."""
        return 1

    @staticmethod
    def action_decoder(action: int) -> tuple:
        """Provides hard-coded change in movement based on given action."""
        if action == 0:
            return 1, 0
        elif action == 1:
            return 0, 1
        elif action == 2:
            return -1, 0
        else:
            return 0, -1


def exp_1_illustrative_analyzer() -> None:
    """Generates data from cliffwalk environment in which agent's sole
    exploitation action is to move downward. Agent has either switching or
    monolithic behavior policy and takes random exploration actions."""
    NUM_EPISODES = 10_000

    switch_player = EpisodePlayer('switching')
    switch_visitations = np.zeros((GRID_WIDTH, GRID_LENGTH))
    for _ in range(NUM_EPISODES):
        switch_visitations += switch_player.one_episode()
    switch_visitations /= NUM_EPISODES

    mono_player = EpisodePlayer('monolithic')
    mono_visitations = np.zeros((GRID_WIDTH, GRID_LENGTH))
    for _ in range(NUM_EPISODES):
        mono_visitations += mono_player.one_episode()
    mono_visitations /= NUM_EPISODES

    data = [switch_visitations, mono_visitations]
    pickle.dump(data, open('exp_1_illustration_data.pkl', 'wb'))


if __name__ == "__main__":
    exp_1_illustrative_analyzer()
