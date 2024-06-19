import pickle

import numpy as np

GRID_LENGTH = 22
GRID_WIDTH = 11


class EpisodePlayer:
    """Generates data from a gridworld environment and policy that learns
    through Q-learning."""
    def __init__(self, policy: np.ndarray) -> None:
        self.policy = policy

        self.state_visitations = None
        self.agent_loc = None
        self.explore_steps_left = None

    def one_episode(self) -> np.ndarray:
        """Updates policy based on one episode in gridworld environment."""
        self.state_visitations = np.zeros((GRID_WIDTH, GRID_LENGTH))
        self.agent_loc = (GRID_WIDTH // 2, GRID_LENGTH // 2)
        self.explore_steps_left = 0
        done = False
        while not done:
            old_x, old_y = self.agent_loc
            self.state_visitations[old_x, old_y] += 1
            action = self.get_action()
            x, y = self.action_decoder(action)
            delta_x, delta_y = self.agent_loc
            new_x, new_y = x + delta_x, y + delta_y
            done = ((new_x < 0 or new_x > (GRID_WIDTH - 1))
                    or (new_y < 0 or new_y > (GRID_LENGTH - 1)))
            self.agent_loc = (new_x, new_y)
            reward = -1
            next_reward = 0 if done else np.max([self.policy[new_x, new_y, act]
                                                for act in range(4)])
            self.policy[old_x, old_y, action] += 0.1 * (next_reward + (
                    reward - self.policy[old_x, old_y, action]))
        return self.policy

    def get_action(self) -> int:
        """Returns policy action and updates the number of remaining
        exploration steps remaining if in switching exploration mode."""
        if np.random.uniform(0, 1) < 0.1:
            action = np.random.randint(4)
        else:
            action = self.exploit_action()
        return action

    def exploit_action(self) -> int:
        """Exploitation action of current policy."""
        x, y = self.agent_loc
        best_score = self.policy[x, y, 0]
        best_action = 0
        for act in range(4):
            score = self.policy[x, y, act]
            if score > best_score:
                best_score = score
                best_action = act
        return best_action

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


def exp_4_illustrative_analyzer() -> None:
    """Generates data from gridworld environment in which agent's sole
    exploitation action is to move off the grid. Agents learns through
    tabular Q-learning."""
    NUM_EPISODES = 100_000

    policy = np.random.normal(0, 0.1, (11, 22, 4))
    episode_player = EpisodePlayer(policy)
    for _ in range(NUM_EPISODES):
        policy = episode_player.one_episode()
    pickle.dump(policy, open('exp_4_illustration_data.p', 'wb'))


if __name__ == "__main__":
    exp_4_illustrative_analyzer()
