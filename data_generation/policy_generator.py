import common.utils as utils

from stable_baselines3 import DQN
from stable_baselines3.common.callbacks import CheckpointCallback


def generate_one_game_policies(game_name: str = 'Breakout') -> None:
    """Generates the policies for one game."""
    env = utils.env_creator(game_name)
    checkpoint_callback = CheckpointCallback(
      save_freq=100_000,
      save_path="../data/"+game_name+"/runs",
      name_prefix=game_name,
    )
    model = DQN(policy="CnnPolicy",
                env=env,
                learning_starts=100_000,
                buffer_size=100_000,
                exploration_final_eps=0.1,
                verbose=True)
    model.learn(total_timesteps=2_500_000,
                callback=checkpoint_callback)


if __name__ == '__main__':
    generate_one_game_policies()
