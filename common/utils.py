import gymnasium as gym
import numpy as np

GAME_NAMES = ['Asterix',
              'BeamRider',
              'Bowling',
              'Breakout',
              'Enduro',
              'MsPacman',
              'Qbert',
              'Riverraid',
              'Seaquest',
              'SpaceInvaders']

EXPLORE_STRATS = ['monolithic',
                  'switching']

EPOCH_NUMBERS = [(i + 1) * 100_000 for i in range(25)]


def env_creator(game_name: str = 'Breakout') -> gym.Env:
    """Creates environment and wraps with necessary preprocessing."""
    env = gym.make('ALE/'+game_name+'-v5',
                   frameskip=1,
                   repeat_action_probability=0.25,
                   full_action_space=False,
                   render_mode='rgb_array')
    env = gym.wrappers.AtariPreprocessing(env,
                                          noop_max=0,
                                          frame_skip=4,
                                          screen_size=84,
                                          terminal_on_life_loss=False,
                                          grayscale_obs=True,
                                          grayscale_newaxis=False,
                                          scale_obs=False)
    env = gym.wrappers.FrameStack(env, 4)
    env = gym.wrappers.TransformReward(env, np.sign)
    return env
