import gymnasium as gym
from gymnasium.utils.play import play,PlayPlot 

env = gym.make("LunarLander-v2", render_mode = 'rgb_array')

def callback(obs_t, obs_tpl, action, rew, terminated, truncated, info):

    return [rew,]

plotter = PlayPlot(callback, 150, ['reward'])

play(env, keys_to_action = {
    'w': 2,
    'a': 1,
    'd': 3,
}, noop = 0, callback=plotter.callback)
