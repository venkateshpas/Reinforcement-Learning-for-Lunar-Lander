import gymnasium as gym
from stable_baselines3.common.evaluation import evaluate_policy 
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor

log_dir = "ppo_lunarlander_tensorboard/"
timesteps = 1e6
env = gym.make('LunarLander-v2', render_mode = 'human')
env = Monitor(env,log_dir)
model = PPO("MlpPolicy", env, verbose = 1, tensorboard_log=log_dir)
model.learn(total_timesteps=timesteps,progress_bar=True)
model.save('ppo_LunarLander_v2')
del model

model = PPO.load('ppo_LunarLander_v2', env)
mean_reward, std_reward = evaluate_policy(model, model.get_env(),n_eval_episodes=10)
vec_env = model.get_env()
obs = vec_env.reset()
ep_reward = 0
for i in range(1e3):
    action, _states = model.predict(obs, deterministic = True)
    obs, reward, done, info = vec_env.step(action)
    ep_reward += reward
    vec_env.render("human")
    if done:
        print(reward)
        ep_reward = 0