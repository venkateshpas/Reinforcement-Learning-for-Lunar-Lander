import matplotlib.pyplot as plt
import numpy as np
import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import torch.distributions as distributions
import matplotlib.pyplot as plt
from functions import *

#Defining the environments: Test and train are required
train_env = gym.make('LunarLander-v2')

#Defining rest of the variables required
input_layer_dimension = train_env.observation_space.shape[0] #Should be 8: (x, y, vx, vy, theta, omega, left leg, right leg)
output_layer_dimension = train_env.action_space.n #Should be the actions probabilities (after softmax) so: [nothing, left engine, main engine, right engine]
train_epochs = 200 #No. of episodes
gamma = 0.99
ppo_steps_parameter = 10
epsilon = 0.25
test_epochs = 10

gamma_list = [0.99,0.98]#,0.97,0.96,0.95, 0.94, 0.93, 0.92, 0.91, 0.9] #Keep the length even always as plots code is written considering it to be even
ppo_steps_parameter_list = [10,9,8,7,6,5]
epsilon_list = [0.25, 0.24, 0.23, 0.22, 0.21, 0.20, 0.19, 0.18, 0.17, 0.16, 0.15]


all_rewards_loop = dict()
mean_rewards_loop = dict()
episode_list_loop = dict()

for gamma in gamma_list:
    episode_list, all_rewards, loss_history_policy, loss_history_value, mean_rewards, mean_episodes = training_cycle(input_layer_dimension, output_layer_dimension, train_epochs, gamma, ppo_steps_parameter, epsilon, test_epochs, train_env)
    #all_rewards_loop[gamma] = all_rewards
    mean_rewards_loop[gamma] = mean_rewards
    episode_list_loop[gamma] = mean_episodes

varied = "Gamma"
plots_for_multiple_training_cycle(gamma_list, episode_list_loop, mean_rewards_loop, varied)

all_rewards_loop = dict()
mean_rewards_loop = dict()
episode_list_loop = dict()

for ppo_steps_parameter in ppo_steps_parameter_list:
    episode_list, all_rewards, loss_history_policy, loss_history_value, mean_rewards, mean_episodes = training_cycle(input_layer_dimension, output_layer_dimension, train_epochs, gamma, ppo_steps_parameter, epsilon, test_epochs, train_env)
    #all_rewards_loop[gamma] = all_rewards
    mean_rewards_loop[gamma] = mean_rewards
    episode_list_loop[gamma] = mean_episodes

varied = "PPO Parameters"
plots_for_multiple_training_cycle(gamma_list, episode_list_loop, mean_rewards_loop, varied)

all_rewards_loop = dict()
mean_rewards_loop = dict()
episode_list_loop = dict()

for epsilon in epsilon_list:
    episode_list, all_rewards, loss_history_policy, loss_history_value, mean_rewards, mean_episodes = training_cycle(input_layer_dimension, output_layer_dimension, train_epochs, gamma, ppo_steps_parameter, epsilon, test_epochs, train_env)
    #all_rewards_loop[gamma] = all_rewards
    mean_rewards_loop[gamma] = mean_rewards
    episode_list_loop[gamma] = mean_episodes

varied = "Epsilon"
plots_for_multiple_training_cycle(gamma_list, episode_list_loop, mean_rewards_loop, varied)