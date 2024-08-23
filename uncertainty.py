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
import pandas

#Part 1: Uncertainity after the training

#Defining the environments: Test and train are required
train_env = gym.make('LunarLander-v2')

#Defining rest of the variables required
input_layer_dimension = train_env.observation_space.shape[0] #Should be 8: (x, y, vx, vy, theta, omega, left leg, right leg)
output_layer_dimension = train_env.action_space.n #Should be the actions probabilities (after softmax) so: [nothing, left engine, main engine, right engine]
test_epochs = 1000

episode_reward_loop, episode_loop = uncertainty_test(input_layer_dimension, output_layer_dimension, test_epochs)
mean, lower_limit, upper_limit = confidence_intervals(episode_reward_loop['Episode Reward'],.95)
rate = episode_reward_loop.query('`Episode Reward` >= 200').size
print(f'The rate of success is {rate} out of {test_epochs}')
print(f'The mean reward during testing is {mean}, the lower limit and upper limit for 95% confidence are {lower_limit}, {upper_limit}')

#Part 2: Uncertainity within the training process
# train_epochs = 10000
# gamma = 0.99
# ppo_steps_parameter = 7
# epsilon = 0.25
# trials = 100
# lr = 0.0001

# episodes_taken = uncertainty_train(trials, input_layer_dimension, output_layer_dimension, train_epochs, gamma, ppo_steps_parameter, epsilon, test_epochs, train_env, lr)

# mean, lower_limit, upper_limit = confidence_intervals(episodes_taken['Episodes'],.95)
# print(f'The mean episodes taken for training is {mean}, the lower limit and upper limit for 95% confidence are {lower_limit}, {upper_limit}')