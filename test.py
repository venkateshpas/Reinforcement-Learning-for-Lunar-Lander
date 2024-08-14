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
train_epochs = 10000 #No. of episodes
gamma = 0.99
ppo_steps_parameter = 10
epsilon = 0.25
test_epochs = 10

actor = Actor(input_layer_dimension, output_layer_dimension)
critic = Critic(input_layer_dimension)

actor.load_state_dict(torch.load('actor_model.pth'))
critic.load_state_dict(torch.load('critic_model.pth'))

# Set the models to evaluation mode (this disables dropout layers, etc.)
actor.eval()
critic.eval()

# Run the agent on the test environment
for epoch in range(1, test_epochs + 1):
    episode_reward = test_loop(actor)
    print(f'Test Episode {epoch}, Total Reward: {episode_reward}')