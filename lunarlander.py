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
all_rewards, loss_history_policy, loss_history_value, mean_rewards = [], [], [], []
episode_list = []


#Create the Neural networks and define the optimizers
actor = Actor(input_layer_dimension, output_layer_dimension)
critic = Critic(input_layer_dimension)
optimizer_actor = optim.Adam(actor.parameters(), lr=0.001) #Choose the Adam Optimizer as it is the state of the art even today
optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)


#Training loop
for epoch in range(1, train_epochs + 1):
    states, actions, log_prob_actions, values, rewards = [], [], [], [], []
    done = False
    episode_reward = 0
    state, _ = train_env.reset() #Always reset before starting an episode: and note down the state

    
    states, actions, log_prob_actions, values, rewards, episode_reward = episode(train_env, actor, critic, state, states, actions, log_prob_actions, values, rewards, done, episode_reward)
    policy_loss, value_loss, optimizer_actor, optimizer_critic = ppo_update(actor, critic, optimizer_actor, optimizer_critic, rewards, gamma, ppo_steps_parameter, epsilon, values,states, actions, log_prob_actions)

    # Store and print episode rewards
    all_rewards.append(episode_reward)
    loss_history_policy.append(policy_loss.item())  # Store policy loss
    loss_history_value.append(value_loss.item())  # Store value loss
    episode_list.append(epoch)
    
    #break if we achieve our goal. that is 200 mean reward upon 100 episodes
    if len(all_rewards) >= 100:
        mean_last_100 = sum(all_rewards[-100:]) / 100
        mean_rewards.append(mean_last_100)
        if epoch % 10 == 0:
            print(f'Epoch: {epoch:3}, Reward: {episode_reward}, Mean of last 100: {mean_last_100}')
        if epoch % 100 == 0:
            episode_reward = test_loop(actor)
        if mean_last_100 >= 200:
            print(f"Mean of last 100 episode rewards exceeds 200 ({mean_last_100}). Stopping training.")
            break
        
actor.eval()


# Run the agent on the test environment
for epoch in range(1, test_epochs + 1):
    episode_reward = test_loop(actor)
    print(f'Test Episode {epoch}, Total Reward: {episode_reward}')

plots_for_report()