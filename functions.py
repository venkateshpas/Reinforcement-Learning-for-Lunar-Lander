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


#Defining the neural networks architecture: Actor, Critic model
class Actor(nn.Module):
    def __init__(self, input_layer, output_layer):
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_layer, 512),       
            nn.Dropout(p=0.15),              
            nn.PReLU(),                     
            nn.Linear(512, 128),
            nn.Dropout(p=0.15),  
            nn.PReLU(), 
            nn.Linear(128, output_layer),
            nn.Softmax(dim=-1)               
        )
        
    def forward(self, x):
        return self.net(x)


class Critic(nn.Module):
    def __init__(self, input_layer): #Evaluating so output should be 1, instead of softmax giving probabilities
        super().__init__()
        
        self.net = nn.Sequential(
            nn.Linear(input_layer, 512),     
            nn.Dropout(p=0.15),            
            nn.PReLU(),                    
            nn.Linear(512, 128),
            nn.Dropout(p=0.15),  
            nn.PReLU(),
            nn.Linear(128, 1)              
        )
        
    def forward(self, x):
        return self.net(x)
    
    
# class Actor(nn.Module):
#     def __init__(self, input_shape, action_space):
#         super(Actor, self).__init__()
#         self.action_space = action_space
        
#         self.fc1 = nn.Linear(input_shape, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 64)
#         self.fc4 = nn.Linear(64, action_space)
        
#         #self.optimizer = optimizer_cls(self.parameters(), lr=lr)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = F.softmax(self.fc4(x), dim=-1)
#         return x


# class Critic(nn.Module):
#     def __init__(self, input_shape):
#         super(Critic, self).__init__()
        
#         self.fc1 = nn.Linear(input_shape, 512)
#         self.fc2 = nn.Linear(512, 256)
#         self.fc3 = nn.Linear(256, 64)
#         self.fc4 = nn.Linear(64, 1)
        
#         #self.optimizer = optimizer_cls(self.parameters(), lr=lr)
        
#     def forward(self, x):
#         x = F.relu(self.fc1(x))
#         x = F.relu(self.fc2(x))
#         x = F.relu(self.fc3(x))
#         x = self.fc4(x)
#         return x

#Looping an episode until it terminates
def episode(train_env, actor, critic, state, states, actions, log_prob_actions, values, rewards, done,episode_reward):
    while not done:
        state = torch.FloatTensor(state).unsqueeze(0) #This adds an extra dimension to the state tensor. Helpful for compatibility reasons
        states.append(state)
        action_pred = actor(state)
        value_pred = critic(state)
        dist = distributions.Categorical(action_pred)
        action = dist.sample()
        log_prob_action = dist.log_prob(action)
        state, reward, terminated, trunked, _ = train_env.step(action.item())
        done = terminated or trunked
        actions.append(action)
        log_prob_actions.append(log_prob_action)
        values.append(value_pred)
        rewards.append(reward)
        episode_reward += reward
    return states, actions, log_prob_actions, values, rewards, episode_reward


def ppo_update(actor, critic, optimizer_actor, optimizer_critic, rewards, gamma, ppo_steps_parameter, epsilon, values, states, actions, log_prob_actions):
    returns, R = [], 0
    for r in reversed(rewards): 
        R = r + R * gamma
        returns.insert(0, R)
    
    returns = torch.tensor(returns)
    returns = (returns - returns.mean()) / returns.std()
    
    values = torch.cat(values).squeeze(-1)
    advantages = returns - values
    advantages = (advantages - advantages.mean()) / advantages.std()

    # Prepare for PPO update
    states = torch.cat(states)
    actions = torch.cat(actions)
    log_prob_actions = torch.cat(log_prob_actions).detach()
    advantages = advantages.detach()

    # PPO update step
    for _ in range(ppo_steps_parameter):
        # Get new action and value predictions
        action_pred = actor(states)
        value_pred = critic(states).squeeze(-1)
        
        # Calculate the ratio term for PPO
        dist = distributions.Categorical(action_pred)
        new_log_prob_actions = dist.log_prob(actions)
        policy_ratio = (new_log_prob_actions - log_prob_actions).exp()
        
        # Calculate both clipped and unclipped objective
        policy_loss_1 = policy_ratio * advantages
        policy_loss_2 = torch.clamp(policy_ratio, min=1.0 - epsilon, max=1.0 + epsilon) * advantages
        
        # Calculate policy and value losses
        policy_loss = -torch.min(policy_loss_1, policy_loss_2).sum()
        value_loss = F.smooth_l1_loss(returns, value_pred).sum()
        
        # Zero the gradients
        optimizer_actor.zero_grad()
        optimizer_critic.zero_grad()
        
        # Perform backpropagation
        policy_loss.backward()
        value_loss.backward()
        
        # Update the network weights
        optimizer_actor.step()
        optimizer_critic.step()
        
    return policy_loss, value_loss, optimizer_actor, optimizer_critic


def test_loop(actor):
    test_env = gym.make('LunarLander-v2',render_mode="human") 
    state,_ = test_env.reset()
    done = False
    episode_reward = 0
    while not done:
        test_env.render()
        state = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            action_prob = actor(state)
        dist = distributions.Categorical(action_prob)
        action = dist.sample()
        state, reward, terminated, trunked, _ = test_env.step(action.item())
        done = terminated or trunked
        episode_reward += reward
    test_env.close()
    return episode_reward

def plots_for_report(episode_list, all_rewards, loss_history_policy, loss_history_value):
    plt.plot(episode_list,all_rewards)
    plt.xlabel('Episode Number')
    plt.ylabel('Episode Reward')
    plt.legend()
    plt.show()

    plt.plot(episode_list,loss_history_policy)
    plt.xlabel('Episode Number')
    plt.ylabel('Loss history policy')
    plt.legend()
    plt.show()

    plt.plot(episode_list,loss_history_value)
    plt.xlabel('Episode Number')
    plt.ylabel('Loss history value')
    plt.legend()
    plt.show()