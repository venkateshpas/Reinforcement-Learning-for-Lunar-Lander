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
import pandas as pd


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

def training_cycle(input_layer_dimension, output_layer_dimension, train_epochs, gamma, ppo_steps_parameter, epsilon, test_epochs, train_env):
    all_rewards, loss_history_policy, loss_history_value, mean_rewards, mean_episodes = [], [], [], [], []
    episode_list = []
    actor = Actor(input_layer_dimension, output_layer_dimension)
    critic = Critic(input_layer_dimension)
    optimizer_actor = optim.Adam(actor.parameters(), lr=0.001) #Choose the Adam Optimizer as it is the state of the art even today
    optimizer_critic = optim.Adam(critic.parameters(), lr=0.001)

    
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
            mean_episodes.append(epoch)
            mean_rewards.append(mean_last_100)
            # if epoch % 10 == 0:
            if epoch % 10 == 0:
                print(f'Epoch: {epoch:3}, Reward: {episode_reward}, Mean of last 100: {mean_last_100}')
            # if epoch % 100 == 0:
            #     episode_reward = test_loop(actor)
            if mean_last_100 >= 200:
                print(f"Mean of last 100 episode rewards exceeds 200 ({mean_last_100}). Stopping training.")
                break
            
    actor.eval()

    torch.save(actor.state_dict(), 'actor_model.pth')
    torch.save(critic.state_dict(), 'critic_model.pth')
    # Run the agent on the test environment
    # for epoch in range(1, test_epochs + 1):
    #     episode_reward = test_loop(actor)
    #     print(f'Test Episode {epoch}, Total Reward: {episode_reward}')
    
    return episode_list, all_rewards, loss_history_policy, loss_history_value, mean_rewards, mean_episodes
    

def plots_for_one_training_cycle(episode_list, all_rewards, loss_history_policy, loss_history_value,mean_rewards, mean_episodes):
    fig, axs = plt.subplots(2,2, figsize=(12,8))

    episode_list_pd = pd.DataFrame(episode_list, columns=['Episode List'])
    episode_list_pd.to_csv('data/single_training/episode_list.csv')
    all_rewards_pd = pd.DataFrame(all_rewards, columns=['All Rewards'])
    all_rewards_pd.to_csv('data/single_training/all_rewards.csv')
    mean_rewards_pd = pd.DataFrame(mean_rewards, columns=['Mean Rewards'])
    mean_rewards_pd.to_csv('data/single_training/mean_rewards.csv')
    mean_episodes_pd = pd.DataFrame(mean_episodes, columns=['Mean Episodes'])
    mean_episodes_pd.to_csv('data/single_training/mean_episodes.csv')
    loss_history_policy_pd = pd.DataFrame(loss_history_policy, columns=['Loss History Policy'])
    loss_history_policy_pd.to_csv('data/single_training/loss_history_policy.csv')
    loss_history_value_pd = pd.DataFrame(loss_history_value, columns=['Loss History Value'])
    loss_history_value_pd.to_csv('data/single_training/loss_history_value.csv')

    axs[0,0].plot(episode_list, all_rewards)
    axs[0, 0].set_title("Variation of reward with episodes")
    axs[0, 0].set_xlabel("Episode Numbers")
    axs[0, 0].set_ylabel("Reward per episode")
    
    axs[0,1].plot(mean_episodes, mean_rewards)
    axs[0,1].set_title("Variation of avg reward per 100 episodes with episodes")
    axs[0,1].set_xlabel("Episode Numbers")
    axs[0,1].set_ylabel("Average Reward for 100 previous episodes")
    
    axs[1,0].plot(episode_list, loss_history_policy)
    axs[1,0].set_title("Variation of policy with episodes")
    axs[1,0].set_xlabel("Episode Numbers")
    axs[1,0].set_ylabel("Loss history policy")
    
    axs[1,1].plot(episode_list, loss_history_value)
    axs[1,1].set_title("Variation of loss value with episodes")
    axs[1,1].set_xlabel("Episode Numbers")
    axs[1,1].set_ylabel("Loss history value")
    
    fig.suptitle("Analysis of the agents performance for a training cycle")
    plt.tight_layout()

    # Show the plot
    plt.show()

def plots_for_multiple_training_cycle(varied_list, episode_list_loop, mean_rewards_loop, varied):
    


    fig, ax = plt.subplots(figsize=(12, 8)) 
    for count in range(len(varied_list)):
        ax.plot(episode_list_loop[varied_list[count]], mean_rewards_loop[varied_list[count]], label=f"{varied}: {varied_list[count]}")
    # Set the title and labels for the axes
    ax.set_title(f"Variation of avg reward per 100 episodes with episodes for different {varied}")
    ax.set_xlabel("Episode Numbers")
    ax.set_ylabel("Average Reward for 100 previous episodes")
    ax.legend()
    # fig, axs = plt.subplots(int(len(varied_list)/2),2, figsize=(12,8), squeeze=False)
    # row = []
    # for i in range(0,int(len(varied_list)/2)):
    #     row.append(i)
    # col = [0,1]
    # count = 0
    # for i in row:
    #     for j in col:
    #         axs[i,j].plot(episode_list_loop[varied_list[count]], mean_rewards_loop[varied_list[count]])
    #         axs[i,j].set_title(f"For {varied}: {varied_list[count]} Variation of avg reward per 100 episodes with episodes")
    #         axs[i,j].set_xlabel("Episode Numbers")
    #         axs[i,j].set_ylabel("Average Reward for 100 previous episodes")
    #         count += 1

    # axs[0,1].plot(mean_episodes, mean_rewards)
    # axs[0,1].set_title("Variation of avg reward per 100 episodes with episodes")
    # axs[0,1].set_xlabel("Episode Numbers")
    # axs[0,1].set_ylabel("Average Reward for 100 previous episodes")
    
    # axs[1,0].plot(episode_list, loss_history_policy)
    # axs[1,0].set_title("Variation of policy with episodes")
    # axs[1,0].set_xlabel("Episode Numbers")
    # axs[1,0].set_ylabel("Loss history policy")
    
    # axs[1,1].plot(episode_list, loss_history_value)
    # axs[1,1].set_title("Variation of loss value with episodes")
    # axs[1,1].set_xlabel("Episode Numbers")
    # axs[1,1].set_ylabel("Loss history value")
    
    plt.tight_layout()

    # Show the plot
    plt.show()