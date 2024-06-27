import gymnasium as gym
import numpy as np
import cv2

env = gym.make("LunarLander-v2", render_mode="rgb_array")

# Set up parameters
show_every = 2000

# Adjust DISCRETE_OS_SIZE based on the environment's observation space
DISCRETE_OS_SIZE = [10] * len(env.observation_space.high)  # Adjust this based on the actual observation space
discrete_os_win_size = (env.observation_space.high - env.observation_space.low) / DISCRETE_OS_SIZE

# Create Q-table with adjusted dimensions
q_table = np.random.uniform(low=-2, high=0, size=(DISCRETE_OS_SIZE + [env.action_space.n]))

learning_rate = 0.1
discount = 0.95  # How important are future actions
episodes = 25000
epsilon = 0.5  # Exploration vs. exploitation
start_epsilon_decaying = 1
end_epsilon_decaying = episodes // 2
epsilon_decay_value = epsilon / (end_epsilon_decaying - start_epsilon_decaying)

def get_discrete_state(state):
    discrete_state = (state - env.observation_space.low) / discrete_os_win_size
    return tuple(np.clip(discrete_state.astype(int), 0, DISCRETE_OS_SIZE[i] - 1) for i in range(len(discrete_state)))

for episode in range(episodes):
    render = False
    if episode % show_every == 0:
        #print(f"Episode: {episode}")
        render = True
    print(f"Episode: {episode}")
    state, _ = env.reset()
    discrete_state = get_discrete_state(state)
    done = False

    while not done:
        if np.random.random() > epsilon:
            action = np.argmax(q_table[discrete_state])
        else:
            action = np.random.randint(0, env.action_space.n)

        # Ensure action is within valid range
        action = np.clip(action, 0, env.action_space.n - 1)

        new_state, reward, done, _, _ = env.step(action)
        new_discrete_state = get_discrete_state(new_state)

        if render:
            img = cv2.cvtColor(env.render(), cv2.COLOR_RGB2BGR)
            cv2.imshow("test", img)
            cv2.waitKey(50)
        lander = env.env
        #if not done and reward < 200:
        if not done:
            max_future_q = np.max(q_table[new_discrete_state])
            current_q = q_table[discrete_state + (action,)]
            new_q = (1 - learning_rate) * current_q + learning_rate * (reward + discount * max_future_q)
            q_table[discrete_state + (action,)] = new_q
        # elif env.env.envs[0].lander.position[1] > 0.1:  # Check if y-position is above 0.1 for success
        #     print(f"We made it on episode {episode}")
        #     q_table[discrete_state + (action,)] = 0
        #elif (lander.lander.position[1] <= 0.2 and abs(lander.lander.position[0]) <= 1.0 and abs(lander.lander.velocity[0]) < 0.05 and abs(lander.lander.velocity[1]) < 0.05):
        #elif not done and reward >= 200:
        elif abs(new_state[0]) <= 0.1 and abs(new_state[1]) <= 0.1 and abs(new_state[2]) <= 0.05 and abs(new_state[3]) <= 0.05 and abs(new_state[4]) <= 0.1 and abs(new_state[5]) <= 0.1 and abs(new_state[6]) == True and abs(new_state[7]) == True:
            print(f"We made it on episode {episode}")
            q_table[discrete_state + (action,)] = 0
            #done = True
        discrete_state = new_discrete_state

    if end_epsilon_decaying >= episode >= start_epsilon_decaying:
        epsilon -= epsilon_decay_value

env.close()
cv2.destroyAllWindows()
