# Load packages
import numpy as np
import gym
import torch as T
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import matplotlib.pyplot as plt
from tqdm import trange
import random as rd
from collections import deque, namedtuple
from DQN_agent import *


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')

# Parameters
N_episodes = 250                             # Number of episodes
discount_factor = 0.99                    # Value of the discount factor
learning_rate = 1e-3                        # Value of the learning rate
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality

# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

N = 64    # Training batch
L = 1e4 # Size of replay buffer
C = round(L/N) # Update frequency of target Neural Network
X = 20   # Number of random vector initially in the buffer in percentage
t_total = 0
update_rate = 1 # updates the net with this frequency
n_average=100

# Random agent initialization
agent = Agent(n_actions, learning_rate, discount_factor)

### Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)

B = ExperienceReplayBuffer(int(L))
for fill in range(int(L*X/100)) :
    action = np.random.randint(0,env.action_space.n) # Random action taking
    state  = env.reset() # Random state taking
    next_state, reward, done, _ = env.step(action)
    exp = Experience(state, action, reward, next_state, done)
    B.append(exp)

for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0
    y_vel = []
    while not done :
        # Take an action with epsilon greedy policy
        action = agent.forward(state, n_actions, N_episodes, i)

        # if (i+1)%1000 == 0: env.render()

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _ = env.step(action)
        y_vel.append(next_state[3])

        # Stops the code when it is stupid and does not land, or landed in a wrong place
        if np.sum(np.abs(y_vel[-n_average:]))/n_average < 0.05 and t>n_average:
            if np.abs(state[1]) < 1e-2:
                if state[6]==1 and state[7]==1:
                    reward = + 50
                    done = True
                else:
                    next_state, reward, done, _ = env.step(0)
            else:
                break

        
        # Append experience to the buffer
        exp = Experience(state, action, reward, next_state, done)
        B.append(exp)

        # Random batch selection
        batch = B.sample_batch(N)

        # Compute target values with neural network
        if t%update_rate==0: agent.backward(batch, (t_total+1)%C==0)

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+= 1
        t_total+=1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
        i, total_episode_reward, t,
        running_average(episode_reward_list, n_ep_running_average)[-1],
        running_average(episode_number_of_steps, n_ep_running_average)[-1]))





T.save(agent.mainNN,f'[Probelm 1] Deep Q-Networks/agents/Agent_gamma{int(discount_factor*100)}_neps{int(N_episodes)}_L{int(L)}.pt')



# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)], episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('')
ax[0].legend()
ax[0].grid(alpha=0.3)


fig.suptitle(f'DQN with Exponential ε-Greedy Policy\nNeps={N_episodes}, γ={discount_factor}, L={int(L)}, ε=[0.01, 0.99]', fontsize=16)

ax[1].plot([i for i in range(1, N_episodes+1)], episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.savefig(f"[Probelm 1] Deep Q-Networks/images/gamma{int(discount_factor*100)}_neps{int(N_episodes)}_L{int(L)}_linear.png")
plt.show()


