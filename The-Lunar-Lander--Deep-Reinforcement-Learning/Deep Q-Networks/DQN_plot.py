# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch as T
from tqdm import trange
import matplotlib.pyplot as plt

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

# Load model
try:
    model = T.load('[Probelm 1] Deep Q-Networks/agents/Agent_gamma99_neps500_L10000.pt')
    print('Network model: {}'.format(model))
except:
    print('File not found!')
    exit(-1)

env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_EPISODES = 50            # Number of episodes to run for trainings
CONFIDENCE_PASS = 50

# Reward
episode_reward_list = []  # Used to store episodes reward

"""# Simulate episodes
print('Checking solution...')
EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
for i in EPISODES:
    EPISODES.set_description("Episode {}".format(i))
    # Reset enviroment data
    done = False
    state = env.reset()
    total_episode_reward = 0.
    while not done:
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        q_values = model(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        next_state, reward, done, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state

    # Append episode reward
    episode_reward_list.append(total_episode_reward)

    # Close environment
    env.close()

avg_reward = np.mean(episode_reward_list)
confidence = np.std(episode_reward_list) * 1.96 / np.sqrt(N_EPISODES)
"""

N = 100

y_coord = np.linspace(0, 1.5 ,N)
omega_coord = np.linspace(-np.pi, np.pi, N)

Q_values=np.empty(shape=(N,N))
Q_arg_values=np.empty(shape=(N,N))
for i in range(N):
    for j in range(N):
        Q_values[i,j] = T.max(model.forward(T.tensor(np.array([0, y_coord[i], 0, 0, omega_coord[j], 0, 0, 0],dtype=np.float32)))).item()
        Q_arg_values[i,j] = T.argmax(model.forward(T.tensor(np.array([0, y_coord[i], 0, 0, omega_coord[j], 0, 0, 0],dtype=np.float32)))).item()

X, Y = np.meshgrid(omega_coord, y_coord)


fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9),subplot_kw={"projection": "3d"})


ax[0].plot_surface(X, Y, Q_values, rstride=1, cstride=1,
                cmap='viridis', edgecolor='none')

ax[0].set_title('Maximum expected reward for the first action for different initial states')
ax[0].set_xlabel('Angle of the lander')
ax[0].set_ylabel('Height of the lander')
ax[0].set_zlabel('Q-values')

fig.suptitle("Behaviour of the model for different initializations")

ax[1].plot_surface(X, Y, Q_arg_values, rstride=1, cstride=1,
                cmap='coolwarm', edgecolor='none')

ax[1].set_title('Action for the first action for different initial states')
ax[1].set_xlabel('Angle of the lander')
ax[1].set_ylabel('Height of the lander')
ax[1].set_zlabel('0 - engines off\t1 - left engine\n2- main engine\t3 - right engine')

plt.savefig("[Probelm 1] Deep Q-Networks/images/y_w_initializations.png")
plt.show()