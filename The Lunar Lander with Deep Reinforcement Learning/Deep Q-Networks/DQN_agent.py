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

class Agent():
    ''' Base agent class, used as a parent class

        Args:
            n_actions (int): number of actions

        Attributes:
            n_actions (int): where we store the number of actions
            last_action (int): last action taken by the agent
    '''
    def __init__(self, n_actions, lr, discount_factor: int = 4):
        super(Agent, self).__init__()  # Initialize the parent class
        self.n_actions = n_actions
        self.last_action = None
        self.discount_factor = discount_factor

        # Network initialization
        self.mainNN = NeuralNet()
        self.targetNN = NeuralNet()

        self.optimizer = T.optim.Adam(self.mainNN.parameters(), lr=lr)
        self.criterion = nn.MSELoss()
        

    def forward(self, state, n_actions, nb_episode, episode_k, epsilon_min = 0.01, epsilon_max = 0.99, mode = 'exponential') -> int:
        ''' Performs a forward computation '''

        Q_value = self.mainNN.forward(T.tensor(state, dtype=T.float, requires_grad=True))
        
        g = (episode_k - 1)/(nb_episode - 1)

        if mode == 'exponential' :
            epsilon = max(epsilon_min, epsilon_max * (epsilon_min/epsilon_max)**g)
        if mode == 'linear' : 
            epsilon = max(epsilon_min, epsilon_max - ((epsilon_max - epsilon_min)*g))
        prob = np.random.uniform(0, 1)
        a = 0
        # print(epsilon)
        if prob < epsilon:
            a = np.random.choice(n_actions)
        else:
            a = T.argmax(Q_value).item()
        return int(a)


    def backward(self, bacth, set_target_equal_main = False):
        # perform the backward pass

        rewards = T.tensor(bacth[2], dtype=T.float, requires_grad=True)
        actions = T.tensor(bacth[1], dtype=T.int64)
        not_done = 1 - T.tensor(bacth[4], dtype=T.int8)

        q_main = self.mainNN.forward(T.tensor(bacth[0], dtype=T.float, requires_grad=True))
        q_target = self.targetNN.forward(T.tensor(bacth[3], dtype=T.float, requires_grad=True))
        
        q_main = q_main.gather(1,actions.unsqueeze(-1))
        
        # Compute the target values
        q = rewards + self.discount_factor*q_target.max(1)[0]*not_done
        
        # Compute gradients
        loss = self.criterion(q_main,q.detach().unsqueeze(-1))
        self.optimizer.zero_grad()
        loss.backward()        
        nn.utils.clip_grad_norm_(self.mainNN.parameters(), max_norm=1.)
        self.optimizer.step()
     
        if set_target_equal_main:
            self.targetNN.load_state_dict(self.mainNN.state_dict())
        
        # for target_param, main_param in zip(self.targetNN.parameters(), self.mainNN.parameters()):
        #     target_param.data.copy_(1e-3*main_param.data + (1.0-1e-3)*target_param.data)

        breakpoint

    

class NeuralNet(nn.Module):
    def __init__(self, input_size:int = 8, output_size:int = 4):
        super(NeuralNet, self).__init__()
        
        hidden_size = 64
        self.layer1 = nn.Linear(input_size, hidden_size)     # Definition of layer 1
        self.layer2 = nn.Linear(hidden_size, hidden_size)             # Definition of layer 2
        self.layer3 = nn.Linear(hidden_size, output_size)    # Definition of layer 3

        self.layer1_activation = nn.ReLU()          # ReLU activation for layer 1
        self.layer2_activation = nn.ReLU()          # ReLU activation for layer 2


    def forward(self, state):
        # First layer
        y1 = self.layer1(state)
        y1 = self.layer1_activation(y1)

        # Second layer
        y2 = self.layer2(y1)
        y2 = self.layer2_activation(y2)

        # Final output
        out = self.layer3(y2)
        return out

class RandomAgent(Agent):
    ''' Agent taking actions uniformly at random, child of the class Agent'''
    def __init__(self, n_actions: int):
        super(RandomAgent, self).__init__(n_actions)

    def forward(self, state: np.ndarray) -> int:
        ''' Compute an action uniformly at random across n_actions possible
            choices

            Returns:
                action (int): the random action
        '''
        self.last_action = np.random.randint(0, self.n_actions)
        return self.last_action


#######################################################################################
#                   Why a replay buffer is important ?                                #
#  Reinforcement learning algorithms use replay buffers to store experience           #
#  trajectories when executing a policy in an environment.                            #
#  During training, the replay buffers are polled for a subset of the trajectories    #
#  (either a sequential subset or a sample) to "replay" the agent experience.         #
#######################################################################################
Experience = namedtuple('Experience',['state', 'action', 'reward', 'next_state', 'done'])

class ExperienceReplayBuffer(object):
    """ Class used to store a buffer containing experiences of the RL agent.
    """
    def __init__(self, maximum_length=30000):
        # Create buffer of maximum length
        self.buffer = deque(maxlen=maximum_length)

    def append(self, experience):
        # Append experience to the buffer
        self.buffer.append(experience)

    def __len__(self):
        # overload len operator
        return len(self.buffer)

    def sample_batch(self, n):
        """ Function used to sample experiences from the buffer.
            returns 5 lists, each of size n. Returns a list of state, actions,
            rewards, next states and done variables.
        """
        # If we try to sample more elements that what are available from the
        # buffer we raise an error
        if n > len(self.buffer):
            raise IndexError('Tried to sample too many elements from the buffer!')

        # Sample without replacement the indices of the experiences
        # np.random.choice takes 3 parameters: number of elements of the buffer,
        # number of elements to sample and replacement.
        indices = np.random.choice(
            len(self.buffer)-1,
            size=n-1,
            replace=False
        )
        indices = np.append(indices, len(self.buffer)-1)

        # Using the indices that we just sampled build a list of chosen experiences
        batch = [self.buffer[i] for i in indices]

        # batch is a list of size n, where each element is an Experience tuple
        # of 5 elements. To convert a list of tuples into
        # a tuple of list we do zip(*batch). In this case this will return a
        # tuple of 5 elements where each element is a list of n elements.
        samples = list(zip(*batch))
        samples[0] = np.array(samples[0]).copy()
        samples[3] = np.array(samples[3])
        return tuple(samples)
