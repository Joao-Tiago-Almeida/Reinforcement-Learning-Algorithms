######################################
# Kungliga Tekniska Högskolan        #
# EL2805 - Reinforcement Learning    #
# LAB 1 - PROBLEM 2 - NEW ATTEMPT    #
# Authors :                          #
#  - João Almeida : 19990501-T210    #
#  - Victor Sanchez : 19980429-T517  #
# Date : 9 january 2022              #
######################################

# Load packages
import numpy as np
import gym
from numpy.core.fromnumeric import argmax
import matplotlib.pyplot as plt
import pickle
from tqdm import trange

# Import and initialize Mountain Car Environment
env = gym.make('MountainCar-v0')
env.reset()
nb_action = env.action_space.n      # tells you the number of actions
low, high = env.observation_space.low, env.observation_space.high


def scale_state_variables(s, low=env.observation_space.low, high=env.observation_space.high):
    # Rescaling of s to the box [0,1]^2
    x = (s - low) / (high - low) # x remain in [low,high] interval
    return x

# Q function
def Q_function(state,weight, eta):
    x = scale_state_variables(state)
    phi = np.cos(np.pi*np.dot(eta , x))
    Q = np.dot(weight,phi)
    return Q

# Functions used during training
def running_average(x, N):
    # Function used to compute the running mean
    #    of the last N elements of a vector x
    
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y

# Eligibility trace algorithm : OK
def eligibility_trace(z, discount_factor, lambda_value, state, nb_basis, weight, action,eta):

    z_new = discount_factor*lambda_value*z

    # when a==a_t
    grad_wa_Q_t = scale_state_variables(state)
    z_new[action,:] += np.cos(np.pi*np.dot(eta,grad_wa_Q_t))

    z_new = np.clip(z_new,-5,5) # Clipping to avoir gradient problem
    return z_new

# Feedback function
def feedback_function(reward, action, state):
    print('/n','The reward is :',reward)
    print('/n The action took is :',action)
    print('/n The state is :',state)


def Sarsa_lambda(env, N_episodes, discount_factor, alpha, lambda_value, nb_basis, eta, agent_type):

    W = np.zeros([nb_action,nb_basis])   # Weight initialization

    R_best = -200
    episode_reward_list = []
    for i in range(N_episodes):

        # Reset enviroment data
        total_episode_reward = 0.
        done = False
        state = env.reset()
        
        # Qvalue initialization
        Q = Q_function(state,W,eta)
        if agent_type == 'random':
            action = np.random.randint(0,3)
        elif agent_type == 'optimal':
            action = np.argmax(Q)

        
        

        z = np.zeros(shape=W.shape)   # Eligibility Trace initialization
        
        # Training process
        while not done:
            #if (i+1)%10==0: env.render()
            
            # Initialization
            next_state, reward, done, _ = env.step(action)
            Q_new = Q_function(next_state,W,eta)
            #next_state = scale_state_variables(next_state)
            if agent_type == 'random':
                next_action = np.random.randint(0,3)
            elif agent_type == 'optimal':
                next_action = np.argmax(Q)
            

            # Update of the episode reward
            total_episode_reward += reward

            # Delta computing for Stochastic Gradient Descent
            delta = reward + discount_factor * Q_new[next_action] - Q[action]
            W += alpha * delta * z

            # Eligibility Trace # Maybe need to add conditions
            z = eligibility_trace(z, discount_factor, lambda_value, state, nb_basis, W, action,eta)   

            # Update for next iteration
            action = next_action
            state = next_state
            Q = Q_new

        if total_episode_reward > R_best:
            W_best = W.copy()
            alpha = alpha * 0.7
            R_best = total_episode_reward

        #print(f'Episode {i} \t Reward: {total_episode_reward}') 
        episode_reward_list.append(total_episode_reward)
        # Close environment
        env.close()

    return W,episode_reward_list



def Plotter(N_EPISODES,eta,w):
    
    # Reward
    episode_reward_list = []  # Used to store episodes reward
    optimal_Qvalue = []
    optimal_policy = []
    position = []
    speed = []
    # Simulate episodes
    print('Checking solution...')
    EPISODES = trange(N_EPISODES, desc='Episode: ', leave=True)
    for i in EPISODES:
        EPISODES.set_description("Episode {}".format(i))
        # Reset enviroment data
        done = False
        state = scale_state_variables(env.reset())
        total_episode_reward = 0.

        qvalues = Q_function(state,w,eta)
        action = np.argmax(qvalues)
        

        while not done:
            # Get next state and reward.  The done variable
            # will be True if you reached the goal position,
            # False otherwise

            next_state, reward, done, _ = env.step(action)
            position.append(next_state[0])
            speed.append(next_state[1])
            optimal_policy.append(action)
            next_state = scale_state_variables(next_state)
            qvalues_next = Q_function(next_state,w,eta)
            next_action = np.argmax(qvalues_next)

            # Update episode reward
            total_episode_reward += reward

            # Update state for next iteration
            state = next_state
            qvalues = qvalues_next
            action = next_action
            optimal_Qvalue.append(max(qvalues))

        # Append episode reward
        episode_reward_list.append(total_episode_reward)

        # Close environment
        env.close()

    return(optimal_Qvalue,optimal_policy,position,speed)



