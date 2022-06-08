import torch
import gym
import pickle


L = 10000
N_episodes = 500
discount_factor = 0.99

net = torch.load(
    f"[Probelm 1] Deep Q-Networks/agents/Agent_gamma{int(discount_factor*100)}_neps{int(N_episodes)}_L{int(L)}.pt"
    )

# Import and initialize the discrete Lunar Laner Environment
env = gym.make('LunarLander-v2')

total_episode_reward = 0.
tempo = []
while total_episode_reward < 400:

    done = False
    state = env.reset()
    total_episode_reward = 0.
    t = 0

    while not done:
        # env.render()
        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        q_values = net(torch.tensor([state]))
        _, action = torch.max(q_values, axis=1)
        next_state, reward, done, _ = env.step(action.item())

        # Update episode reward
        total_episode_reward += reward

        # Update state for next iteration
        state = next_state
        t+=1

    tempo.append(t)

print(f"\rReward = {total_episode_reward}")