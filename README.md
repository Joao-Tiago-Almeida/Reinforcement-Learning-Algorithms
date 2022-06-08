# Reinforcement-Learning-Algorithms

Exercises of the Reinforcement Learning course (EL2805) at KTH, fall 2021.

The first exercise consisted of using different algorithms to compute the path for the agent to leave the maze without being caught by the minotaur. There were multiple variants, for example on whether the minotaur could wait in the same place or the maximum number of times the agent could move inside the maze. In this problem, it was applied the following algorithms:
- Dynamic programing
- Value Iteration
- Q-Learning
- SARSA
Additionally, topics as the epsilon-function,  the Bellman Equation, between others.

![](./MDPs%20and%20RL%20with%20linear%20function%20approximators/exiting%20the%20maze.gif)

In this assigment, it was used the Fourrier basis as a linear function approximator, to address the problem of the [Mountain Car](https://www.gymlibrary.ml/environments/classic_control/mountain_car/).

![](./MDPs%20and%20RL%20with%20linear%20function%20approximators/car.mov)

To approach the [Lunar Lander](https://www.gymlibrary.ml/environments/box2d/lunar_lander/) problem, with an distrect action space but continuous state space, it was used deep-reinforcement-learning. We updated the main netwrok with past experiences saved in the target network, that was being updated after some eiposed with the weights of the main network. Additionally, we impleted a relay buffer that randomized past states which allowed to train the network over time.  

![](./The%20Lunar%20Lander%20with%20Deep%20Reinforcement%20Learning/Deep%20Q-Networks/videos/DQN_R302_landing.mov)
