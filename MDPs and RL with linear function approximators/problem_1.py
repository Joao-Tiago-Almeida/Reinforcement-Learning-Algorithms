######################################
# Kungliga Tekniska Högskolan        #
# EL2805 - Reinforcement Learning    #
# LAB 1 - PROBLEM 1 - NEW ATTEMPT    #
# Authors :                          #
#  - João Almeida : 19990501-T210    #
#  - Victor Sanchez : 19980429-T517  #
# Date : 9 january 2022              #
######################################

# Import of package
import numpy as np
import matplotlib.pyplot as plt
import time
from IPython import display
import random

from numpy.core.fromnumeric import argmax

# Implemented methods
methods = ['DynProg', 'ValIter'];

# Some colours
LIGHT_RED    = '#FFC4CC';
LIGHT_GREEN  = '#95FD99';
BLACK        = '#000000';
WHITE        = '#FFFFFF';
LIGHT_PURPLE = '#E8D0FF';
LIGHT_ORANGE = '#FAE0C3';
YELLOW       = '#F4D03F';

class Maze:

    # Actions
    STAY       = 0
    MOVE_LEFT  = 1
    MOVE_RIGHT = 2
    MOVE_UP    = 3
    MOVE_DOWN  = 4

    # Give names to actions
    actions_names = {
        STAY: "stay",
        MOVE_LEFT: "move left",
        MOVE_RIGHT: "move right",
        MOVE_UP: "move up",
        MOVE_DOWN: "move down"
    }

    # Reward values
    STEP_REWARD = -1
    GOAL_REWARD = 0
    IMPOSSIBLE_REWARD = -1


    def __init__(self, maze, weights=None, random_rewards=False):
        """ Constructor of the environment Maze.
        """
        self.maze                     = maze;
        self.actions                  = self.__actions();
        self.states, self.map         = self.__states();
        self.n_actions                = len(self.actions);
        self.n_states                 = len(self.states);
        self.transition_probabilities = self.__transitions();
        self.rewards                  = self.__rewards(weights=weights,
                                                random_rewards=random_rewards);

    def __actions(self):
        actions = dict();
        actions[self.STAY]       = (0, 0);
        actions[self.MOVE_LEFT]  = (0,-1);
        actions[self.MOVE_RIGHT] = (0, 1);
        actions[self.MOVE_UP]    = (-1,0);
        actions[self.MOVE_DOWN]  = (1,0);
        return actions;

    def __states(self):
        states = dict();
        map = dict();
        end = False;
        s = 0;
        for i in range(self.maze.shape[0]):
            for j in range(self.maze.shape[1]):
                if self.maze[i,j] != 1:
                    states[s] = (i,j);
                    map[(i,j)] = s;
                    s += 1;
        return states, map

    def __move(self, state, action):
        """ Makes a step in the maze, given a current position and an action.
            If the action STAY or an inadmissible action is used, the agent stays in place.

            :return tuple next_cell: Position (x,y) on the maze that agent transitions to.
        """
        # Compute the future position given current (state, action)
        row = self.states[state][0] + self.actions[action][0];
        col = self.states[state][1] + self.actions[action][1];
        # Is the future position an impossible one ?
        hitting_maze_walls =  (row == -1) or (row == self.maze.shape[0]) or \
                              (col == -1) or (col == self.maze.shape[1]) or \
                              (self.maze[row,col] == 1);
        # Based on the impossiblity check return the next state.
        if hitting_maze_walls:
            return state;
        else:
            return self.map[(row, col)];

    def __transitions(self):
        """ Computes the transition probabilities for every state action pair.
            :return numpy.tensor transition probabilities: tensor of transition
            probabilities of dimension S*S*A
        """
        # Initialize the transition probailities tensor (S,S,A)
        dimensions = (self.n_states,self.n_states,self.n_actions);
        transition_probabilities = np.zeros(dimensions);

        # Compute the transition probabilities. Note that the transitions
        # are deterministic.
        for s in range(self.n_states):
            for a in range(self.n_actions):
                next_s = self.__move(s,a);
                transition_probabilities[next_s, s, a] = 1;
        return transition_probabilities;

    def __rewards(self, weights=None, random_rewards=None):

        rewards = np.zeros((self.n_states, self.n_actions));

        # If the rewards are not described by a weight matrix
        if weights is None:
            for s in range(self.n_states):
                for a in range(self.n_actions):
                    next_s = self.__move(s,a);
                    # Rewrd for hitting a wall
                    if s == next_s and a != self.STAY:
                        rewards[s,a] = self.IMPOSSIBLE_REWARD;
                    # Reward for reaching the exit
                    elif s == next_s and self.maze[self.states[next_s]] == 2:
                        rewards[s,a] = self.GOAL_REWARD;
                    # Reward for taking a step to an empty cell that is not the exit
                    else:
                        rewards[s,a] = self.STEP_REWARD;

                    # If there exists trapped cells with probability 0.5
                    if random_rewards and self.maze[self.states[next_s]]<0:
                        row, col = self.states[next_s];
                        # With probability 0.5 the reward is
                        r1 = (1 + abs(self.maze[row, col])) * rewards[s,a];
                        # With probability 0.5 the reward is
                        r2 = rewards[s,a];
                        # The average reward
                        rewards[s,a] = 0.5*r1 + 0.5*r2;
        # If the weights are descrobed by a weight matrix
        else:
            for s in range(self.n_states):
                 for a in range(self.n_actions):
                     next_s = self.__move(s,a);
                     i,j = self.states[next_s];
                     # Simply put the reward as the weights o the next state.
                     rewards[s,a] = weights[i][j];

        return rewards;

    def simulate(self, start, policy, method, life_0 = 30):
        if method not in methods:
            error = 'ERROR: the argument method must be in {}'.format(methods);
            raise NameError(error);

        # Initialize current state, next state and life && Get the initial life
        life = life_0;

        path = list();
        if method == 'DynProg':
            # Deduce the horizon from the policy shape
            horizon = policy.shape[1];
            # Initialize current state and time
            t = 0;
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            while t < horizon-1:
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s,t]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                t +=1;
                s = next_s;
        if method == 'ValIter':
            s = self.map[start];
            # Add the starting position in the maze to the path
            path.append(start);
            # Move to next state given the policy and the current state
            next_s = self.__move(s,policy[s]);
            # Add the position in the maze corresponding to the next state
            # to the path
            path.append(self.states[next_s]);
            # Loop while state is not the goal state
            while s != next_s and  life>0:
                # Update state
                s = next_s;
                # Move to next state given the policy and the current state
                next_s = self.__move(s,policy[s]);
                # Add the position in the maze corresponding to the next state
                # to the path
                path.append(self.states[next_s])
                # Update time and state for next iteration
                life -= 1;
            
            if life == 0:
                path = [start,]
        return path, life

    def show(self):
        print('The states are :')
        print(self.states)
        print('The actions are:')
        print(self.actions)
        print('The mapping of the states:')
        print(self.map)
        print('The rewards:')
        print(self.rewards)

def dynamic_programming(env, horizon, minotaurs_stay=False):
    """ Solves the shortest path problem using dynamic programming
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input int horizon        : The time T up to which we solve the problem.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """

    # The dynamic prgramming requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;
    T         = horizon;

    # The variables involved in the dynamic programming backwards recursions
    V      = np.zeros((n_states, T+1));
    policy = np.zeros((n_states, T+1));
    Q      = np.zeros((n_states, n_actions));

    # Initialization
    Q            = np.copy(r);
    V[:, T]      = np.max(Q,1);
    policy[:, T] = np.argmax(Q,1);

    # initialize the minotaurs path
    maze = env.maze
    minotaurs_start = (np.where(maze == 2)[0][0],np.where(maze == 2)[1][0])

    minotaurs_path = randomize_minotaur_path(
                        start=minotaurs_start,
                        maze_dimensions=maze.shape,
                        T=T,
                        stay=minotaurs_stay
                    )
    # import pickle
    # minotaurs_path = pickle.load( open( "minotaurs_path.p", "rb" ) )

    # The dynamic programming bakwards recursion
    for t in range(T-1,-1,-1):
        # Update the value function acccording to the bellman equation
        for s in range(n_states):
            for a in range(n_actions):
                # Update of the temporary Q values
                Q[s,a] = r[s,a] + np.dot(p[:,s,a], V[:,t+1])
        
        # Update by taking the maximum Q value w.r.t the action a
        V[:,t], policy[:,t] = max_considering_the_minotaur(Q,env,minotaurs_path[t],minotaurs_stay)

        # The optimal action is the one that maximizes the Q function
    return V, policy, minotaurs_path

def value_iteration(env, discount_factor, epsilon, minotaurs_stay=False): 
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float discount_factor        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value iteration algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    r         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # Required variables and temporary ones for the VI to run
    V   = np.zeros(n_states);
    Q   = np.zeros((n_states, n_actions));
    BV  = np.zeros(n_states);
    # Iteration counter
    n   = 0;
    # Tolerance error
    tol = epsilon*(1-discount_factor)/discount_factor

    # Initialization of the VI
    for s in range(n_states):
        for a in range(n_actions):
            Q[s, a] = r[s, a] + discount_factor*np.dot(p[:,s,a],V);
    BV = np.max(Q, 1);

    # # initialize the minotaurs path
    # maze = env.maze
    minotaurs_start = (np.where(maze == 2)[0][0],np.where(maze == 2)[1][0])
    minotaurs_path = [minotaurs_start,]
    minotaurs_stay = False

    # Starting live
    n_runs = 200

    # Iterate until convergence
    while np.linalg.norm(V - BV) >= tol and n < n_runs:
        # Increment by one the numbers of iteration
        n += 1;
        # Update the value function
        V = np.copy(BV);
        # Compute the new BV
        for s in range(n_states):
            for a in range(n_actions):
                Q[s, a] = r[s, a] + discount_factor*np.dot(p[:,s,a],V);

        # Computing maximum avergae reward and policy
        # BV, policy = max_considering_the_minotaur(Q,env,minotaurs_path[-1],minotaurs_stay)
        # MODIFICATION HERE
        BV = np.max(Q, 1);  #like in lab0

        # Show error
        #print(np.linalg.norm(V - BV))

        # get the minotaurs next position

        # if random.uniform(0,1) < 0.65:
        #     move = randomize_minotaur_path(start=minotaurs_path[-1],maze_dimensions=maze.shape,T=1,stay=minotaurs_stay)[1]
        # else:
        #     move = orientated_minotaur_path(start = minotaurs_path[-1], goal = np.array(pathA[t]), T = 1)

        # minotaurs_path.append(move)

        
    # MODIFICATION HERE
    policy = np.argmax(Q,1) #like in lab0
    return V, policy;

def q_learning_or_sarsa(env, discount_factor, epsilon, alpha, algorithm = "SARSA", start = (0,0), axes=[], Q = [], policy=[]):
    """ Solves the shortest path problem using value iteration
        :input Maze env           : The maze environment in which we seek to
                                    find the shortest path.
        :input float discount_factor        : The discount factor.
        :input float epsilon      : accuracy of the value iteration procedure.
        :return numpy.array V     : Optimal values for every state at every
                                    time, dimension S*T
        :return numpy.array policy: Optimal time-varying policy at every state,
                                    dimension S*T
    """
    # The value itearation algorithm requires the knowledge of :
    # - Transition probabilities
    # - Rewards
    # - State space
    # - Action space
    # - The finite horizon
    p         = env.transition_probabilities;
    R         = env.rewards;
    n_states  = env.n_states;
    n_actions = env.n_actions;

    # number of times the pair (s,a) was visited
    n_s_a     = np.zeros(shape = R.shape)   

    stop = (np.where(env.maze == 2)[0][0], np.where(env.maze == 2)[1][0])

    s0 = env.map[start]          # Initial state
    s = s0
    final_s = env.map[stop]     # terminal satet
    next_s = -1

    # Required variables and temporary ones for the VI to run
    if Q == []: Q   = R.copy()
    if policy == []:  policy = np.ones(shape=(Q.shape[0],1))
    V = np.ones(shape=(Q.shape[0],1))
    
    # Q[final_s] = 0
    BQ  = np.zeros((n_states, n_actions));
    # Iteration counter (episodes and different states in each episode)
    n_eps_init = 5000
    n_eps = 0
    n_run_per_ep = 500

    # Tolerance error
    #tol = (1 - discount_factor)* epsilon/discount_factor;
    tol = 1e-3

    # initialize the minotaurs path
    maze = env.maze
    minotaurs_start = (np.where(maze == 2)[0][0],np.where(maze == 2)[1][0])
    minotaurs_stay = False

    value_function_per_episode = [[],[]]

    # Iterate until convergence
    while (n_eps==0 or np.linalg.norm(Q - BQ) >= tol) and n_eps < n_eps_init:
        # Update the numbers of iteration
        n_eps+=1
        life = n_run_per_ep
        # delta = 0.8               #Question (i) 2)
        # epsilon = 1./(n_eps**delta) #Question (i) 2)
        # restart minotours path
        minotaurs_path = [minotaurs_start,]

        # save last Q matrix
        visited_states = np.sum(n_s_a,1)>0
        BQ[visited_states] = Q[visited_states].copy()
        
        # Initialize S
        next_s = s0#np.unravel_index(n_s_a.argmin(), n_s_a.shape)[0]#random.randint(0, n_states-1)
        if algorithm == "SARSA":
            # Choose action (a) using e-greedy
            a = epsilon_greedy(epsilon, Q[next_s], np.sum(p[:,s,:],axis=0),env,minotaurs_path[-1])

        # Repeat (for each step of each episode)
        
        while not (s == final_s and next_s == final_s) and life > 0:
            
            life -= 1
            s = next_s

            if(list(env.map.keys())[s] == minotaurs_path[-1]):
                break

            if algorithm == "Q-Learning":
                # Choose action (a) using e-greedy
                a = epsilon_greedy(epsilon, Q[s], np.sum(p[:,s,:],axis=0),env,minotaurs_path[-1])

            # Taking action (a) and observe r and s'
            next_s = np.argmax(p[:,s,a])
            r = R[s,a]
            
            #visit the state 
            n_s_a[s,a] += 1
            # step size
            alpha = 1/(np.power(n_s_a[s,a],alpha)) #

            if algorithm == "Q-Learning":
                Q[s, a] = Q[s, a] + alpha*(r + discount_factor*np.max(Q[next_s]) - Q[s, a]);
                
            elif algorithm == "SARSA":
                
                # Choose action (a) using e-greedy
                next_a = epsilon_greedy(epsilon, Q[next_s], np.sum(p[:,s,:],axis=0),env,minotaurs_path[-1])

                Q[s, a] = Q[s, a] + alpha*(r + discount_factor*Q[next_s,next_a] - Q[s, a]);
                a = next_a
            else:
                return None

            # Show error
            # print(np.linalg.norm(Q - BQ))

            # get the minotaurs next position
            minotaurs_path.append(
                randomize_minotaur_path(
                    start=minotaurs_path[-1],
                    maze_dimensions=maze.shape,
                    T=1,
                    stay=minotaurs_stay
                )[1])

        V, policy = max_considering_the_minotaur(Q,env,minotaurs_path[-1],minotaurs_stay)
        value_function_per_episode[0].append(n_eps)
        value_function_per_episode[1].append(np.sum(V))

    if axes != []:
        axes.plot(value_function_per_episode[0], value_function_per_episode[1])


    print(f"Number of episodes: {n_eps}")

    # # Return the obtained policy
    # for s in range(Q.shape[0]):

    #     if np.argmax(Q[s]) == 0 and s != final_s:
    #         policy[s] = np.argmax(Q[s,1:])+1
    #     else:
    #         policy[s] = np.argmax(Q[s])

    for s in range(Q.shape[0]):
        if s == final_s: continue
        policy[s] = 1+np.argmax(Q[s,1:])
    
    return Q, policy;

def epsilon_greedy(epsilon, Qs, A, env, minotaurs_coord):
    prob = random.uniform(0, 1)
    As = list(np.where(A==1)[0]) # available actions
    a = 0
    if prob < epsilon:
        a = random.choice(As)
    else:
        # a = As[np.argmax(Qs[As])]
        _, a = max_considering_the_minotaur(Qs,env,minotaurs_coord)
    return int(a)

def draw_maze(maze):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, -2: YELLOW};

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('The Maze');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    rows,cols    = maze.shape;
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                            cellColours=colored_maze,
                            cellLoc='center',
                            loc=(0,0),
                            edges='closed');
    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);

def animate_solution(maze, path, minotaurs_path = []):

    # Map a color to each cell in the maze
    col_map = {0: WHITE, 1: BLACK, 2: LIGHT_GREEN, -6: LIGHT_RED, -1: LIGHT_RED, -2: YELLOW};

    # Size of the maze
    rows,cols = maze.shape;

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows));

    # Remove the axis ticks and add title title
    ax = plt.gca();
    ax.set_title('Policy simulation');
    ax.set_xticks([]);
    ax.set_yticks([]);

    # Give a color to each cell
    colored_maze = [[col_map[maze[j,i]] for i in range(cols)] for j in range(rows)];

    # Create figure of the size of the maze
    fig = plt.figure(1, figsize=(cols,rows))

    # Create a table to color
    grid = plt.table(cellText=None,
                     cellColours=colored_maze,
                     cellLoc='center',
                     loc=(0,0),
                     edges='closed');

    # Modify the hight and width of the cells in the table
    tc = grid.properties()['children']
    for cell in tc:
        cell.set_height(1.0/rows);
        cell.set_width(1.0/cols);


    # Update the color at each frame
    for i in range(len(path)):
        grid.get_celld()[(path[i])].set_facecolor(LIGHT_ORANGE)
        grid.get_celld()[(path[i])].get_text().set_text('Player')
        if minotaurs_path:
            grid.get_celld()[(minotaurs_path[i])].set_facecolor(LIGHT_PURPLE)
            grid.get_celld()[(minotaurs_path[i])].get_text().set_text('Minoaturs')
        if i > 0:
            if minotaurs_path and path[i] == minotaurs_path[i] and maze[path[i-1]] != 2:    # already in the exit
                grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
                grid.get_celld()[(path[i-1])].get_text().set_text('')
                grid.get_celld()[(minotaurs_path[i-1])].set_facecolor(col_map[maze[minotaurs_path[i-1]]])
                grid.get_celld()[(minotaurs_path[i-1])].get_text().set_text('')
                grid.get_celld()[(minotaurs_path[i])].set_facecolor(LIGHT_RED)
                grid.get_celld()[(minotaurs_path[i])].get_text().set_text('GAME OVER')
                break
            elif path[i] == path[i-1] and maze[path[i]] == 2:
                grid.get_celld()[(path[i])].set_facecolor(LIGHT_GREEN)
                grid.get_celld()[(path[i])].get_text().set_text('Player is out')
                if minotaurs_path and minotaurs_path[i] != minotaurs_path[i-1]:
                    grid.get_celld()[(minotaurs_path[i-1])].set_facecolor(col_map[maze[minotaurs_path[i-1]]])
                    grid.get_celld()[(minotaurs_path[i-1])].get_text().set_text('')
                break
            else:
                if (minotaurs_path and minotaurs_path[i] != path[i-1]) or not minotaurs_path:
                    if path[i] != path[i-1]:
                        grid.get_celld()[(path[i-1])].set_facecolor(col_map[maze[path[i-1]]])
                        grid.get_celld()[(path[i-1])].get_text().set_text('')
                if minotaurs_path and minotaurs_path[i] != minotaurs_path[i-1] and minotaurs_path[i-1] != path[i]:
                    grid.get_celld()[(minotaurs_path[i-1])].set_facecolor(col_map[maze[minotaurs_path[i-1]]])
                    grid.get_celld()[(minotaurs_path[i-1])].get_text().set_text('')

        # plt.savefig(f"figure {i}")
        display.display(fig)
        display.clear_output(wait=True)
        time.sleep(0.5)

def randomize_minotaur_path(start = np.array([0,0]), maze_dimensions = (7,8), T = 20, stay = False) -> list:
    path = [()]*(T+1)
    
    path[0] = tuple(list(start))

    for t in range(1,T+1):
        actions = []
        if stay: actions.append(np.array([0,0]))                                   # stay
        if path[t-1][1] > 0: actions.append(np.array([0,-1]))                      # left
        if path[t-1][1] < maze_dimensions[1]-1: actions.append(np.array([0,1]))    # right
        if path[t-1][0] > 0: actions.append(np.array([-1,0]))                      # up
        if path[t-1][0] < maze_dimensions[0]-1: actions.append(np.array([1,0]))    # down

        n = random.randint(0,len(actions)-1)
        path[t] = tuple(list(path[t-1] + np.array(actions[n])))
        
    return list(path)

def orientated_minotaur_path(start = np.array([0,0]), goal = np.array([]), T = 1):
    
    pos = start
    for _ in range(T):
        diff = goal-pos
        diff_idx = np.argmax(np.abs(diff))
        if diff_idx == 0 : # up or down
            pos += np.array([1,0])*np.sign(diff[diff_idx])
        else:   # left or right
            pos += np.array([0,1])*np.sign(diff[diff_idx])

        if np.max(pos) == 0: # we cannot stay
            pos = randomize_minotaur_path(start = pos, T = 1)[1]

    return tuple(pos)

def max_considering_the_minotaur(Q,env,minotaurs_coord,minotaurs_stay=False):

    actions = env.actions
    p = env.transition_probabilities
    map = {x[1]:x[0] for x in env.map.items()}

    l1_max = 2

    n_states = Q.shape[0] if Q.size/Q.shape[0] > 1 else 1

    V = np.zeros(shape=(n_states,))
    policy = np.zeros(shape=(n_states,))

    for s in range(n_states): # every state
        l1_norm = {l1:[] for l1 in range(l1_max+1)}

        for a in actions:
            if not any(p[:,s,a]):  # if there is not a future state
                continue
                
            if np.argmax(p[:,s,a]) == s and a != 0:   # if the state does not change (due to walls conflits) neither the action change
                continue

            dist = min(np.linalg.norm(np.array(map[s])+np.array(actions[a]) - np.array(minotaurs_coord),1),l1_max)
            
            l1_norm[int(dist)].append((Q[s,a],a) if Q.size/Q.shape[0] > 1 else (Q[a],a))


        if minotaurs_stay:
            l1_norm[1]+=l1_norm[0]
            l1_norm[0] = []

        for l in range(2,l1_max+1):
            l1_norm[0] += l1_norm.pop(l)

        for l1 in list(l1_norm.keys()):
             if l1_norm[l1]:
                V[s], policy[s] = max(shuffle(l1_norm[l1]), key=lambda x: [x[0]])
                break

    return V, policy

"""
p  - confidence of the interval
"""
def monte_carlo_method(p,start,maze,env,method,horizon=20,minotaurs_stay=False, life_mean = 30):

    N = round(10/(1-p)) if p<1 else p  # number of times MCM will work for a 95% confidance
    
    d_results = {0:0,1:0,2:0}
    l_results = []
    exit = ([np.where(maze == 2)[0][0],np.where(maze == 2)[1][0]])

    for _ in range(N):

        if method == 'DynProg':
            _, policy, minotaurs_path = dynamic_programming(env,horizon,minotaurs_stay)  
        else: 
            _, policy = value_iteration(env, 0.5, 0.7, minotaurs_stay);
            
        life0 = starting_life(life_mean)
        player_path, _ = env.simulate(start, policy, method, life0);
        if method == 'ValIter':
            minotaurs_path = randomize_minotaur_path(start = (6,5), maze_dimensions = maze.shape, T = len(player_path)-1, stay = False)

        r = path_result(player_path,minotaurs_path,tuple(exit))

        d_results[r] += 1
        
        l_results.append(r)

    return d_results, l_results
    
"""
Return Value
0 - The player succeeded to leave the maze
1 - The player was eaten by the minotaur
2 - Something rare happen...
"""
def path_result(player,minotaur,exit)->int:
    
    result = -1
    
    for t in range(len(player)):
        if(tuple(player[t-1])==exit and player[t]==exit and minotaur[t-1]!=exit):
            result = 0
            break
        elif (player[t]==minotaur[t]):
            result = 2
            break
    else:
        result = 1

    return result


maze = np.array([
    [0, 0, 1, 0, 0, 0, 0, 0],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 1, 0, 0, 1, 1, 1],
    [0, 0, 1, 0, 0, 1, 0, 0],
    [0, 0, 0, 0, 0, 0, 0, 0],
    [0, 1, 1, 1, 1, 1, 1, 0],
    [0, 0, 0, 0, 1, 2, 0, 0]
])

def f(v):
    count=0
    m = np.chararray(shape=maze.shape)
    sym = ['S','L','R','U','D']

    for y in range(maze.shape[0]):
        for x in range(maze.shape[1]):
            if maze[y,x]!=1:
                m[y,x] = sym[int(v[count])]
                count+=1
            else:
                m[y,x] = 'W'

    return m
    

starting_life = lambda mean: np.random.geometric(p=1/mean, size=1)[0]
shuffle = lambda vector : random.sample(list(vector),len(vector))

breakpoint



        # minotaurs_path.append(move)

env = Maze(maze)
epsilon = 0.0001
V, policy = value_iteration(env,29/30, epsilon, minotaurs_stay=True)
path, _ = env.simulate((0,0), policy, 'ValIter', life_0 = starting_life(30))
# animate_solution(maze,path,randomize_minotaur_path(start = (6,5), maze_dimensions = maze.shape, T = len(path)-1))


minotaurs_path = [(6,5)]
for agent in path:
    minotaurs_path.append(orientated_minotaur_path(start = minotaurs_path[-1], goal = np.array(agent), T = 1))


animate_solution(maze,path,minotaurs_path[:-1])