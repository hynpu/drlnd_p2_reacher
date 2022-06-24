# Project 2 (Continuous control) for Udacity Deep Reinforcement Learning Nanodegree

The project 2 solution for Udacity Deep Reinforcement Learning nano degree.
![trained agents](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/agents%20illustration.gif)


# Run the code

* 1. download this repository
* 2. install the requirements in a separate Anaconda environment: `pip install -r requirements.txt`
* 3. run the solution file [**Continuous_Control.ipynb**](https://github.com/hynpu/drlnd_p2_reacher/blob/main/Continuous_Control.ipynb)


# Goal
In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible. The following two success cretiria have been considered.

* Option 1: Solve the First Version

  The task is episodic, and in order to solve the environment, your agent must get an average score of +30 over 100 consecutive episodes.

* Option 2: Solve the Second Version

  The barrier for solving the second version of the environment is slightly different, to take into account the presence of many agents. In particular, your agents must get an average score of +30 (over 100 consecutive episodes, and over all agents). Specifically,

  * After each episode, we add up the rewards that each agent received (without discounting), to get a score for each agent. This yields 20 (potentially different) scores. We then take the average of these 20 scores.
  * This yields an average score for each episode (where the average is over all 20 agents).
  
**In this project, the second option is used for success creteria. The main reason is that the single agent training is slower than the 20 agents training in terms of experience collection and reduce correlations**

# DDPG in detail

This Youtube video explained DDPG in a very clean way, and it is highly recommend to watch through the video and get some basic understanding of DDPG: 

[![DDPG youtube video](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/youtube%20link.PNG)](https://www.youtube.com/watch?v=oydExwuuUCw)

Deep Deterministic Policy Gradient (DDPG) is an algorithm which concurrently learns a Q-function and a policy. It uses off-policy data and the Bellman equation to learn the Q-function, and uses the Q-function to learn the policy. A high-level DDPG structure looks the following, and you can see it has some DQN features like the replay buffer, critic network and so on. As mentioned earlier: computing the maximum over actions in the target is a challenge in continuous action spaces. DDPG deals with this by using a target policy network to compute an action which approximately maximizes $Q_{\phi_{\text{targ}}}$. The target policy network is found the same way as the target Q-function: by polyak averaging the policy parameters over the course of training.

Putting it all together, Q-learning in DDPG is performed by minimizing the following MSBE loss with stochastic gradient descent:

![DDPG illustration](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/ddpg%20eqn.png)

The below image shows the compasiron between DDPG and DQN. 

![DDPG vs DQN](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/dqn-ddpg.png)


# Approach

The high level structure shows as the following, and the code under [**ddpg_agent.py**](https://github.com/hynpu/drlnd_p2_reacher/blob/main/ddpg_agent.py) follows the diagram:

![DDPG illustration](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/ddpg%20illustration.png)

## 1. The state and action space of this environment

We can set up the environment (20 agents) with the following code, and the reward of +0.1 is provided for each step that the agent's hand is in the goal location. The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

    # reset the environment
    env_info = env.reset(train_mode=True)[brain_name]

    # number of agents
    num_agents = len(env_info.agents)
    print('Number of agents:', num_agents)

    # size of each action
    action_size = brain.vector_action_space_size
    print('Size of each action:', action_size)

    # examine the state space 
    states = env_info.vector_observations
    state_size = states.shape[1]
    print('There are {} agents. Each observes a state with length: {}'.format(states.shape[0], state_size))
    print('The state for the first agent looks like:', states[0])
    
## 2. Explore the environment by taking random actions

We then take some random actions based on the environment we created just now, and see how the agents perform (apparently it will be bad without learning)

    env_info = env.reset(train_mode=False)[brain_name]     # reset the environment    
    states = env_info.vector_observations                  # get the current state (for each agent)
    scores = np.zeros(num_agents)                          # initialize the score (for each agent)

    while True:
        actions = np.random.randn(num_agents, action_size) # select an action (for each agent)
        actions = np.clip(actions, -1, 1)                  # all actions between -1 and 1
        env_info = env.step(actions)[brain_name]           # send all actions to tne environment
        next_states = env_info.vector_observations         # get next state (for each agent)
        rewards = env_info.rewards                         # get reward (for each agent)
        dones = env_info.local_done                        # see if episode finished
        scores += env_info.rewards                         # update the score (for each agent)
        states = next_states                               # roll over states to next time step
        if np.any(dones):                                  # exit loop if episode finished
            break
    print('Total score (averaged over agents) this episode: {}'.format(np.mean(scores)))

## 3. Implement the DDPG algo to train the agent

The last step is to implement the DDPG algorithm to trian the agents. The code can be found in [**ddpg_agent.py**](https://github.com/hynpu/drlnd_p2_reacher/blob/main/ddpg_agent.py). However, I would like to mention several techniques to improve the speed and convergence:
  
* Use 20 agents to improve training speed: "Often, cooperation among multiple RL agents is much more critical: multiple agents must collaborate to complete a common goal, expedite learning, protect privacy, offer resiliency against failures and adversarial attacks, and overcome the physical limitations of a single RL agent behaving alone.", the related discussions can be found in this repo: [Udacity discuss channel](https://knowledge.udacity.com/questions/281228)

* Adjust the OU noise by adding decreasing factors, and related discussions can be found in this repo: [Udacity discuss channel](https://knowledge.udacity.com/questions/25366)

* Change different discount factor GAMMA to see the performance. The agent does not need to see too far to predict its next movement. So slightly reduce the GAMMA value to focus more on the current states.


# Results:

The average rewards along with the traning process show as following:

![Results](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/episode-rewards.png)
