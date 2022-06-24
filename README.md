# Project 2 (Continuous control) for Udacity Deep Reinforcement Learning Nanodegree

The project 2 solution for Udacity Deep Reinforcement Learning nano degree.
![trained agents](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/agents%20illustration.gif)


# Run the code

* 1. download this repository
* 2. install the requirements in a separate Anaconda environment: `pip install -r requirements.txt`
* 3. run the solution file [*Continuous_Control.ipynb*](https://github.com/hynpu/drlnd_p2_reacher/blob/main/Continuous_Control.ipynb)


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

  
# Techniques to improve the speed and convergence
  
* Use 20 agents to improve training speed: "Often, cooperation among multiple RL agents is much more critical: multiple agents must collaborate to complete a common goal, expedite learning, protect privacy, offer resiliency against failures and adversarial attacks, and overcome the physical limitations of a single RL agent behaving alone.", the related discussions can be found in this repo: [Udacity discuss channel](https://knowledge.udacity.com/questions/281228)

* Adjust the OU noise by adding decreasing factors, and related discussions can be found in this repo: [Udacity discuss channel](https://knowledge.udacity.com/questions/25366)

* Change different discount factor GAMMA to see the performance. The agent does not need to see too far to predict its next movement. So slightly reduce the GAMMA value to focus more on the current states.

# Approach

## 


The high level structure shows as the following:

![DDPG illustration](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/ddpg%20illustration.png)

# Results:

The average rewards along with the traning process show as following:

![Results](https://github.com/hynpu/drlnd_p2_reacher/blob/main/images/episode-rewards.png)
