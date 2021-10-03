# Exploring Munchausen Reinforcement Learning
This is the project repository of my team in the "Advanced Deep Learning for Robotics" course at [TUM](https://www.tum.de/en/). Our project's
topic is "Exploring Munchausen Reinforcement Learning" based on this [paper](https://arxiv.org/abs/2007.14430).

For a detailed discussion, see the [report](https://github.com/ketatam/Exploring-Munchausen-Reinforcement-Learning/blob/main/report_and_presentation/report.pdf) and the [final presentation](https://github.com/ketatam/Exploring-Munchausen-Reinforcement-Learning/blob/main/report_and_presentation/presentation.pdf).
### Setup
* Create a virtual environment.
* Run `pip3 install -r requirements.txt`
### Code Structure
This repository is structured as follows:
* The directories `M-DQN` and `M-SAC` contain the implementations of the RL agents DQN and SAC extended with the Munchausen
term, respectively.
  
* The directories `rl-baselines3-zoo` contains a copy of this [repository](https://github.com/DLR-RM/rl-baselines3-zoo),
where we included the implementations of M-DQN so that we can easily train and test the M-DQN agent and also compare it 
  to other classical agents. To do so, just follow the steps described in the original repository and insert `M-DQN`
  as the agent argument.
  
* The directory `particles-env`contains a modified version of this [repository](https://github.com/openai/multiagent-particle-envs).
The modified version contains code for a particles environment, where an agent wants to reach a goal, while avoiding
  obstacles. Besides, M-SAC agent is implemented and included in the code, so that it can be trained and compared to the
  classical SAC agent. 

* The directory `action-gap` contains implementation of callbacks for experiment manager of rl-baselines3-zoo which logs action-gap for tensorboard.