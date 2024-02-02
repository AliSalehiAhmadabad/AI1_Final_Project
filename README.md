# Artificial Intelligence for Robotics Final Project
Project Title: Q-Learning for FrozenLake Environment

Author: Ali Salehi Ahmadabad

Student Number: S6110138

Related Professor: Mr. Armando Tacchella

Feb 2024

--------------------------------------------------------------------------

## Introduction

This repository contains a Python implementation of the Q-learning algorithm applied to the FrozenLake environment using OpenAI Gym. Q-learning is a reinforcement learning technique used to find an optimal policy for an agent interacting with an environment.

## Implementation Details

### Environment

The FrozenLake environment from the OpenAI Gym library represents a grid-world where an agent navigates from a starting position to the goal while avoiding holes in the ice.

### Q-Learning Algorithm

The Q-learning algorithm updates Q-values, representing the expected cumulative rewards for taking a specific action in a particular state. Key components include:

- **Epsilon-Greedy Policy:** A strategy for selecting actions, balancing exploration and exploitation.

- **Q-Value Update:** Values are updated based on the temporal difference error, the difference between expected and actual rewards.

- **Hyperparameters:** The learning rate (`alpha`), discount factor (`gamma`), and exploration rate (`epsilon`) were fine-tuned.

## Results

The algorithm was trained over a specified number of episodes, and the average reward was evaluated periodically. Results demonstrate the effectiveness of the learned policy.

### Training

The algorithm was trained for a total of `num_episodes` episodes. A progress bar monitored the learning process, updating the average reward every 1000 episodes.

### Evaluation

After training, the learned Q-values were used to assess the agent's performance. The average reward was calculated over a set number of evaluation episodes.

## Conclusion

In conclusion, the Q-learning algorithm successfully learned a policy for the FrozenLake environment, allowing the agent to navigate the grid-world and reach the goal while avoiding hazards. The choice of hyperparameters significantly influenced performance, highlighting the importance of tuning.



