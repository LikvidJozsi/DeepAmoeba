# Amoeba

The goal of this project is to use deep reinforcement learning to train an Amoeba (aka. TicTacToea and Gomoku) playing agent. The variant of the game
used is one with an unbounded map with the goal of getting a five long continous amoeba. Initially instead of an unbounded one,
a sufficiently large map is used, since unbounded maps come with extra complexity for the agents.

# Description

AmoebaTrainer is the orchestrator of learning, the training is done in its train function. Learning is done in episodes, one episode consists of  the following steps:
- playing a certain number of games against one agent or many, using the GameGroup class
- extracting supervised training samples from these games using a RewardCalculator
- training the learning agent using these samples
- evaluating the new agent version using an Evaluator. This Evaluator may calculate many metrics, for example winrate against certain other agents, or the [Élő score](https://en.wikipedia.org/wiki/Elo_rating_system) of the agent providing a scalar value describing the performance of the system.

# Usage

To start a basic learning process we need to decide on some basic parameters:

```python
map_size = (8, 8)
```
 
 TODO replace outdated tutorial

