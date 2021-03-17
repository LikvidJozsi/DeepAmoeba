# Amoeba

The goal of this project is to use deep reinforcement learning to train an Amoeba (aka. TicTacToe) playing agent. The variant of the game
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
We also need to decide whether we want to have a view. The available views are ConsoleView and GraphicalView. A view makes it possible watch games. They can display only one game at at time so they are not useful during learning when hundreds of games are played simultaneously. Rather views should be used to get a sense of some games for a human to play.
```python
view = ConsoleView()
view = GraphicalView(map_size)
```
We need to decide on the agents used for learning. Currently the only agent that is usable as a learning agent is the NeuralNetwork agent, later on there may be more types of neural networks. A teaching agent should also be chosen, it can be the learning agent itself in which case we have self play, but it could be something else, for example a RandomAgent. Learning against multiple agents is not yet well integrated into the process.
```python
agent = NeuralNetwork(map_size)
random_agent = RandomAgent()
```
Now we need to create an AmoebaTrainer to orchestrate the learning. The RewardCalculator used during learning is configurable, for example PolicyGradients can use losses as deterrent learning examples. Currently the only RewardCalculator available is PolicyGradients.
```python
trainer = AmoebaTrainer(agent, random_agent,
                               reward_calculator=PolicyGradients(teach_with_losses=False))
```
Now all we need is start training, and the progress will be logged into the console.
```python
games_played_per_episode = 1000
trainer.train(games_played_per_episode, map_size=map_size, view=None,
                  num_episodes=5)
```

