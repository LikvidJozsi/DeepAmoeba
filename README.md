# Amoeba

This project is an implementation of the AlphaZero algorithm that focuses on performance using a compact CNN. This is a non-trivial
goal, since making the
CNN smaller puts more relative load on the CPU side. The project utilizes multiprocessing, vectorization on multiple levels, 
execution of thousands of games in parallel, python code compilation (numba and cython) and more.

The project started as
a [Master's thesis](https://drive.google.com/file/d/1ixjSL4YSgY6uNmwSPVfP26RlXGydukxL/view?usp=sharing) and transitioned
into a passion project.

# What is this useful for?

The goal is making experimenting with the AlphaZero algorithm easier by making it possible to do meaningful training on a
midrange PC in reasonable time. Of course, this requires the usage of smaller networks than the original 20 block resnet of AlphaZero.
But simply doing that won't yield satisfactory results, since as the GPU side speeds up, it needs more work from the CPU side to
keep it fed with data. Unfortunately, the CPU side of the computation is not trivial, since the MCTS algorithm requires several
evaluations of the UCB formula for each search. This means the CPU can easily become a bottleneck. This project solves that
problem by providing several optimizations/features:
- the board positions are fed to the gpu in large batches (potentially thousands at a time)
- to fill the GPU inference batch, hundreds of games are played in parallel
- these games can be distributed among any number of CPU processes (MCTS workers)
- collecting and sending the batches to the GPU for inference can bottleneck a single process, so MCTS workers are assigned to multiple inference workers, each with their on tensorflow sessions
- given all this, the CPU side can still be sluggish because of the speed limitations of python. To improve this, UCB calculation is compiled using the Numba library and other critical parts of the code are compiled using Cython
- on top of this, the project implements a training loop with many features (handling and sampling training data, providing many evaluation options)
- this is all in the languange of datascience: python, making experimentation and extension easy and fast.r

Limitations:
- The project was designed with eventual game agnosticity in mind, but for now, it only works with Gomoku.
- The documentation is lacking (nonexistent), but at least the code is designed with readability in mind (mostly).

# How to install

Installation was tested using python 3.9 on windows 10 (don't judge):
1. clone repository
2. install packages in requirements.txt (gpu support for tensorflow is highly recommended so also install CUDA and CuDNN )
3. run "python setup.py build_ext --inplace" to compile the cython parts of the code
4. run Train.py in the Scripts folder. If it runs, installation is successful.

# How to use

The scripts folder contains the most common usecases. Most notably:
- GenerateDataset.py: It creates a dataset using simple MCTS, no neural networks involved. Useful before as inital startup training data
- PreTrain.py: uses a generated starting dataset to train a CNN. Useful for finetuning network training hyperparameters and giving the training loop a head-start
- Train.py: This is the main self-play focused reinforecement training loop. Can be started either with a fresh network, or one produced be PreTraining.py
- PrintTrainingLogs.py: Print several graphs showing metrics collected during the main training loop, organized be episode
