# Amoeba

This project is an implementation of AlphaZero that focuses on performance using a compact CNN. This is a non-trivial
goal, since making the
CNN smaller puts more relative load on the CPU side. The project utilizes heterogeneous multiprocessing(MCTS and GPU
inference processes),
massive vectorization on multiple levels, execution of thousands of games in parallel, python code compilation (numba
and cython)
and more.

The project started as
a [Master's thesis](https://drive.google.com/file/d/1ixjSL4YSgY6uNmwSPVfP26RlXGydukxL/view?usp=sharing) and transitioned
into a passion project.

# How to install

Installation was tested using python 3.9 on windows 10 (don't judge):
1. clone repository
2. install packages in requirements.txt (gpu support for tensorflow is highly recommended so also install CUDA and CuDNN )
3. run "python setup.py build_ext --inplace" to compile the cython parts of the code
4. run Train.py in the Scripts folder. If it runs, installation is successful.
