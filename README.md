# mmab
Multiplayer multiarm bandits

# Install
All commands are to be run from the unzipped directory unless specified otherwise.

## Create virtual environment
``` python
conda create --name mmab pip
conda activate mmab
```

## Requirements
``` python
pip install -r requirements.txt
```

For the core algorithm
* numpy

For plotting and running experiments
* matplotlib
* joblib

For running tests
* pytest

## Install MMAB
``` python
pip install -e .
```

Run tests
``` python
pytest
```


## Experiments

### Reproducing Figure 1: Benchmark of ETC, UCB and Cautious Cautious Greedy on synthetic data with $\nu^* = 0$ (left) and $\nu^* = 1$ (right) 

Move into the `experiments` directory

`cd experiments`

Run the benchmark

`python 0_players.py` (Computation time: 22 seconds)

`python 0_players.py` (22 seconds)

Go back to the root directory `cd ..`, move into the plotting directory 

`cd plotting`

and plot the data
