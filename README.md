# mmab
Multiplayer multiarm bandits

## Install
All commands are to be run from the unzipped directory unless specified otherwise.

### Create virtual environment
``` python
conda create --name mmab pip
conda activate mmab
```

### Requirements
``` bash
pip install -r requirements.txt
```

For the core algorithm
* numpy

For plotting and running experiments
* matplotlib
* joblib

For running tests
* pytest

### Install MMAB
``` bash
pip install -e .
```

Run tests
``` bash
pytest
```


## Experiments

### Reproducing Figure 1: Benchmark of ETC, UCB and Cautious Greedy on synthetic data with $\nu^* = 0$ (left) and $\nu^* = 1$ (right) 

Move into the `experiments` directory and run the benchmark:

`python experiment_aistats2024.py`

The pdf is then available in the `figure` directory under the name "figure1.pdf"

### Reproducing Figure 2: same experiment as Figure 1 with larger range of values for the horizon $T$

Move into the `experiments` directory and run the benchmark:

`python rebuttal.py`

The pdf is then available in the `figure` directory under the name "figure1_rebuttal.pdf"

