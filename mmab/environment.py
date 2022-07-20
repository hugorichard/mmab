#!/usr/bin/env python3
# Environment to play the multiplayer bandit game

import numpy as np


def reward(mu, M, p, rng):
    """Sample a reward.

    Parameters
    ----------
    mu : float
        Mean reward
    M : int
        Number of players
    p : float
        Probability of a player being active
    rng : Random State
        Random state

    Return
    --------
    reward: int
        Reward
    """
    if M == 0:
        return 0
    X = rng.binomial(1, mu)
    onlyone = rng.binomial(1, p, size=int(M))
    onlyone = np.sum(onlyone) == 1
    if onlyone:
        return X
    else:
        return 0


def rewards(mu, n_players, p, rng):
    """Sample rewards for all arm.

    Parameters
    ----------
    mu : array of size K
        Mean reward
    n_players : array of size K
        Number of players per arm
    p : float
        Probability of a player being active
    rng : Random State
        Random state

    Return
    --------

    rewards: array of size K
        Rewards

    """
    K = len(mu)
    r = np.zeros(K)
    for k in range(K):
        r[k] = reward(mu[k], n_players[k], p, rng)
    return r
