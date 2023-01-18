#!/usr/bin/env python3
# Environment to play the multiplayer bandit game

import numpy as np
from mmab.utils import g


def deterministic_reward(mu, M, p):
    """Give the mean reward"""
    return g(M, p).dot(mu)

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
    noncollision: int
        Returns 0 if no collision 1 otherwise
    """

    X = rng.binomial(1, mu)
    if M == 0:
        return (0, 0)
    onlyone = rng.binomial(1, p, size=int(M))
    onlyone = np.sum(onlyone) == 1
    if onlyone:
        return (X, 1)
    else:
        return (0, 0)


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

    collisions: array of size K
        Collisions 0 if no collision 1 otherwise

    """
    K = len(mu)
    r = np.zeros(K)
    c = np.zeros(K)
    for k in range(K):
        r[k], c[k] = reward(mu[k], n_players[k], p, rng)
    return r, c


def sample_n_players(proba, rng):
    M, K = proba.shape
    n_players = np.zeros(K)

    for i in range(M):
        p = rng.rand()
        cum = np.cumsum(proba[i])
        n_players[np.argmax(p < cum)] += 1

    return n_players
