#!/usr/bin/env python3

import numpy as np


def n_players_to_proba(n_players):
    """
    Return an array of size M, K
    where arm m plays arm k with proba
    m, k
    """
    M = int(np.sum(n_players))
    K = n_players.shape[0]
    proba = np.zeros((M, K))
    index = 0
    for k in range(K):
        proba[index : index + int(n_players[k]), k] = 1
        index = index + int(n_players[k])
    return proba


def g(n, p):
    """Return the probability to have a collision for an arm with n players on it."""
    return n * p * (1 - p) ** n


def arm_assignment(rewards, M, p, init=None):
    """Solves the problem of optimality assigning player to arms when rewards are known.

    Parameters
    ----------
    rewards: np array of size K
        the expected reward of each time (taking collision into account)
    init: np array of size K or None
        If not None, starts the algorithm by the init
    """
    K = len(rewards)
    if init is None:
        init = np.zeros(K)
    mu = rewards
    n_players = np.copy(init)
    z = np.array([(1 - p) ** init[k] for k in range(K)])
    l = np.array([p / (1 - p) * init[k] for k in range(K)])
    M_remaining = M - np.sum(init)

    for m in range(int(M_remaining)):
        k_star = np.argmax(mu * z * (1 - l))
        z[k_star] = z[k_star] * (1 - p)
        l[k_star] = l[k_star] + p / (1 - p)
        n_players[k_star] += 1
    return n_players


def delta(n_noncollisions, T):
    """Compute the delta

    Parameters
    ----------
    n_noncollisions: array of shape K

    Return
    ------
    delta: np array of shape K
        delta for all arms
    """
    K = len(n_noncollisions)
    delta = np.zeros(K)
    delta[n_noncollisions == 0] = +np.inf
    delta[n_noncollisions != 0] = np.sqrt(
        np.log(2 * K**2 * T**2) / (2 * n_noncollisions[n_noncollisions != 0])
    )
    return delta
