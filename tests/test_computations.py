#!/usr/bin/env python3

from mmab.environment import rewards
from mmab.csara import h, l
import numpy as np


def utility(mu, M, p):
    return mu * M * (1 - p) ** (M - 1) * p


def utilities(mu, n_players, p):
    u = []
    for k in range(len(mu)):
        u.append(utility(mu[k], n_players[k], p))
    return np.array(u)


def test_utility():
    mu = np.array([0.1, 0.9])
    n_players = np.array([2, 3])
    p = 0.15
    rng = np.random.RandomState(0)
    rs = []
    for _ in range(20000):
        r, _ = rewards(mu, n_players, p, rng)
        rs.append(r)
    means = np.mean(rs, axis=0)
    np.testing.assert_array_almost_equal(means, utilities(mu, n_players, p), 2)


def test_criterion():
    mu = np.array([0.1, 0.9])
    n_players = np.array([2, 3])
    n_players2 = np.array([1, 4])
    p = 0.15
    u = np.sum(utilities(mu, n_players, p))
    u2 = np.sum(utilities(mu, n_players2, p))
    diff = h(mu[1], n_players[1], p) - l(mu[0], n_players[0], p)
    np.testing.assert_allclose(u2, u + p * diff)
