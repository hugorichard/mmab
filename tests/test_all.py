#!/usr/bin/env python3

from mmab.csara import CSARA, argmax
from mmab.environment import rewards, reward
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)


M = 100
K = 2
mu = np.array([0.1, 0.9])
p = 0.02
T = 1000


def test_argmax():
    K = 10

    def f(a, b):
        return a + b

    assert argmax(f, K) == (8, 9)


def test_agent():
    agent = CSARA(M, K, p, T)
    rs = []
    N = []
    for t in range(T):
        n_players = agent.play()
        r = rewards(mu, n_players, p, rng)
        agent.update(r)
        rs.append(r)



    assert False
