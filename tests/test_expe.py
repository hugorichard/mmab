#!/usr/bin/env python3
import numpy as np
from mmab.environment import rewards, sample_n_players
from mmab.utils import n_players_to_proba


def test_expe():

    M = 4  # Number of players
    K = 2  # Number of arms
    mu = np.array([0.01, 0.99])  # Reward of each arm
    p = 0.3  # Probability that a player is active at each round
    T = 3  # Number of rounds
    rng = np.random.RandomState(0)

    probas1 = []
    probas1.append(n_players_to_proba(np.array([1, 3])))
    probas1.append(n_players_to_proba(np.array([4, 0])))
    probas1.append(n_players_to_proba(np.array([4, 0])))
    probas1.append(n_players_to_proba(np.array([4, 0])))
    probas1.append(n_players_to_proba(np.array([4, 0])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))
    probas1.append(n_players_to_proba(np.array([2, 2])))

    probas2 = []
    probas2.append(n_players_to_proba(np.array([1, 3])))
    probas2.append(n_players_to_proba(np.array([0, 4])))
    probas2.append(n_players_to_proba(np.array([0, 4])))
    probas2.append(n_players_to_proba(np.array([0, 4])))
    probas2.append(n_players_to_proba(np.array([0, 4])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))
    probas2.append(n_players_to_proba(np.array([2, 2])))

    rss = []
    for probas in [probas1, probas2]:
        print("-----")
        rng = np.random.RandomState(0)
        rs = []
        for t in range(len(probas1)):
            proba = probas[t]
            n_players = sample_n_players(proba, rng)
            r = rewards(mu, n_players, p, rng)
            rs.append(np.sum(r))
        rss.append(rs)

    rss = np.array(rss)
    print(rss[0])
    print(rss[1])

    import matplotlib.pyplot as plt
    plt.plot(rss[1] - rss[0])
    plt.show()

    assert False
