#!/usr/bin/env python3
# Code for Greedy

import numpy as np
from mmab.utils import n_players_to_proba, arm_assignment, delta, g


class Greedy:
    """Greedy agent."""

    def __init__(self, M, K, p, T, verbose=False):
        """Greedy finds the optimal assignment thanks to estimated means but constrains
        certain arms to have at least one player for exploration purposes.

        Parameters
        ----------
        M : int
            Number of players
        K : int
            Number of arms
        p : float
            activation probability
        T : int
            Horizon

        Attributes:
        -----------
        t_: int
            current timestep
        mean_r_ : np array of size K
            current reward
        n_collisions_ : np array of size K
            number of collisions
        """

        self.M = M
        self.name = "Greedy"
        self.K = K
        self.p = p
        self.T = T
        self.t_ = 1
        self.delta_ = None
        # Fair initialisatopn
        n_players = np.zeros(K)
        for k in range(K - 1):
            n_players[k] = int(M / K)
        n_players[K - 1] = M - int(M / K) * (K - 1)
        self.n_players_ = n_players
        self.r_sum_ = np.zeros(K)  # sum of rewards
        self.n_noncollisions_ = np.zeros(K)  # sum of noncollisions
        self.verbose = verbose

    def play(self):
        """Play a round."""
        return n_players_to_proba(self.n_players_)

    def update(self, r, c):
        """Update parameters.

        Parameters
        ----------
        r : array of size K
            Array of received reward
        c : array of size K
            Array of non-collisions

        """
        p = self.p
        M = self.M
        T = self.T
        K = len(r)

        r_sum = self.r_sum_
        n_noncollisions = self.n_noncollisions_

        r_sum += r
        n_noncollisions += c
        mean_r = np.zeros(K)
        I = n_noncollisions != 0
        mean_r[I] = r_sum[I] / n_noncollisions[I]
        upper_bounds = np.minimum(mean_r + delta(n_noncollisions, T), 1)
        lower_bounds = np.maximum(mean_r - delta(n_noncollisions, T), 0)

        init = np.ones(K)
        M_L = arm_assignment(lower_bounds, M, p)
        M_H = arm_assignment(upper_bounds, M, p, init=init)
        I = np.argsort(upper_bounds)
        i = 0

        while g(M_L, p).dot(lower_bounds) > g(M_H, p).dot(upper_bounds):
            init[I[i]] = 0
            M_L = arm_assignment(lower_bounds, M, p)
            M_H = arm_assignment(upper_bounds, M, p, init=init)
            i += 1

        self.n_players_ = arm_assignment(mean_r, M, p, init=init)
        # print(self.t_, "Greedy", mean_r, init, self.n_players_)
        self.t_ = self.t_ + 1
        self.r_sum_ = r_sum
        self.n_noncollisions_ = n_noncollisions
