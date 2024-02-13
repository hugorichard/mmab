#!/usr/bin/env python3
# Code for Cautious Greedy

import numpy as np
from mmab.utils import n_players_to_proba, arm_assignment, delta, g


class CautiousGreedy:
    """Cautious Greedy agent."""

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
        self.name = "Cautious Greedy"
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
        self.nu = 0
        self.active = np.ones(K)
        self.accept = np.zeros(K)
        self.underpressure = np.zeros(K)
        self.rr_n = 0
        self.rr = np.ones(K) # arms currently selected in the RR
        self.statistics = (np.zeros(K), np.zeros(K), np.ones(K))
        self.n_underpressure = 9


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

        self.underpressure = rr(np.maximum(self.active - self.accept, 0), self.rr_n, np.sum(self.underpressure == 1))
        E = np.maximum(self.active - self.underpressure, 0)
        lower_bounds, means, upper_bounds = self.statistics
        self.n_players_ = arm_assignment(means, M, p, init=E)
        if self.rr_n == len(np.maximum(self.active - self.accept, 0)):
            self.rr_n = 0
            M_L = arm_assignment(lower_bounds, M, p)
            I = np.argsort(upper_bounds)
            init = np.zeros(K)
            init[I[:self.nu]] = 1
            M_H = arm_assignment(upper_bounds, M, p, init=init)

            while g(M_L, p).dot(lower_bounds) > g(M_H, p).dot(upper_bounds):
                self.nu += 1
                init = np.zeros(K)
                init[I[:self.nu]] = 1
                M_L = arm_assignment(lower_bounds, M, p)
                M_H = arm_assignment(upper_bounds, M, p, init=init)

            accept = np.zeros(K)
            for k in range(K):
                if lower_bounds[k] > np.sort(upper_bounds)[self.nu + 1]:
                    accept[k] = 1

            active = np.ones(K)
            for k in range(K):
                if upper_bounds[k] < np.sort(lower_bounds)[self.nu + 1]:
                    active[k] = 0

            self.accept = accept
            self.active = active
            self.n_underpressure = self.nu - (K - len(active))

        if self.t_ == 2 ** (np.floor(np.log2(self.t_))):
            mean_r = np.zeros(K)
            I = n_noncollisions != 0
            mean_r[I] = r_sum[I] / n_noncollisions[I]
            upper_bounds = np.minimum(mean_r + delta(n_noncollisions, T), 1)
            lower_bounds = np.maximum(mean_r - delta(n_noncollisions, T), 0)
            self.statistics = (lower_bounds, mean_r, upper_bounds)

        self.rr_n += 1


def rr(A, m, l):
    iA = np.arange(len(A))[A == 1]
    n = m % len(iA)
    if n + l >= len(iA):
        iU = iA[:int((n + l) % len(iA))].tolist()
        iU += iA[int(n % len(iA)):].tolist()
    else:
        iU = iA[n:n+l]
    U = np.zeros(len(A))
    U[iU] = 1
    return U.astype(int)


