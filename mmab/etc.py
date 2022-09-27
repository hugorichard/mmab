#!/usr/bin/env python3

from mmab.utils import n_players_to_proba, arm_assignment
import numpy as np


class ETC:
    """ETC agent."""

    def __init__(self, M, K, p, T, verbose=False):
        """Init parameters."""
        self.M = M
        self.K = K
        self.p = p
        self.T = T
        self.t_ = 1
        self.mean_r_ = np.zeros(K)
        self.commit = False
        c = M * p / K * (1 - p / K) ** (M - 1)
        tau = np.sqrt(np.log(2 * T) / 2) / c * 2 * T
        self.tau = tau ** (2 / 3)
        self.delta_ = None
        self.proba_ = np.ones((M, K)) / K
        self.verbose = verbose

    def play(self):
        """Play a round."""
        return self.proba_

    def update(self, r, c):
        """Update parameters."""
        t = self.t_
        K = self.K
        p = self.p
        M = self.M
        tau = self.tau

        if self.commit is False:
            mean_r = self.mean_r_
            if t == 1:
                mean_r = r
            else:
                mean_r = (mean_r * (t - 1) + r) / t
            self.mean_r_ = mean_r

        if t > tau and self.commit is False:
            self.commit = True
            mu = self.mean_r_ / M * p / K * (1 - p / K) ** (M - 1)
            n_players = arm_assignment(mu, M, p)

            self.proba_ = n_players_to_proba(n_players)
        self.t_ = self.t_ + 1
