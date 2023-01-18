#!/usr/bin/env python3
# Code for Greedy

import numpy as np
from mmab.utils import n_players_to_proba, arm_assignment, delta, g


class OPT:
    """OPT agent."""

    def __init__(self, M, K, p, T, mu, verbose=False):
        """OPT finds the optimal assignment thanks to the true means

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
        self.name = "OPT"
        self.K = K
        self.p = p
        self.T = T
        self.t_ = 1
        self.n_players_ = arm_assignment(mu, M, p)
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
        self.t_ = self.t_ + 1
