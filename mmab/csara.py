#!/usr/bin/env python3
# Code for centralized SARA

import numpy as np


def delta(t, K, T, M):
    return np.sqrt(
        1 / 2 * (1 / t + 1 / t**2) * np.log(K * T * M * np.sqrt(t / 4 + 1))
    )


def h(mu, M, p):
    return mu * (1 - p) ** (M - 1) * (1 - (1 + M) * p)


def l(mu, M, p):
    return mu * (1 - p) ** (M - 2) * (1 - M * p)


def argmax(f, K):
    max = -np.inf
    argmax = None
    for k in range(K):
        for k2 in range(K):
            if k == k2:
                continue
            if f(k, k2) > max:
                max = f(k, k2)
                argmax = (k, k2)
    return argmax


class CSARA:
    """CSARA agent."""

    def __init__(self, M, K, p, T):
        """Init parameters."""
        self.M = M
        self.K = K
        self.p = p
        self.T = T
        self.mean_weight_ = np.zeros(K)
        self.mean_r_ = np.zeros(K)
        self.t_ = 1
        self.delta_ = None
        n_players = np.zeros(K)
        for k in range(K - 1):
            n_players[k] = int(M / K)
        n_players[K - 1] = M - int(M / K) * (K - 1)
        self.n_players_ = n_players

    def play(self):
        """Play a round."""
        return self.n_players_

    def update(self, r):
        """Update parameters."""
        t = self.t_
        K = self.K
        p = self.p
        M = self.M
        T = self.T

        mean_r = self.mean_r_
        mean_weight = self.mean_weight_
        n_players = self.n_players_
        if t == 1:
            mean_r = r
        else:
            mean_r = (mean_r * (t - 1) + r) / t
        self.mean_r_ = mean_r

        weight = np.zeros(K)
        for k in range(K):
            weight[k] = n_players[k] * (1 - p) ** (n_players[k] - 1) * p

        if t == 1:
            mean_weight = weight
        else:
            mean_weight = (mean_weight * (t - 1) + weight) / t
        self.mean_weight_ = mean_weight

        mu_h = (mean_r + delta(t, K, T, M)) / mean_weight
        mu_l = (mean_r - delta(t, K, T, M)) / mean_weight
        mu_h = np.minimum(mu_h, 1)
        mu_l = np.maximum(mu_l, 0)

        self.delta_ = delta(t, K, T, M)
        self.mu_h_ = mu_h
        self.mu_l_ = mu_l

        def criterion(k, k2):
            return h(mu_l[k], n_players[k], p) - l(mu_h[k2], n_players[k2], p)

        k, k2 = argmax(criterion, K)

        if criterion(k, k2) > 0:
            n_players[k] += 1
            n_players[k2] -= 1

        self.n_players_ = n_players
        self.t_ = self.t_ + 1
