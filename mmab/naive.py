#!/usr/bin/env python3

from mmab.utils import n_players_to_proba


class Naive:
    """NAIVE agent."""

    def __init__(self, n_players):
        """Init parameters."""
        self.n_players = n_players

    def play(self):
        """Play a round."""
        return n_players_to_proba(self.n_players)

    def update(self, r, c):
        pass


class Uniform:
    """NAIVE agent."""

    def __init__(self, proba):
        """Init parameters."""
        self.proba = proba

    def play(self):
        """Play a round."""
        return self.proba

    def update(self, r, c):
        """Update params."""
        pass
