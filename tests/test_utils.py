#!/usr/bin/env python3

from mmab.utils import n_players_to_proba
from mmab.environment import sample_n_players
import numpy as np


def test_n_players_to_proba():
    rng = np.random.RandomState(0)
    n_players = np.array([4, 5, 2])
    proba = n_players_to_proba(n_players)
    n_players_reco = sample_n_players(proba, rng)
    np.testing.assert_allclose(n_players_reco, n_players)
