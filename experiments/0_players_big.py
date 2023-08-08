from mmab.csara import CSARA
from mmab.ucb import UCB, delta
from mmab.opt import OPT
from mmab.greedy import Greedy
from mmab.etc import ETC
from mmab.environment import rewards, deterministic_reward, sample_n_players
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


M = 10  # Number of players
mu = np.array([0.9, 0.8, 0.8, 0])  # Reward of each arm
K = len(mu)  # Number of arms
p = 0.1  # Probability that a player is active at each round
T = int(1e5)  # Number of rounds

verbose = True

names_agents = [
    ("OPT", OPT(M, K, p, T, mu,)),
    ("UCB", UCB(M, K, p, T, verbose=True)),
    ("Greedy", Greedy(M, K, p, T, verbose=True)),
]
agents = [names_agents[i][1] for i in range(len(names_agents))]
names = [names_agents[i][0] for i in range(len(names_agents))]


def do_exp(seed):
    rss = []
    for name, agent in names_agents:
        rng = np.random.RandomState(seed)
        rs = []
        for t in range(T):
            proba = agent.play()
            n_players = sample_n_players(proba, rng)
            d_rewards = deterministic_reward(mu, n_players, p)
            r, c = rewards(mu, n_players, p, rng)
            agent.update(r, c)
            rs.append(np.sum(d_rewards))
            if t == 0 and name == "OPT" and seed==0:
                print(t, name, n_players)
            if t % (T-1) == 0 and seed==0:
                print(t, name, n_players)
        rss.append(rs)
    rss = np.array(rss)
    return rss

seeds = 50
rss = Parallel(n_jobs=-1, verbose=100)(delayed(do_exp)(seed) for seed in range(seeds))
rss = np.array(rss)
rss = [np.cumsum(rss[seed, 0] - rss[seed], axis=1) for seed in range(seeds)]

rss_median = np.median(rss, axis=0)
rss_high = np.quantile(rss, 0.9, axis=0)
rss_low = np.quantile(rss, 0.1, axis=0)

np.save("../data/0_players_big_rss", rss)
np.save("../data/0_players_big_rss_median", rss_median)
np.save("../data/0_players_big_rss_high", rss_high)
np.save("../data/0_players_big_rss_low", rss_low)