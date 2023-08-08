from mmab.csara import CSARA
from mmab.ucb import UCB, delta
from mmab.opt import OPT
from mmab.greedy import Greedy
from mmab.etc import ETC
from mmab.environment import rewards, deterministic_reward, sample_n_players
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os


M = 3  # Number of players
mu = np.array([0.99, 0.01])  # Reward of each arm
K = len(mu)  # Number of arms
p = 0.1  # Probability that a player is active at each round
T = int(1e9)  # Number of rounds

names_agents = [
    ("OPT", OPT(M, K, p, T, mu,)),
    ("UCB", UCB(M, K, p, T, verbose=True)),
    ("Greedy", Greedy(M, K, p, T, verbose=True)),
]
agents = [names_agents[i][1] for i in range(len(names_agents))]
names = [names_agents[i][0] for i in range(len(names_agents))]
seeds = 50

for name in names:
    for seed in range(seeds):
        os.system("rm ../data/0_players_%s_%s.csv" % (name, seed))



def do_exp(seed, name):
    rng = np.random.RandomState(seed)
    if name == "OPT":
        agent = OPT(M, K, p, T, mu)
    elif name == "UCB":
        agent =  UCB(M, K, p, T)
    elif name == "Greedy":
        agent =  Greedy(M, K, p, T)

    file = open("../data/0_players_%s_%s.csv" % (name, seed), "w")
    file.write("")
    file.close()

    file = open("../data/0_players_%s_%s.csv" % (name, seed), "w")
    for t in range(T):
        proba = agent.play()
        n_players = sample_n_players(proba, rng)
        d_rewards = deterministic_reward(mu, n_players, p)
        r, c = rewards(mu, n_players, p, rng)
        agent.update(r, c)
        reward = np.sum(d_rewards)
        file.writelines(["%f, %f\n" % (t, reward)])
    file.close()

Parallel(n_jobs=1000, backend="threading", verbose=100)(delayed(do_exp)(seed, name) for seed in range(seeds) for name in names)