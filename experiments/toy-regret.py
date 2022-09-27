from mmab.csara import CSARA
from mmab.ucb import UCB, delta
from mmab.naive import Naive, Uniform
from mmab.greedy import Greedy
from mmab.etc import ETC
from mmab.environment import rewards, sample_n_players
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


M = 4  # Number of players
K = 2  # Number of arms
mu = np.array([0.01, 0.99])  # Reward of each arm
p = 0.3  # Probability that a player is active at each round
T = 10  # Number of rounds

names_agents = [
    ("3 on the best arm", Naive(np.array([1, 3]))),
    ("ETC", ETC(M, K, p, T)),
    # ("CSARA", CSARA(M, K, p, T, init="fair")),
    ("UCB", UCB(M, K, p, T)),
    ("Greedy", Greedy(M, K, p, T)),
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
            r, c = rewards(mu, n_players, p, rng)
            if t < 10  and name == "UCB":
                print(name, t, n_players)
                print(r, c)
            if 10 < t < 70 and t % 10 == 0  and name == "UCB":
                print(name, t, n_players)
            if 2300 > t > 70 and t % 100 == 0  and name == "UCB":
                print(name, t, n_players)
                print(r, c)

            agent.update(r, c)
            rs.append(np.sum(r))
        rss.append(rs)
    rss = np.array(rss)
    return rss

print(do_exp(0))

# rss = Parallel(n_jobs=10, verbose=True)(delayed(do_exp)(seed) for seed in range(50))
# rss = np.array(rss)
# rss = [np.cumsum(rss[seed, 0] - rss[seed], axis=1) for seed in range(50)]

# rss_median = np.median(rss, axis=0)
# rss_high = np.quantile(rss, 0.9, axis=0)
# rss_low = np.quantile(rss, 0.1, axis=0)


# %%

print(rss[:, -1, :])

# %%


cm = plt.cm.Set1
COLORS = {
    "ETC": cm(2),
    "CSARA": cm(3),
    "4 on the best arm": cm(1),
    "3 on the worse arm": cm(0),
    "Uniform": cm(4),
    "UCB": cm(6),
    "Greedy": cm(7),
}

plt.figure(figsize=(5, 4))
for i, name in enumerate(names):
    if i == 0:
        continue
    plt.plot(
        np.arange(T),
        rss_median[i],
        label=name,
        color=COLORS[name],
    )
    plt.fill_between(
        np.arange(T), rss_high[i], rss_low[i], color=COLORS[name], alpha=0.3
    )
plt.ylabel("Regret")
plt.xlabel("Time $t$")
plt.legend()
plt.savefig("../figures/toy-regret.pdf", bbox_inches="tight")
