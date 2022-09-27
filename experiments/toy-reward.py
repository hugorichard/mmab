from mmab.csara import CSARA
from mmab.naive import NAIVE
from mmab.etc import ETC
from mmab.environment import rewards, sample_n_players
from mmab.csara import h, l
import numpy as np
import matplotlib.pyplot as plt



M = 4 # Number of players
K = 2 # Number of arms
mu = np.array([0.01, 0.99]) # Reward of each arm
p = 0.3 # Probability that a player is active at each round
T = 10000 # Number of rounds

# %%

names_agents = [
    ("4 on the best arm", NAIVE(np.array([0, 4]))),
    ("3 on the best arm", NAIVE(np.array([1, 3]))),
    ("2 on the best arm", NAIVE(np.array([2, 2]))),
    ("1 on the best arm", NAIVE(np.array([3, 1]))),
    ("0 on the best arm", NAIVE(np.array([4, 0]))),
    ("ETC", ETC(M, K, p, T)),
    ("CSARA", CSARA(M, K, p, T)),
]
agents = [names_agents[i][1] for i in range(len(names_agents))]
names = [names_agents[i][0] for i in range(len(names_agents))]

rss = []
for name, agent in names_agents:
    rng = np.random.RandomState(0)
    rs = []
    for t in range(T):
        proba = agent.play()
        n_players = sample_n_players(proba, rng)
        r = rewards(mu, n_players, p, rng)
        agent.update(r)
        rs.append(np.sum(r))
    rss.append(rs)

# %%


# rc = {
#     "pdf.fonttype": 42,
#     "text.usetex": True,
#     "font.size": 14,
#     "xtick.labelsize": 12,
#     "ytick.labelsize": 12,
#     "text.usetex": True,
# }
# plt.rcParams.update(rc)

plt.figure(figsize=(5, 4))
for i, name in enumerate(names):
    plt.plot(np.cumsum(rss[i]), label=name)
plt.ylabel("Cumulative reward")
plt.xlabel("Number of rounds")
plt.legend()
plt.savefig("../figures/toy-reward.pdf", bbox_inches="tight")
