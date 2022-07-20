from mmab.csara import CSARA
from mmab.naive import NAIVE
from mmab.environment import rewards
from mmab.csara import h, l
import numpy as np
import matplotlib.pyplot as plt

rng = np.random.RandomState(0)


M = 4
K = 2
mu = np.array([0.01, 0.99])
p = 0.3
T = 10000

# M = -1.5 / np.log(1 - p)
# M1 = -1 / np.log(1 - p)
# M2 = -2/ np.log(1 - p)

# %%
rss = []
for name, agent in [("NAIVE", NAIVE(np.array([0, 4]))), ("NAIVE", NAIVE(np.array([1, 3]))), ("CSARA", CSARA(M, K, p, T))]:
    rs = []
    for t in range(T):
        n_players = agent.play()
        print(name, n_players)
        r = rewards(mu, n_players, p, rng)
        agent.update(r)
        rs.append(np.sum(r))
    rss.append(rs)

# %%

plt.figure()
plt.plot(np.cumsum(rss[0]), label="CSARA")
plt.plot(np.cumsum(rss[1]), label="0-4")
plt.plot(np.cumsum(rss[2]), label="1-3")
plt.legend()
plt.show()
