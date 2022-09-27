from mmab.csara import CSARA
from mmab.naive import Naive, Uniform
from mmab.etc import ETC
from mmab.environment import rewards, sample_n_players
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed


def do_exp(seed, T):
    M = 4  # Number of players
    K = 2  # Number of arms
    mu = np.array([0.01, 0.99])  # Reward of each arm
    p = 0.3  # Probability that a player is active at each round
    names_agents = [
        ("3 on the best arm", Naive(np.array([1, 3]))),
        ("ETC", ETC(M, K, p, T)),
        ("CSARA", CSARA(M, K, p, T, init=np.array([3, 1]))),
    ]
    agents = [names_agents[i][1] for i in range(len(names_agents))]
    names = [names_agents[i][0] for i in range(len(names_agents))]
    rss = []
    for name, agent in names_agents:
        rng = np.random.RandomState(seed)
        rs = []
        for t in range(T):
            proba = agent.play()
            n_players = sample_n_players(proba, rng)
            r = rewards(mu, n_players, p, rng)
            agent.update(r)
            rs.append(np.sum(r))
        rss.append(np.sum(rs))
    rss = np.array(rss)
    return rss


cm = plt.cm.Set1
COLORS = {"ETC": cm(2), "CSARA": cm(3), "4 on the best arm": cm(1), "Uniform": cm(4)}

Ts = np.arange(1000, 10001, 1000)
seeds = np.arange(50)
rss = Parallel(n_jobs=10, verbose=True)(
    delayed(do_exp)(seed, T) for seed in seeds for T in Ts
)
rss = np.array(rss).reshape(len(seeds), len(Ts), 3)

rss_copy = np.copy(rss)


rss = np.copy(rss_copy)
print(rss.shape)
rss = rss.reshape(50, 10, 3)

print(rss.shape)
rss = np.array(
    [[rss[s, t, 0] - rss[s, t] for t in range(len(Ts))] for s in range(len(seeds))]
)

rss_median = np.median(rss, axis=0)
rss_high = np.quantile(rss, 0.9, axis=0)
rss_low = np.quantile(rss, 0.1, axis=0)

names = ["3 on the best arm", "ETC", "CSARA"]

plt.figure(figsize=(5, 4))
for i, name in enumerate(names):
    if i == 0:
        continue
    plt.plot(
        Ts,
        rss_median[:, i],
        label=name,
        color=COLORS[name],
    )
    plt.fill_between(Ts, rss_high[:, i], rss_low[:, i], color=COLORS[name], alpha=0.3)
plt.ylabel("Regret")
plt.xlabel("Horizon $T$")
plt.legend()
plt.savefig("../figures/toy-regret-horizon.pdf", bbox_inches="tight")
