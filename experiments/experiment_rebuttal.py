from mmab.cautious_greedy import CautiousGreedy
from mmab.ucb import UCB, delta
from mmab.opt import OPT
from mmab.greedy import Greedy
from mmab.etc import ETC
from mmab.environment import rewards, deterministic_reward, sample_n_players
import numpy as np
import matplotlib.pyplot as plt
from joblib import Parallel, delayed
import os
print(os.path.abspath("../figures/figure1.pdf"))

names = ["OPT", "Cautious Greedy", "UCB", "ETC"]
H = [10, 1000, 10000, 20000, 50000, 100000]
seeds = 50

def plot1():
    M = 30  # Number of players
    mu = np.array([0.8, 0.5])  # Reward of each arm
    K = len(mu)  # Number of arms
    p = 0.01  # Probability that a player is active at each round

    verbose = True

    def do_exp(seed, name):
        print(seed, name)
        rss = []
        for h in H:
            if name == "OPT":
                agent = OPT(M, K, p, h, mu,)
            if name == "UCB":
                agent = UCB(M, K, p, h, verbose=True)
            if name == "Cautious Greedy":
                agent = CautiousGreedy(M, K, p, h, verbose=True)
            if name == "ETC":
                agent = ETC(M, K, p, h)
            rng = np.random.RandomState(seed)
            rs = 0
            for t in range(h):
                proba = agent.play()
                n_players = sample_n_players(proba, rng)
                # print(n_players)
                d_rewards = deterministic_reward(mu, n_players, p)
                r, c = rewards(mu, n_players, p, rng)
                agent.update(r, c)
                rs += np.sum(d_rewards)
                # if t == 0 and name == "OPT" and seed==0:
                #     print(t, name, n_players)
                # if t % 1000 == 0 and seed==0:
                #     print(t, name, n_players)
            rss.append(rs)
        rss = np.array(rss)
        # print(rss)
        return rss

    res = Parallel(n_jobs=7, verbose=True)(delayed(do_exp)(seed, name) for seed in range(seeds) for name in names)
    res = np.array(res)

    plot = np.zeros(( len(names), seeds, len(H) ))
    for i, name in enumerate(names):
        for j, seed in enumerate(range(seeds)):
            plot[i, j] = res[j*len(names) + i]
    print(plot.shape)
    plotm = np.mean(plot, axis=1)
    plot_h = np.quantile(plot, 0.9, axis=1)
    plot_l = np.quantile(plot, 0.1, axis=1)
    return plotm, plot_h, plot_l


def plot2():
    M = 3  # Number of players
    mu = np.array([0.99, 0.01])  # Reward of each arm
    K = len(mu)  # Number of arms
    p = 0.1  # Probability that a player is active at each round

    verbose = True


    def do_exp(seed, name):
        print(seed, name)
        rss = []
        for h in H:
            if name == "OPT":
                agent = OPT(M, K, p, h, mu,)
            if name == "UCB":
                agent = UCB(M, K, p, h, verbose=True)
            if name == "Cautious Greedy":
                agent = CautiousGreedy(M, K, p, h, verbose=True)
            if name == "ETC":
                agent = ETC(M, K, p, h)
            rng = np.random.RandomState(seed)
            rs = 0
            for t in range(h):
                proba = agent.play()
                n_players = sample_n_players(proba, rng)
                # print(n_players)
                d_rewards = deterministic_reward(mu, n_players, p)
                r, c = rewards(mu, n_players, p, rng)
                agent.update(r, c)
                rs += np.sum(d_rewards)
                # if t == 0 and name == "OPT" and seed==0:
                #     print(t, name, n_players)
                # if t % 1000 == 0 and seed==0:
                #     print(t, name, n_players)
            rss.append(rs)
        rss = np.array(rss)
        # print(rss)
        return rss

    res = Parallel(n_jobs=7, verbose=True)(delayed(do_exp)(seed, name) for seed in range(seeds) for name in names)
    res = np.array(res)

    plot = np.zeros(( len(names), seeds, len(H) ))
    for i, name in enumerate(names):
        for j, seed in enumerate(range(seeds)):
            plot[i, j] = res[j*len(names) + i]
    print(plot.shape)
    plotm = np.mean(plot, axis=1)
    plot_h = np.quantile(plot, 0.9, axis=1)
    plot_l = np.quantile(plot, 0.1, axis=1)
    return plotm, plot_h, plot_l


pm, ph, pl = plot1()
pm2, ph2, pl2 = plot2()

# %%


rc = {
    "pdf.fonttype": 42,
    "text.usetex": True,
    "font.size": 16,
    "xtick.labelsize": 16,
    "ytick.labelsize": 16,
    "text.usetex": True,
    "font.family": "serif",
}
plt.rcParams.update(rc)


markers = {
    "Cautious Greedy": "*",
    "UCB": "+",
    "ETC":"x"
}

fig, axes = plt.subplots(ncols=2, figsize=(10, 1))
for ax in axes:
    ax.ticklabel_format(axis='y', style='sci', scilimits=(2,3))
    ax.spines['top'].set_visible(False)
    ax.spines['right'].set_visible(False)

for a, (plotm, plot_h, plot_l) in enumerate([(pm, ph, pl), (pm2, ph2, pl2)]):
    plot_opt = plotm[0]
    plot_opth = plot_h[0]
    plot_optl = plot_l[0]
    for i, name in enumerate(names):
        if i == 0:
            continue
        axes[a].plot(H, plot_opt- plotm[i], label=name, marker=markers[name])
        axes[a].fill_between(H, plot_optl- plot_l[i], plot_opth- plot_h[i], alpha=0.3)
        axes[a].set_ylabel("Regret")
        axes[a].set_xlabel("Horizon T")


plt.subplots_adjust(hspace=0.5)
plt.legend(ncols=3, loc="upper right", bbox_to_anchor=(0.5, 1.5), frameon=False)
plt.savefig("../figures/figure1_rebuttal.pdf", bbox_inches="tight")
plt.close()
import os
os.system("pdfcrop ../figures/figure1_rebuttal.pdf ../figures/figure1_rebuttal.pdf")
