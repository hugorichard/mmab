import numpy as np
import matplotlib.pyplot as plt

names = ["OPT", "ETC", "UCB", "Greedy"]

experiments = ["0_players", "no_elimination"]

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

f, axes = plt.subplots(ncols=len(experiments), sharey=False, figsize=(12, 3))
for a, experiment in enumerate(experiments):
    rss_median = np.load("../data/" + experiment + "_rss_median.npy")
    rss_high = np.load("../data/" + experiment + "_rss_high.npy")
    rss_low = np.load("../data/" + experiment + "_rss_low.npy")

    _, T = rss_median.shape

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

    I = np.arange(T, step=10)
    for i, name in enumerate(names):
        if i == 0:
            continue
        axes[a].plot(
            np.arange(T),
            rss_median[i],
            label=name,
            color=COLORS[name],
        )
        axes[a].fill_between(
            np.arange(T)[I], rss_high[i][I], rss_low[i][I], color=COLORS[name], alpha=0.3
        )
    axes[a].set_ylabel("Cumulated Regret")
    axes[a].set_xlabel("Timestep $t$")
plt.legend()
plt.savefig("../figures/all.pdf", bbox_inches="tight")
