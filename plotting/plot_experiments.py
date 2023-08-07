import numpy as np
import matplotlib.pyplot as plt

names = ["OPT", "UCB", "Greedy"]


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

experiment =  "no_elimination"
rss_median = np.load("../data/" + experiment + "_rss_medianT1e4.npy")[:, :10000]
rss_high = np.load("../data/" + experiment + "_rss_highT1e4.npy")[:, :10000]
rss_low = np.load("../data/" + experiment + "_rss_lowT1e4.npy")[:, :10000]

_, T = rss_median.shape
I = np.arange(T, step=10)
plt.figure(figsize=(5, 2))
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
        np.arange(T)[I], rss_high[i][I], rss_low[i][I], color=COLORS[name], alpha=0.3,
    )
plt.ylabel("Cumulative Regret")
plt.xlabel("Timestep $t$")
plt.legend(bbox_to_anchor=(1, 1.5), ncol=3)
plt.savefig("../figures/" + experiment + ".pdf", bbox_inches="tight")
plt.savefig("../figures/" + experiment + ".png", bbox_inches="tight")


experiment = "0_players"
rss_median = np.load("../data/" + experiment + "_rss_median.npy")[:, :10000]
rss_high = np.load("../data/" + experiment + "_rss_high.npy")[:, :10000]
rss_low = np.load("../data/" + experiment + "_rss_low.npy")[:, :10000]

_, T = rss_median.shape

I = np.arange(T, step=10)
plt.figure(figsize=(5, 2))
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
        np.arange(T)[I], rss_high[i][I], rss_low[i][I], color=COLORS[name], alpha=0.3,
    )
plt.ylabel("Cumulative Regret")
plt.xlabel("Timestep $t$")
# plt.legend(bbox_to_anchor=(1, 1.5), ncol=3)
plt.savefig("../figures/" + experiment + ".pdf", bbox_inches="tight")
plt.savefig("../figures/" + experiment + ".png", bbox_inches="tight")



