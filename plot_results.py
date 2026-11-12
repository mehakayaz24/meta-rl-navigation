"""Combine training logs and evaluation results of both runs into figures/results.png.

    python plot_results.py --runs runs --figs figures
"""
import argparse, json
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

p = argparse.ArgumentParser()
p.add_argument("--runs", default="runs")
p.add_argument("--figs", default="figures")
p.add_argument("--seed", type=int, default=0)
a = p.parse_args()

lab = {"remap0": "fixed place cells", "remap1": "remapped place cells"}
col = {"remap0": "#2b6cb0", "remap1": "#c05621"}
fig, ax = plt.subplots(1, 3, figsize=(13, 3.6))
res = {}
for m in lab:
    log = np.array(json.load(open(f"{a.runs}/{m}_seed{a.seed}_log.json")))
    ax[0].plot(np.convolve(log, np.ones(50) / 50, mode="valid"), c=col[m], label=lab[m])
    res[m] = json.load(open(f"{a.figs}/{m}_seed{a.seed}_results.json"))
    ax[1].plot(range(1, 7), res[m]["steps_per_trial"], "o-", c=col[m], label=lab[m])
ax[0].set(xlabel="training update", ylabel="goals reached per episode", title="Training"); ax[0].legend()
ax[1].set(xlabel="trial within episode", ylabel="steps to reach goal", title="One-shot learning of the goal"); ax[1].legend()
x, w = np.arange(2), 0.35
for i, m in enumerate(lab):
    ax[2].bar(x + (i - 0.5) * w, [res[m]["decode_before"], res[m]["decode_after"]], w, color=col[m], label=lab[m])
ax[2].axhline(res["remap0"]["chance"], ls="--", c="gray", label="chance")
ax[2].set_xticks(x, ["before 1st reward", "after 1st reward"]); ax[2].set_ylim(0, 1)
ax[2].set(ylabel="goal decoding accuracy", title="Goal location in GRU hidden state"); ax[2].legend(fontsize=8)
for s in ax:
    s.spines[["top", "right"]].set_visible(False)
fig.tight_layout(); fig.savefig(f"{a.figs}/results.png", dpi=150)
print("saved", f"{a.figs}/results.png")
