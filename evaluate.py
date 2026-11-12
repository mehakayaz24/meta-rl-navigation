"""Evaluate a trained agent and analyse its hidden representations.

Produces:
  1. steps needed to reach the goal on trial 1, 2, 3 ... of an episode.
     If the agent has meta-learned, trial 1 is slow (exploration) and later
     trials are much faster (it remembers where the goal is).
  2. a linear decoder that reads the goal location out of the GRU hidden state,
     at different points in the episode (before vs. after the first reward).

Example:
    python evaluate.py runs/remap0_seed0.pt
"""
import argparse, json, os, warnings
warnings.filterwarnings("ignore")
import numpy as np
import torch
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
from envs import PlaceCellGrid
from agent import RecurrentActorCritic
from train import run_episode


def steps_per_trial(reached, max_trials=6):
    """reached: [T, B] booleans -> mean steps to goal for trial k (complete trials only)."""
    T, B = reached.shape
    per_trial = [[] for _ in range(max_trials)]
    for b in range(B):
        hits = np.flatnonzero(reached[:, b])
        prev = -1
        for k, t in enumerate(hits[:max_trials]):
            per_trial[k].append(t - prev)
            prev = t
    return [np.mean(x) if len(x) > 10 else np.nan for x in per_trial]


def main():
    p = argparse.ArgumentParser()
    p.add_argument("checkpoint")
    p.add_argument("--episodes", type=int, default=20)
    p.add_argument("--out", default="figures")
    a = p.parse_args()

    ck = torch.load(a.checkpoint, weights_only=False)
    cfg = ck["args"]
    env = PlaceCellGrid(64, cfg["size"], cfg["episode_len"], remap=cfg["remap"], seed=123)
    agent = RecurrentActorCritic(env.obs_dim, hidden=cfg["hidden"])
    agent.load_state_dict(ck["state_dict"]); agent.eval()
    tag = os.path.basename(a.checkpoint).replace(".pt", "")
    os.makedirs(a.out, exist_ok=True)

    reached, H, goals, first_hit = [], [], [], []
    with torch.no_grad():
        for _ in range(a.episodes):
            ep = run_episode(env, agent, record=True)
            reached.append(ep["reached"]); H.append(ep["hiddens"])
            goals.append(env.goal[:, 0] * env.size + env.goal[:, 1])
            r = ep["reached"]
            first_hit.append(np.where(r.any(0), r.argmax(0), r.shape[0]))
    reached = np.concatenate(reached, 1)            # [T, N]
    H = np.concatenate(H, 1)                         # [T, N, hidden]
    goals = np.concatenate(goals)                    # [N]
    first_hit = np.concatenate(first_hit)            # [N]

    # 1) behaviour: steps to goal per trial
    spt = steps_per_trial(reached)
    print("mean steps to goal, trial 1..6:", np.round(spt, 1))

    # 2) representation: decode goal from hidden state before vs after first reward
    def decode(mask_fn):
        X, y = [], []
        for n in range(H.shape[1]):
            ts = mask_fn(n)
            if len(ts):
                t = ts[len(ts) // 2]
                X.append(H[t, n]); y.append(goals[n])
        X, y = np.array(X), np.array(y)
        if len(np.unique(y)) < 2:
            return np.nan, 0
        clf = LogisticRegression(max_iter=2000)
        return cross_val_score(clf, X, y, cv=5).mean(), len(y)

    T = H.shape[0]
    before, nb = decode(lambda n: np.arange(0, first_hit[n]))
    after, na = decode(lambda n: np.arange(first_hit[n] + 1, T))
    chance = 1 / env.n_states
    print(f"goal decoding accuracy  before 1st reward: {before:.2f} (n={nb})"
          f" | after: {after:.2f} (n={na}) | chance: {chance:.2f}")

    # figure
    fig, ax = plt.subplots(1, 2, figsize=(9, 3.5))
    k = np.arange(1, len(spt) + 1)
    ax[0].plot(k, spt, "o-")
    ax[0].set_xlabel("trial within episode"); ax[0].set_ylabel("steps to reach goal")
    ax[0].set_title("Behaviour")
    ax[1].bar(["before\n1st reward", "after\n1st reward"], [before, after])
    ax[1].axhline(chance, ls="--", c="gray", label="chance")
    ax[1].set_ylabel("goal decoding accuracy"); ax[1].set_ylim(0, 1)
    ax[1].set_title("Hidden-state representation"); ax[1].legend()
    fig.suptitle(f"{tag}")
    fig.tight_layout(); fig.savefig(f"{a.out}/{tag}.png", dpi=150)
    json.dump({"steps_per_trial": [None if np.isnan(x) else float(x) for x in spt],
               "decode_before": float(before), "decode_after": float(after), "chance": chance},
              open(f"{a.out}/{tag}_results.json", "w"), indent=2)
    print("saved", f"{a.out}/{tag}.png")


if __name__ == "__main__":
    main()
