# Meta-Reinforcement Learning for One-Shot Goal Navigation

A small PyTorch project in which a recurrent agent **learns how to learn** a goal location. After training across many environments, it finds a hidden goal once by exploring and then returns to it almost directly, without any further weight updates. The memory of the goal lives only in the activity of its recurrent network.

I built this to learn the basics of deep reinforcement learning and neural-representation analysis, inspired by work on schema learning and hippocampal remapping.

![Results](figures/results.png)

## The task

- A 5 × 5 grid. The agent observes its position only through **25 place cells**, each a Gaussian bump of activity centred on one square.
- Each **episode** is a new environment with a new hidden goal. Reaching the goal gives reward +1 and teleports the agent to a random start, so one episode (60 steps) contains several **trials**.
- **Remapping condition (`--remap`)**: in every new environment the place cells are randomly shuffled, like hippocampal remapping. The agent cannot rely on a fixed mapping from input to location.

## The agent

A GRU actor-critic (Wang et al., 2018). At each step it receives the place-cell activity, its previous action and its previous reward, and outputs an action and a value estimate. It is trained with advantage actor-critic (A2C) on batches of 32 environments.

## Results (seed 0, 4000 updates, about 12 minutes on a CPU)

| | Fixed place cells | Remapped place cells |
|---|---|---|
| Steps to goal, trial 1 (exploration) | 14.6 | 16.7 |
| Steps to goal, trials 2–6 | ≈ 3.4 | ≈ 5.4 |
| Goal decoded from hidden state, before 1st reward | 24 % | 14 % |
| Goal decoded from hidden state, after 1st reward | **96 %** | **78 %** |

Chance level for decoding is 4 % (25 locations).

**What this shows**

1. **One-shot learning.** In both conditions, the first trial is slow and the second is roughly 3 to 4 times faster. The agent stores the goal after a single experience.
2. **The goal is stored in the network's activity.** A linear decoder reads the goal location from the GRU hidden state with high accuracy *after* the first reward, but not before.
3. **Remapping makes it harder but not impossible.** With shuffled place cells the agent learns more slowly and its later trials are less efficient. Yet it still learns within each episode. How the network solves this without a fixed input map is the interesting open question.

## Run it

```bash
pip install -r requirements.txt
python train.py --updates 4000              # fixed place cells
python train.py --updates 4000 --remap      # remapped place cells
python evaluate.py runs/remap0_seed0.pt     # behaviour + decoding analysis
python evaluate.py runs/remap1_seed0.pt
python plot_results.py                      # combined figure
```

Works on Google Colab with no changes.

## Files

| File | What it does |
|---|---|
| `envs.py` | vectorised gridworld with place-cell observations and optional remapping |
| `agent.py` | GRU actor-critic network |
| `train.py` | A2C training loop |
| `evaluate.py` | steps-per-trial analysis and linear decoding of the goal from hidden states |
| `plot_results.py` | combined results figure |

## Next steps

- More seeds and error bars
- Paired-association version: several cue–goal pairs per environment
- Look at *how* the hidden state represents position under remapping (PCA, dimensionality)

## References

- Wang, J. X. et al. (2018). Prefrontal cortex as a meta-reinforcement learning system. *Nature Neuroscience*, 21, 860–868.
- Duan, Y. et al. (2016). RL²: Fast reinforcement learning via slow reinforcement learning. *arXiv:1611.02779*.
- Tse, D. et al. (2007). Schemas and memory consolidation. *Science*, 316, 76–82.
