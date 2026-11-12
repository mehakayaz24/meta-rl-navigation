"""
Gridworld navigation task for meta-reinforcement learning.

Each *episode* is a new environment:
  - a new hidden goal location is sampled
  - (optionally) the place-cell code is shuffled, mimicking hippocampal remapping

Within an episode the agent gets several *trials*: every time it reaches the goal
it receives reward +1 and is moved to a random start location. A good meta-learner
needs exploration on the first trial and can then return to the goal much faster
on later trials -- "one-shot" learning of the goal location.

The environment is vectorised: it simulates a batch of independent episodes at once.
"""
import numpy as np

ACTIONS = np.array([[-1, 0], [1, 0], [0, -1], [0, 1]])  # up, down, left, right


class PlaceCellGrid:
    def __init__(self, batch_size=32, size=5, episode_len=60, sigma=0.75,
                 remap=False, seed=None):
        self.B, self.size, self.T = batch_size, size, episode_len
        self.n_states = size * size
        self.remap = remap
        self.rng = np.random.default_rng(seed)

        # Place cells: one Gaussian bump centred on every grid square.
        ys, xs = np.divmod(np.arange(self.n_states), size)
        centres = np.stack([ys, xs], 1).astype(float)
        d2 = ((centres[:, None, :] - centres[None, :, :]) ** 2).sum(-1)
        self.codes = np.exp(-d2 / (2 * sigma ** 2))   # [state, cell]
        self.obs_dim = self.n_states

    # ---- helpers ---------------------------------------------------------
    def _idx(self, pos):
        return pos[:, 0] * self.size + pos[:, 1]

    def _random_pos(self, n):
        return self.rng.integers(0, self.size, size=(n, 2))

    def _observe(self):
        rates = self.codes[self._idx(self.pos)]                 # [B, cells]
        # remapping = each environment uses its own permutation of the cells
        return np.take_along_axis(rates, self.perm, axis=1).astype(np.float32)

    # ---- API ---------------------------------------------------------------
    def reset(self):
        self.t = 0
        self.goal = self._random_pos(self.B)
        self.pos = self._random_pos(self.B)
        same = (self.pos == self.goal).all(1)
        while same.any():                                       # never start on the goal
            self.pos[same] = self._random_pos(same.sum())
            same = (self.pos == self.goal).all(1)
        if self.remap:
            self.perm = np.stack([self.rng.permutation(self.obs_dim) for _ in range(self.B)])
        else:
            self.perm = np.tile(np.arange(self.obs_dim), (self.B, 1))
        return self._observe()

    def step(self, actions):
        self.pos = np.clip(self.pos + ACTIONS[actions], 0, self.size - 1)
        reached = (self.pos == self.goal).all(1)
        reward = reached.astype(np.float32)
        if reached.any():                                        # start a new trial
            new = self._random_pos(reached.sum())
            clash = (new == self.goal[reached]).all(1)
            while clash.any():
                new[clash] = self._random_pos(clash.sum())
                clash = (new == self.goal[reached]).all(1)
            self.pos[reached] = new
        self.t += 1
        done = self.t >= self.T
        return self._observe(), reward, reached, done
