"""Recurrent actor-critic agent (GRU), the standard architecture for meta-RL
(Wang et al., 2018; Duan et al., 2016). The input at each step is the current
observation, the previous action and the previous reward, so the recurrent state
can learn *within an episode* where the goal is."""
import torch
import torch.nn as nn


class RecurrentActorCritic(nn.Module):
    def __init__(self, obs_dim, n_actions=4, hidden=128):
        super().__init__()
        self.n_actions = n_actions
        self.hidden = hidden
        self.gru = nn.GRUCell(obs_dim + n_actions + 1, hidden)
        self.policy = nn.Linear(hidden, n_actions)
        self.value = nn.Linear(hidden, 1)

    def init_state(self, batch_size):
        return torch.zeros(batch_size, self.hidden)

    def forward(self, obs, prev_action, prev_reward, h):
        a = nn.functional.one_hot(prev_action, self.n_actions).float()
        x = torch.cat([obs, a, prev_reward.unsqueeze(1)], dim=1)
        h = self.gru(x, h)
        return self.policy(h), self.value(h).squeeze(1), h
