import torch
import torch.nn as nn


class MaskedCategorical(torch.distributions.Categorical):
    def __init__(self, logits: torch.Tensor, mask: torch.Tensor = None):
        if mask is not None:
            logits = logits.masked_fill(mask == 0, -1e9)
        super().__init__(logits=logits)


class ActorCritic(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        hid = 64
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
        )
        self.pi = nn.Linear(hid, n_actions)
        self.v = nn.Linear(hid, 1)

    def forward(self, obs: torch.Tensor):
        x = self.body(obs)
        return self.pi(x), self.v(x)

    def act(self, obs: torch.Tensor, mask: torch.Tensor):
        logits, v = self.forward(obs)
        dist = MaskedCategorical(logits=logits, mask=mask)
        a = dist.sample()
        logp = dist.log_prob(a)
        ent = dist.entropy()
        return a, logp, ent, v.squeeze(-1)


class Actor(nn.Module):
    def __init__(self, obs_dim: int, n_actions: int):
        super().__init__()
        hid = 64
        self.body = nn.Sequential(
            nn.Linear(obs_dim, hid), nn.Tanh(),
            nn.Linear(hid, hid), nn.Tanh(),
        )
        self.pi = nn.Linear(hid, n_actions)

    def forward(self, obs: torch.Tensor):
        x = self.body(obs)
        return self.pi(x)
