"""
Reti neurali per MADDPG:
- Actor: input N (osservazione = vettore target/stato condiviso), output 1
  (azione scalare sulla propria dimensione), attivazione finale Sigmoid ->
  output in [0,1].
- Critic centralizzato: input = concat(tutte le osservazioni [N*N],
  tutte le azioni [N*1]) = N*(N+1), output = singolo valore Q.

N e' derivato da env.hotel_env.FEATURE_NAMES, cosi' aggiungere una feature
(es. 'prezzo') aggiorna automaticamente le dimensioni di tutte le reti.
"""

import torch
import torch.nn as nn

from env.hotel_env import FEATURE_NAMES

N_AGENTS = len(FEATURE_NAMES)
OBS_DIM_PER_AGENT = N_AGENTS  # ogni attore osserva l'intero vettore di stato
ACTION_DIM_PER_AGENT = 1

CRITIC_INPUT_DIM = (OBS_DIM_PER_AGENT * N_AGENTS) + (ACTION_DIM_PER_AGENT * N_AGENTS)


class Actor(nn.Module):
    def __init__(self, obs_dim: int = OBS_DIM_PER_AGENT, action_dim: int = ACTION_DIM_PER_AGENT, hidden: int = 64):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(obs_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, action_dim),
            nn.Sigmoid(),  # garantisce output in [0, 1]
        )

    def forward(self, obs: torch.Tensor) -> torch.Tensor:
        return self.net(obs)


class CentralizedCritic(nn.Module):
    def __init__(self, input_dim: int = CRITIC_INPUT_DIM, hidden: int = 128):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, hidden),
            nn.ReLU(),
            nn.Linear(hidden, hidden),
            nn.ReLU(),
            nn.Linear(hidden, 1),
        )

    def forward(self, all_obs: torch.Tensor, all_actions: torch.Tensor) -> torch.Tensor:
        x = torch.cat([all_obs, all_actions], dim=-1)
        return self.net(x)
