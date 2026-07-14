"""
Implementazione MADDPG (Multi-Agent Deep Deterministic Policy Gradient) con
4 attori decentralizzati e 4 critici centralizzati, seguendo Lowe et al. 2017.
"""

from __future__ import annotations

import copy
import random
from collections import deque, namedtuple

import numpy as np
import torch
import torch.nn.functional as F
import torch.optim as optim

from .networks import Actor, CentralizedCritic, N_AGENTS, OBS_DIM_PER_AGENT, ACTION_DIM_PER_AGENT

Transition = namedtuple(
    "Transition", ["obs", "actions", "rewards", "next_obs", "dones"]
)


class ReplayBuffer:
    def __init__(self, capacity: int = 100_000):
        self.buffer = deque(maxlen=capacity)

    def push(self, obs, actions, rewards, next_obs, dones):
        self.buffer.append(Transition(obs, actions, rewards, next_obs, dones))

    def sample(self, batch_size: int):
        batch = random.sample(self.buffer, batch_size)
        return Transition(*zip(*batch))

    def __len__(self):
        return len(self.buffer)


class MADDPG:
    def __init__(
        self,
        n_agents: int = N_AGENTS,
        obs_dim: int = OBS_DIM_PER_AGENT,
        action_dim: int = ACTION_DIM_PER_AGENT,
        lr_actor: float = 1e-3,
        lr_critic: float = 1e-3,
        gamma: float = 0.95,
        tau: float = 0.01,
        device: str = "cpu",
    ):
        self.n_agents = n_agents
        self.gamma = gamma
        self.tau = tau
        self.device = torch.device(device)

        self.actors = [Actor(obs_dim, action_dim).to(self.device) for _ in range(n_agents)]
        self.critics = [CentralizedCritic().to(self.device) for _ in range(n_agents)]
        self.target_actors = [copy.deepcopy(a) for a in self.actors]
        self.target_critics = [copy.deepcopy(c) for c in self.critics]

        self.actor_optims = [optim.Adam(a.parameters(), lr=lr_actor) for a in self.actors]
        self.critic_optims = [optim.Adam(c.parameters(), lr=lr_critic) for c in self.critics]

        self.replay_buffer = ReplayBuffer()

    # ------------------------------------------------------------------ #
    # Azione
    # ------------------------------------------------------------------ #
    def select_actions(self, obs_list, noise_scale: float = 0.0):
        actions = []
        for i, actor in enumerate(self.actors):
            obs_t = torch.as_tensor(obs_list[i], dtype=torch.float32, device=self.device).unsqueeze(0)
            with torch.no_grad():
                a = actor(obs_t).cpu().numpy().flatten()
            if noise_scale > 0:
                a = a + np.random.normal(0, noise_scale, size=a.shape)
            a = np.clip(a, 0.0, 1.0)
            actions.append(a)
        return actions

    # ------------------------------------------------------------------ #
    # Update
    # ------------------------------------------------------------------ #
    def update(self, batch_size: int = 128):
        if len(self.replay_buffer) < batch_size:
            return None

        batch = self.replay_buffer.sample(batch_size)

        # shape: [n_agents][batch, dim] -> tensors
        obs = [torch.as_tensor(np.array([o[i] for o in batch.obs]), dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
        next_obs = [torch.as_tensor(np.array([o[i] for o in batch.next_obs]), dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
        actions = [torch.as_tensor(np.array([a[i] for a in batch.actions]), dtype=torch.float32, device=self.device) for i in range(self.n_agents)]
        rewards = [torch.as_tensor(np.array([r[i] for r in batch.rewards]), dtype=torch.float32, device=self.device).unsqueeze(-1) for i in range(self.n_agents)]
        dones = [torch.as_tensor(np.array([d[i] for d in batch.dones]), dtype=torch.float32, device=self.device).unsqueeze(-1) for i in range(self.n_agents)]

        all_obs = torch.cat(obs, dim=-1)
        all_next_obs = torch.cat(next_obs, dim=-1)
        all_actions = torch.cat(actions, dim=-1)

        with torch.no_grad():
            target_next_actions = torch.cat(
                [self.target_actors[i](next_obs[i]) for i in range(self.n_agents)], dim=-1
            )

        losses = {"critic": [], "actor": []}

        for i in range(self.n_agents):
            # --- Critic update ---
            with torch.no_grad():
                target_q = self.target_critics[i](all_next_obs, target_next_actions)
                y = rewards[i] + self.gamma * (1 - dones[i]) * target_q

            current_q = self.critics[i](all_obs, all_actions)
            critic_loss = F.mse_loss(current_q, y)

            self.critic_optims[i].zero_grad()
            critic_loss.backward()
            self.critic_optims[i].step()
            losses["critic"].append(critic_loss.item())

            # --- Actor update ---
            pred_actions = [
                self.actors[j](obs[j]) if j == i else actions[j].detach()
                for j in range(self.n_agents)
            ]
            pred_actions_cat = torch.cat(pred_actions, dim=-1)
            actor_loss = -self.critics[i](all_obs, pred_actions_cat).mean()

            self.actor_optims[i].zero_grad()
            actor_loss.backward()
            self.actor_optims[i].step()
            losses["actor"].append(actor_loss.item())

        self._soft_update_targets()
        return losses

    def _soft_update_targets(self):
        for i in range(self.n_agents):
            for target_param, param in zip(self.target_actors[i].parameters(), self.actors[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)
            for target_param, param in zip(self.target_critics[i].parameters(), self.critics[i].parameters()):
                target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

    # ------------------------------------------------------------------ #
    # Persistenza
    # ------------------------------------------------------------------ #
    def save_actors(self, out_dir: str, feature_names):
        import os

        os.makedirs(out_dir, exist_ok=True)
        for i, name in enumerate(feature_names):
            path = os.path.join(out_dir, f"attore_{name}.pth")
            torch.save(self.actors[i].state_dict(), path)
