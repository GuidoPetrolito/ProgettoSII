"""
MultiAgentHotelEnv
-------------------
Ambiente Gymnasium personalizzato per addestrare N agenti (uno per dimensione
dello spazio latente delle preferenze) con MADDPG. Ad ogni episodio viene
generato un vettore target segreto casuale nello spazio [0,1]^N; gli agenti
osservano lo stato corrente (il target, condiviso: ogni attore osserva tutte
le N dimensioni) ed emettono un'azione scalare in [0,1] per la propria
dimensione.

La ricompensa e' densa e identica per tutti gli agenti:
    R = 1.0 - (distanza_euclidea(azioni, target) * 1.5)

FEATURE_NAMES definisce sia la dimensionalita' dello spazio latente sia
l'ordine canonico delle feature, condiviso da env, reti neurali, LLM gateway,
state manager e recommender. Aggiungere una nuova feature (es. 'prezzo')
richiede di aggiungerla qui e nulla piu' cambia nella forma dell'ambiente.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

FEATURE_NAMES = ["reputazione", "comfort", "popolarita", "leisure", "prezzo"]
N_AGENTS = len(FEATURE_NAMES)


class MultiAgentHotelEnv(gym.Env):
    metadata = {"render_modes": []}

    def __init__(self, max_steps: int = 25, seed: int | None = None):
        super().__init__()
        self.n_agents = N_AGENTS
        self.max_steps = max_steps
        self._step_count = 0
        self._rng = np.random.default_rng(seed)

        # Ogni agente osserva l'intero vettore target (n_agents,) e agisce su 1 dimensione.
        self.observation_space = spaces.Tuple(
            [spaces.Box(low=0.0, high=1.0, shape=(self.n_agents,), dtype=np.float32) for _ in range(self.n_agents)]
        )
        self.action_space = spaces.Tuple(
            [spaces.Box(low=0.0, high=1.0, shape=(1,), dtype=np.float32) for _ in range(self.n_agents)]
        )

        self._target = np.zeros(self.n_agents, dtype=np.float32)
        self._last_actions = np.zeros(self.n_agents, dtype=np.float32)

    def reset(self, *, seed: int | None = None, options=None):
        if seed is not None:
            self._rng = np.random.default_rng(seed)
        self._step_count = 0
        self._target = self._rng.uniform(0.0, 1.0, size=self.n_agents).astype(np.float32)
        self._last_actions = self._rng.uniform(0.0, 1.0, size=self.n_agents).astype(np.float32)

        obs = [self._target.copy() for _ in range(self.n_agents)]
        info = {"target": self._target.copy()}
        return obs, info

    def step(self, actions: list[np.ndarray]):
        self._step_count += 1
        joint_action = np.array(
            [float(np.clip(np.asarray(a).reshape(-1)[0], 0.0, 1.0)) for a in actions], dtype=np.float32
        )
        self._last_actions = joint_action

        distance = float(np.linalg.norm(joint_action - self._target))
        reward = 1.0 - (distance * 1.5)

        rewards = [reward for _ in range(self.n_agents)]
        obs = [self._target.copy() for _ in range(self.n_agents)]

        terminated = distance < 0.02
        truncated = self._step_count >= self.max_steps

        info = {"target": self._target.copy(), "distance": distance, "joint_action": joint_action}
        return obs, rewards, [terminated] * self.n_agents, [truncated] * self.n_agents, info

    def render(self):
        pass

    def close(self):
        pass
