"""
Script di addestramento MADDPG per MultiAgentHotelEnv.

Esegue 1000 episodi con decadimento lineare del rumore di esplorazione
(da NOISE_START a NOISE_END) e salva i pesi ottimali (miglior reward media
mobile) degli attori come 'attore_<nome_feature>.pth'.

Uso:
    python -m marl.train
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np

from env.hotel_env import MultiAgentHotelEnv, FEATURE_NAMES
from marl.maddpg import MADDPG

N_EPISODES = 1000
MAX_STEPS_PER_EP = 25
NOISE_START = 0.4
NOISE_END = 0.02
BATCH_SIZE = 128
WARMUP_STEPS = 1000  # step casuali prima di iniziare gli update
MODEL_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")


def linear_decay(episode: int, total: int, start: float, end: float) -> float:
    frac = min(episode / max(total - 1, 1), 1.0)
    return start + frac * (end - start)


def train(n_episodes: int = N_EPISODES, verbose_every: int = 50):
    env = MultiAgentHotelEnv(max_steps=MAX_STEPS_PER_EP, seed=0)
    maddpg = MADDPG(n_agents=env.n_agents, obs_dim=env.n_agents, action_dim=1)

    reward_history = []
    best_avg_reward = -np.inf
    global_step = 0

    for episode in range(n_episodes):
        obs, _ = env.reset(seed=None)
        noise_scale = linear_decay(episode, n_episodes, NOISE_START, NOISE_END)
        episode_reward = 0.0

        for t in range(MAX_STEPS_PER_EP):
            actions = maddpg.select_actions(obs, noise_scale=noise_scale)
            next_obs, rewards, terminated, truncated, info = env.step(actions)
            dones = [term or trunc for term, trunc in zip(terminated, truncated)]

            maddpg.replay_buffer.push(obs, actions, rewards, next_obs, dones)
            obs = next_obs
            episode_reward += float(np.mean(rewards))
            global_step += 1

            if global_step > WARMUP_STEPS:
                maddpg.update(batch_size=BATCH_SIZE)

            if all(dones):
                break

        reward_history.append(episode_reward)
        avg_reward = float(np.mean(reward_history[-20:]))

        if avg_reward > best_avg_reward and episode > 20:
            best_avg_reward = avg_reward
            maddpg.save_actors(MODEL_DIR, FEATURE_NAMES)

        if episode % verbose_every == 0 or episode == n_episodes - 1:
            print(
                f"[Episodio {episode:4d}/{n_episodes}] "
                f"reward_ep={episode_reward:7.3f}  avg20={avg_reward:7.3f}  "
                f"noise={noise_scale:.3f}  buffer={len(maddpg.replay_buffer)}"
            )

    # salvataggio finale di garanzia (nel caso il best non fosse stato raggiunto tardi)
    maddpg.save_actors(MODEL_DIR, FEATURE_NAMES)
    print(f"Addestramento completato. Modelli salvati in: {MODEL_DIR}")
    return maddpg, reward_history


if __name__ == "__main__":
    train()
