"""
Diagnostica di convergenza degli attori MADDPG.

Se gli attori fossero perfettamente addestrati, actor(stato) dovrebbe restituire
un'azione molto vicina allo stato stesso (dato che l'ambiente di training li
premia per avvicinarsi a QUALSIASI vettore target casuale in [0,1]^5 - quindi
un target = stato dovrebbe produrre un'azione ~= stato). Grandi scostamenti
indicano attori sotto-addestrati.

Uso:
    python -m diagnose_actors
"""

import os
import numpy as np
import torch

from env.hotel_env import FEATURE_NAMES
from marl.networks import Actor

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")


def load_actors():
    actors = {}
    missing = []
    for name in FEATURE_NAMES:
        path = os.path.join(MODELS_DIR, f"attore_{name}.pth")
        actor = Actor()
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            missing.append(name)
        actor.eval()
        actors[name] = actor
    if missing:
        print(f"ATTENZIONE: pesi mancanti per {missing}, uso inizializzazione casuale.")
    return actors


def evaluate(actors, n_samples: int = 200, seed: int = 0):
    rng = np.random.default_rng(seed)
    deviations = []
    for _ in range(n_samples):
        state = rng.uniform(0.0, 1.0, size=len(FEATURE_NAMES)).astype(np.float32)
        obs = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action = np.zeros(len(FEATURE_NAMES), dtype=np.float32)
        with torch.no_grad():
            for i, name in enumerate(FEATURE_NAMES):
                action[i] = float(actors[name](obs).item())
        deviations.append(np.linalg.norm(action - state))
    return np.array(deviations)


if __name__ == "__main__":
    actors = load_actors()
    deviations = evaluate(actors)
    print(f"Deviazione media |actor(stato) - stato|  su 200 stati casuali: {deviations.mean():.4f}")
    print(f"Deviazione mediana: {np.median(deviations):.4f}")
    print(f"Deviazione max: {deviations.max():.4f}")
    print()
    if deviations.mean() < 0.15:
        print("OK: gli attori sono ben convergenti, la deviazione e' piccola.")
    elif deviations.mean() < 0.35:
        print("ATTENZIONE: deviazione moderata. Il training potrebbe non essere completo (1000 episodi).")
    else:
        print("PROBLEMA: deviazione alta. Gli attori NON hanno imparato a preservare lo stato in ingresso.")
        print("Verifica di aver completato tutti i 1000 episodi di python -m marl.train")
        print("e controlla la curva di reward nei log (dovrebbe convergere verso un valore positivo stabile).")
