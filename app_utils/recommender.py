"""
Logica di raccomandazione:
1. Filtro testuale rigido su target_city (se presente).
2. Passa lo stato utente s_t attraverso i 4 attori MADDPG pre-addestrati
   per ottenere il vettore delle azioni coordinate (raccomandazione "ideale").
3. Matching geometrico stile KNN (k=1): minimizza la norma L2 PESATA tra il
   vettore azione e le feature normalizzate del dataset hotel filtrato.

Nota sulla pesatura: una dimensione mai menzionata esplicitamente dall'utente
resta al valore di default 0.5, ma questo NON significa che l'utente voglia
davvero un hotel "neutro" su quell'aspetto - significa solo che non gli
interessa. Se la trattiamo con lo stesso peso delle dimensioni realmente
richieste, un hotel puo' essere scartato ingiustamente solo perche' ha una
popolarita' o un mood "casuali" su un aspetto a cui l'utente non ha mai dato
importanza. Le dimensioni "touched" (mai esplicitamente specificate almeno
una volta dall'utente) pesano WEIGHT_TOUCHED; le altre pesano WEIGHT_DEFAULT.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch

from env.hotel_env import FEATURE_NAMES as FEATURE_ORDER

WEIGHT_TOUCHED = 1.0
WEIGHT_DEFAULT = 0.3


def get_action_vector(actors: dict, state: dict) -> np.ndarray:
    """
    actors: dict {feature_name: torch.nn.Module} - i 4 attori caricati.
    state: dict con le 4 chiavi di FEATURE_ORDER (lo stato utente s_t).

    Ogni attore osserva l'intero vettore di stato (4,) e produce un'azione
    scalare per la propria dimensione, coerentemente con l'architettura di
    addestramento (osservazione condivisa a 4 dim, azione individuale a 1 dim).
    """
    obs = torch.tensor([state[k] for k in FEATURE_ORDER], dtype=torch.float32).unsqueeze(0)
    action = np.zeros(len(FEATURE_ORDER), dtype=np.float32)
    with torch.no_grad():
        for i, key in enumerate(FEATURE_ORDER):
            actor = actors[key]
            action[i] = float(actor(obs).item())
    return action


def filter_by_city(df: pd.DataFrame, target_city: str | None) -> pd.DataFrame:
    if not target_city:
        return df
    mask = df["city"].str.contains(target_city, case=False, na=False)
    filtered = df[mask]
    return filtered if len(filtered) > 0 else df  # fallback: nessun match -> non filtrare


def build_weight_vector(touched: set[str] | None) -> np.ndarray:
    """touched=None -> nessuna pesatura (comportamento legacy, tutti i pesi a 1.0)."""
    if touched is None:
        return np.ones(len(FEATURE_ORDER), dtype=np.float32)
    return np.array(
        [WEIGHT_TOUCHED if key in touched else WEIGHT_DEFAULT for key in FEATURE_ORDER],
        dtype=np.float32,
    )


def find_best_match(
    df: pd.DataFrame, action_vector: np.ndarray, weights: np.ndarray | None = None
) -> pd.Series:
    """K-Nearest Neighbors (k=1) via norma L2 PESATA tra action_vector e le feature hotel."""
    if weights is None:
        weights = np.ones(len(FEATURE_ORDER), dtype=np.float32)
    feats = df[FEATURE_ORDER].to_numpy(dtype=np.float32)
    diff = feats - action_vector.reshape(1, -1)
    distances = np.sqrt(np.sum(weights.reshape(1, -1) * (diff**2), axis=1))
    best_idx = int(np.argmin(distances))
    row = df.iloc[best_idx].copy()
    row["distance"] = float(distances[best_idx])
    return row


def recommend(
    df: pd.DataFrame,
    actors: dict,
    state: dict,
    target_city: str | None,
    touched: set[str] | None = None,
):
    filtered = filter_by_city(df, target_city)
    action_vector = get_action_vector(actors, state)
    weights = build_weight_vector(touched)
    best_row = find_best_match(filtered, action_vector, weights)
    return best_row, action_vector, filtered
