"""
Metriche di valutazione per il sistema multi-agente MADDPG e per la pipeline
di raccomandazione end-to-end.

"Accuratezza" e "precisione" non sono concetti nativi per un sistema di
reinforcement learning continuo, quindi vengono operazionalizzate a due livelli
distinti e complementari:

1. FEDELTA' DEGLI ATTORI (metriche di regressione continua)
   Quanto bene i 5 attori riproducono un target arbitrario nello spazio [0,1]^5,
   esattamente il compito su cui sono stati addestrati. Metriche: MAE, RMSE,
   bias medio per dimensione, "accuracy@soglia" (frazione di episodi in cui la
   distanza finale scende sotto una soglia di tolleranza).

2. QUALITA' DELLA RACCOMANDAZIONE END-TO-END (metriche di classificazione)
   Lo spazio [0,1] di ciascuna dimensione viene discretizzato in 3 classi
   (basso/medio/alto), cosi' "l'utente vuole un hotel economico" diventa una
   vera etichetta di classe confrontabile con la classe reale dell'hotel
   raccomandato. Su questa formulazione si possono calcolare accuratezza,
   precisione, richiamo e F1 nel senso classico (scikit-learn), per ciascuna
   delle 5 dimensioni e in aggregato.

Viene inoltre calcolato un terzo indicatore, il MATCHING OPTIMALITY RATE:
la frazione di prove in cui l'hotel raccomandato dalla pipeline completa
(attori + KNN pesato) coincide con il vero best-match teorico ottenuto
cercando direttamente nel dataset a partire dal target grezzo, bypassando
gli attori. Isola l'errore introdotto dagli attori da quello dovuto alla
sola scarsita' del dataset (si veda la Sezione 10.6 della relazione tecnica).
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)

from env.hotel_env import FEATURE_NAMES
from app_utils.recommender import filter_by_city, find_best_match

BIN_LABELS = ["basso", "medio", "alto"]
BIN_EDGES = [0.0, 1 / 3, 2 / 3, 1.0]


def discretize(value: float) -> str:
    """Mappa un valore continuo in [0,1] su una delle 3 classi basso/medio/alto."""
    if value < BIN_EDGES[1]:
        return BIN_LABELS[0]
    if value < BIN_EDGES[2]:
        return BIN_LABELS[1]
    return BIN_LABELS[2]


# --------------------------------------------------------------------------- #
# 1. Fedelta' degli attori (metriche di regressione continua)
# --------------------------------------------------------------------------- #
def actor_fidelity_metrics(
    actors: dict, n_samples: int = 500, seed: int = 0, thresholds=(0.05, 0.10, 0.20)
) -> dict:
    """
    Valuta quanto bene actor(stato) riproduce lo stato stesso su n_samples
    stati campionati uniformemente in [0,1]^5 (nessun rumore di esplorazione:
    valutazione greedy/deterministica, coerente con l'uso in produzione).
    """
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.0, 1.0, size=(n_samples, len(FEATURE_NAMES))).astype(np.float32)
    actions = np.zeros_like(targets)

    with torch.no_grad():
        for i, name in enumerate(FEATURE_NAMES):
            obs = torch.tensor(targets, dtype=torch.float32)
            actions[:, i] = actors[name](obs).squeeze(-1).numpy()

    errors = actions - targets  # (n_samples, 5)
    abs_errors = np.abs(errors)
    distances = np.linalg.norm(errors, axis=1)

    per_dim_mae = {FEATURE_NAMES[i]: float(abs_errors[:, i].mean()) for i in range(len(FEATURE_NAMES))}
    per_dim_rmse = {
        FEATURE_NAMES[i]: float(np.sqrt((errors[:, i] ** 2).mean())) for i in range(len(FEATURE_NAMES))
    }
    per_dim_bias = {FEATURE_NAMES[i]: float(errors[:, i].mean()) for i in range(len(FEATURE_NAMES))}

    accuracy_at = {f"accuracy@{t}": float((distances < t).mean()) for t in thresholds}

    return {
        "n_samples": n_samples,
        "mae_per_dimension": per_dim_mae,
        "rmse_per_dimension": per_dim_rmse,
        "bias_per_dimension": per_dim_bias,  # segno: >0 sovrastima, <0 sottostima
        "mae_overall": float(abs_errors.mean()),
        "rmse_overall": float(np.sqrt((errors**2).mean())),
        "mean_l2_distance": float(distances.mean()),
        "median_l2_distance": float(np.median(distances)),
        **accuracy_at,
    }


def accuracy_precision_decomposition(fidelity: dict) -> dict:
    """
    Scompone l'errore di ciascuna dimensione nelle sue due componenti classiche
    (analogia del bersaglio di tiro a segno), usando l'identita' statistica
    RMSE^2 = Bias^2 + Varianza:

      - ACCURATEZZA (|bias|): quanto l'attore sbaglia SISTEMATICAMENTE in una
        direzione (sovra/sotto-stima costante). Un bias alto indica un errore
        "prevedibile" che potrebbe essere corretto, ad es. con piu' episodi
        di training o una funzione di ricompensa diversa.
      - PRECISIONE (std dell'errore): quanto l'attore e' INSTABILE da uno
        stato all'altro, indipendentemente dal bias. Un'alta dispersione
        indica che l'attore non ha ancora convergenza uniforme sull'intero
        spazio [0,1]^5, e serve piu' training o piu' capacita' del modello.

    Se la dispersione domina sul bias, l'errore e' prevalentemente CASUALE
    (l'attore e' "instabile" ma non ha una direzione di errore preferita).
    Se il bias domina sulla dispersione, l'errore e' prevalentemente
    SISTEMATICO (l'attore e' consistente ma sbaglia sempre allo stesso modo).
    """
    decomposition = {}
    for name, rmse_val in fidelity["rmse_per_dimension"].items():
        b = fidelity["bias_per_dimension"][name]
        variance = max(rmse_val**2 - b**2, 0.0)
        std = float(variance**0.5)
        decomposition[name] = {
            "accuratezza_bias_assoluto": abs(b),
            "precisione_std_errore": std,
            "errore_dominante": "sistematico (bias)" if abs(b) > std else "casuale (dispersione)",
        }
    return decomposition


# --------------------------------------------------------------------------- #
# 2. Qualita' della classificazione: RL da solo vs sistema intero
# --------------------------------------------------------------------------- #
def classification_scores(y_true: list[str], y_pred: list[str]) -> dict:
    """Helper condiviso (pubblico): accuracy/precision/recall/F1 macro + matrice di confusione.
    Riutilizzato anche da evaluate_full_system.py per la valutazione con LLM incluso."""
    return {
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=BIN_LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=BIN_LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=BIN_LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=BIN_LABELS).tolist(),
        "confusion_matrix_labels": BIN_LABELS,
    }


def rl_only_classification_metrics(actors: dict, n_trials: int = 300, seed: int = 0) -> dict:
    """
    Precision/recall/F1/accuracy usando SOLO l'output grezzo degli attori,
    SENZA passare dal matching KNN sul dataset reale: confronta la classe
    discretizzata (basso/medio/alto) del target con la classe discretizzata
    dell'azione prodotta dall'attore. Isola la componente di classificazione
    dovuta al solo RL, da confrontare con classification_metrics() che
    include anche l'effetto (positivo o negativo) del matching sul dataset.
    """
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.0, 1.0, size=(n_trials, len(FEATURE_NAMES))).astype(np.float32)

    y_true = {name: [] for name in FEATURE_NAMES}
    y_pred = {name: [] for name in FEATURE_NAMES}

    with torch.no_grad():
        for row in targets:
            obs = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            for i, name in enumerate(FEATURE_NAMES):
                action_val = float(actors[name](obs).item())
                y_true[name].append(discretize(float(row[i])))
                y_pred[name].append(discretize(action_val))

    results = {name: classification_scores(y_true[name], y_pred[name]) for name in FEATURE_NAMES}

    overall_true = sum((y_true[name] for name in FEATURE_NAMES), [])
    overall_pred = sum((y_pred[name] for name in FEATURE_NAMES), [])
    overall = classification_scores(overall_true, overall_pred)
    overall.pop("confusion_matrix")
    overall.pop("confusion_matrix_labels")
    results["overall"] = overall
    results["n_trials"] = n_trials
    return results


def classification_metrics(
    df: pd.DataFrame,
    actors: dict,
    n_trials: int = 300,
    seed: int = 0,
    weights: np.ndarray | None = None,
) -> dict:
    """
    Genera n_trials profili utente casuali, li fa passare per l'intera pipeline
    (attori -> KNN pesata) e confronta la classe discretizzata (basso/medio/alto)
    del target richiesto con la classe discretizzata REALE dell'hotel raccomandato,
    per ciascuna delle 5 dimensioni. Ritorna precision/recall/F1/accuracy
    macro-medie per dimensione, piu' un dettaglio del classification_report.

    A differenza di rl_only_classification_metrics(), qui il "predetto" e' la
    feature REALE dell'hotel piu' vicino nel dataset, non l'azione grezza
    dell'attore: misura quindi l'INTERO sistema (RL + matching), inclusi gli
    effetti della scarsita' del dataset (Sezione 10.6 della relazione tecnica).
    """
    rng = np.random.default_rng(seed)
    targets = rng.uniform(0.0, 1.0, size=(n_trials, len(FEATURE_NAMES))).astype(np.float32)

    y_true = {name: [] for name in FEATURE_NAMES}
    y_pred = {name: [] for name in FEATURE_NAMES}

    with torch.no_grad():
        for row in targets:
            obs = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            action = np.array([actors[name](obs).item() for name in FEATURE_NAMES], dtype=np.float32)
            best_row = find_best_match(df, action, weights=weights)
            for i, name in enumerate(FEATURE_NAMES):
                y_true[name].append(discretize(float(row[i])))
                y_pred[name].append(discretize(float(best_row[name])))

    results = {name: classification_scores(y_true[name], y_pred[name]) for name in FEATURE_NAMES}

    overall_true = sum((y_true[name] for name in FEATURE_NAMES), [])
    overall_pred = sum((y_pred[name] for name in FEATURE_NAMES), [])
    overall = classification_scores(overall_true, overall_pred)
    overall.pop("confusion_matrix")
    overall.pop("confusion_matrix_labels")
    results["overall"] = overall
    results["n_trials"] = n_trials
    return results


def compare_rl_vs_system(rl_metrics: dict, system_metrics: dict) -> dict:
    """
    Costruisce una tabella di confronto diretto RL-da-solo vs sistema-intero,
    per ciascuna dimensione e in aggregato, con il delta (sistema - RL) su
    ciascuna metrica. Un delta negativo su precision/recall significa che il
    matching sul dataset reale PEGGIORA la classificazione rispetto a quanto
    gia' fa il solo RL (es. per scarsita' del dataset); un delta positivo
    significa che il matching la MIGLIORA (es. "arrotondando" verso l'hotel
    reale piu' vicino puo' correggere piccoli errori dell'attore).
    """
    names = [n for n in rl_metrics.keys() if n != "n_trials"]
    comparison = {}
    for name in names:
        rl = rl_metrics[name]
        sys_ = system_metrics[name]
        comparison[name] = {
            metric: {
                "rl_only": rl[metric],
                "sistema_intero": sys_[metric],
                "delta": round(sys_[metric] - rl[metric], 4),
            }
            for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro")
        }
    return comparison


# --------------------------------------------------------------------------- #
# 3. Matching optimality rate (isola l'errore da attori vs scarsita' dataset)
# --------------------------------------------------------------------------- #
def matching_optimality_rate(
    df: pd.DataFrame,
    actors: dict,
    n_trials: int = 200,
    seed: int = 0,
    target_city: str | None = None,
) -> dict:
    """
    Per ogni prova, confronta l'hotel scelto dalla pipeline completa
    (attori -> KNN) con il vero best-match ottenuto cercando direttamente
    nel dataset a partire dal target grezzo (bypassando gli attori).
    Se le due scelte coincidono raramente, l'errore e' attribuibile agli
    attori; se il best-match teorico stesso ha comunque una distanza residua
    alta, l'errore e' attribuibile alla scarsita' del dataset.
    """
    rng = np.random.default_rng(seed)
    subset = filter_by_city(df, target_city)
    targets = rng.uniform(0.0, 1.0, size=(n_trials, len(FEATURE_NAMES))).astype(np.float32)

    matches, regret, theoretical_dist = 0, [], []
    with torch.no_grad():
        for row in targets:
            obs = torch.tensor(row, dtype=torch.float32).unsqueeze(0)
            action = np.array([actors[name](obs).item() for name in FEATURE_NAMES], dtype=np.float32)

            system_choice = find_best_match(subset, action)
            theoretical_best = find_best_match(subset, row)

            matches += int(system_choice["name"] == theoretical_best["name"])
            regret.append(float(system_choice["distance"] - theoretical_best["distance"]))
            theoretical_dist.append(float(theoretical_best["distance"]))

    return {
        "n_trials": n_trials,
        "optimality_rate": matches / n_trials,
        "mean_regret": float(np.mean(regret)),  # quanto peggio del best teorico, in media
        "mean_theoretical_best_distance": float(np.mean(theoretical_dist)),  # limite dovuto al dataset
    }
