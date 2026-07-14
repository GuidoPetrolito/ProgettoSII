"""
Genera un report di valutazione completo del sistema multi-agente.

Uso:
    python -m evaluation.run_evaluation

Richiede che i pesi degli attori (models/attore_*.pth) e il dataset
(data/hotels_dataset.csv) siano gia' presenti.
"""

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import pandas as pd
import torch

from env.hotel_env import FEATURE_NAMES
from marl.networks import Actor
from evaluation.metrics import (
    actor_fidelity_metrics,
    accuracy_precision_decomposition,
    rl_only_classification_metrics,
    classification_metrics,
    compare_rl_vs_system,
    matching_optimality_rate,
)

MODELS_DIR = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "models")
DATA_PATH = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "data", "hotels_dataset.csv")


def load_actors():
    actors = {}
    for name in FEATURE_NAMES:
        path = os.path.join(MODELS_DIR, f"attore_{name}.pth")
        actor = Actor()
        if os.path.exists(path):
            actor.load_state_dict(torch.load(path, map_location="cpu"))
        else:
            print(f"ATTENZIONE: pesi mancanti per '{name}', uso inizializzazione casuale.")
        actor.eval()
        actors[name] = actor
    return actors


def print_section(title):
    print()
    print("=" * 78)
    print(title)
    print("=" * 78)


def main():
    actors = load_actors()
    df = pd.read_csv(DATA_PATH)

    # --- 1. Fedelta' degli attori ---
    print_section("1. FEDELTA' DEGLI ATTORI (actor(stato) vs stato, 500 campioni casuali)")
    fidelity = actor_fidelity_metrics(actors, n_samples=500)
    print(f"MAE complessivo:  {fidelity['mae_overall']:.4f}")
    print(f"RMSE complessivo: {fidelity['rmse_overall']:.4f}")
    print(f"Distanza L2 media: {fidelity['mean_l2_distance']:.4f}  (mediana: {fidelity['median_l2_distance']:.4f})")
    print(f"Accuracy@0.05: {fidelity['accuracy@0.05']:.1%}   Accuracy@0.10: {fidelity['accuracy@0.1']:.1%}   Accuracy@0.20: {fidelity['accuracy@0.2']:.1%}")
    print()
    print(f"{'Dimensione':<14}{'MAE':<10}{'RMSE':<10}{'Bias':<10}")
    for name in FEATURE_NAMES:
        print(f"{name:<14}{fidelity['mae_per_dimension'][name]:<10.4f}{fidelity['rmse_per_dimension'][name]:<10.4f}{fidelity['bias_per_dimension'][name]:<+10.4f}")

    print()
    print("Scomposizione accuratezza (bias sistematico) vs precisione (dispersione):")
    print(f"{'Dimensione':<14}{'Accuratezza':<14}{'Precisione':<14}{'Errore dominante'}")
    decomposition = accuracy_precision_decomposition(fidelity)
    for name in FEATURE_NAMES:
        d = decomposition[name]
        print(f"{name:<14}{d['accuratezza_bias_assoluto']:<14.4f}{d['precisione_std_errore']:<14.4f}{d['errore_dominante']}")

    # --- 2a. Solo RL (nessun matching sul dataset) ---
    print_section("2a. CLASSIFICAZIONE SOLO RL (basso/medio/alto, azione grezza dell'attore, 300 prove)")
    rl_cls = rl_only_classification_metrics(actors, n_trials=300)
    print(f"{'Dimensione':<14}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    for name in FEATURE_NAMES:
        m = rl_cls[name]
        print(f"{name:<14}{m['accuracy']:<12.3f}{m['precision_macro']:<12.3f}{m['recall_macro']:<12.3f}{m['f1_macro']:<10.3f}")
    print("-" * 60)
    o = rl_cls["overall"]
    print(f"{'OVERALL':<14}{o['accuracy']:<12.3f}{o['precision_macro']:<12.3f}{o['recall_macro']:<12.3f}{o['f1_macro']:<10.3f}")

    # --- 2b. Sistema intero (RL + matching KNN sul dataset reale) ---
    print_section("2b. CLASSIFICAZIONE SISTEMA INTERO (basso/medio/alto, hotel reale abbinato, 300 prove)")
    cls = classification_metrics(df, actors, n_trials=300)
    print(f"{'Dimensione':<14}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1':<10}")
    for name in FEATURE_NAMES:
        m = cls[name]
        print(f"{name:<14}{m['accuracy']:<12.3f}{m['precision_macro']:<12.3f}{m['recall_macro']:<12.3f}{m['f1_macro']:<10.3f}")
    print("-" * 60)
    o = cls["overall"]
    print(f"{'OVERALL':<14}{o['accuracy']:<12.3f}{o['precision_macro']:<12.3f}{o['recall_macro']:<12.3f}{o['f1_macro']:<10.3f}")

    # --- 2c. Confronto diretto RL-da-solo vs sistema-intero ---
    print_section("2c. CONFRONTO DIRETTO: SOLO RL  vs  SISTEMA INTERO  (delta = sistema - RL)")
    comparison = compare_rl_vs_system(rl_cls, cls)
    print(f"{'Dimensione':<14}{'Metrica':<16}{'Solo RL':<12}{'Sistema':<12}{'Delta'}")
    for name in FEATURE_NAMES + ["overall"]:
        for metric in ("accuracy", "precision_macro", "recall_macro", "f1_macro"):
            d = comparison[name][metric]
            sign = "+" if d["delta"] >= 0 else ""
            print(f"{name:<14}{metric:<16}{d['rl_only']:<12.3f}{d['sistema_intero']:<12.3f}{sign}{d['delta']:.3f}")
        print()
    print("Nota: un delta negativo su precision/recall indica che il matching sul")
    print("dataset reale PEGGIORA la classificazione rispetto al solo RL (tipicamente")
    print("per scarsita' del dataset); un delta positivo indica che la MIGLIORA.")

    # --- 3. Matching optimality rate ---
    print_section("3. MATCHING OPTIMALITY RATE (attori+KNN vs best-match teorico, 200 prove)")
    opt = matching_optimality_rate(df, actors, n_trials=200)
    print(f"Tasso di scelta ottima: {opt['optimality_rate']:.1%}")
    print(f"Regret medio (quanto peggio del best teorico): {opt['mean_regret']:.4f}")
    print(f"Distanza media del best teorico (limite dovuto al dataset): {opt['mean_theoretical_best_distance']:.4f}")

    print()
    print("=" * 78)
    print("Nota interpretativa: se il regret medio e' vicino a 0 ma la distanza del")
    print("best teorico resta alta, il collo di bottiglia e' la densita' del dataset,")
    print("non gli attori. Se il regret e' alto, gli attori necessitano di ulteriore")
    print("addestramento (vedi anche diagnose_actors.py).")


if __name__ == "__main__":
    main()
