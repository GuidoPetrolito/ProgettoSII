"""
Valutazione dell'INTERO sistema: LLM (estrazione intento + rete di sicurezza
euristica) -> aggiornamento Markoviano -> attori MADDPG -> matching KNN pesato.

Le valutazioni in metrics.py (rl_only_classification_metrics, classification_metrics)
partono da un target sintetico generato casualmente DIRETTAMENTE nello spazio
latente [0,1]^5, bypassando completamente il linguaggio naturale. Misurano quindi
"RL puro" vs "RL + matching sul dataset", ma NON misurano se una frase in
linguaggio naturale viene tradotta correttamente in quel target - che e' il
punto piu' fragile di tutta la pipeline (si vedano i difetti 1, 3, 4 della
Sezione 10 della relazione tecnica).

Qui la verita' di base non puo' essere generata casualmente: e' un'etichetta
scritta da un umano (vedi llm_test_set.py) su cosa una certa frase dovrebbe
produrre. Questo permette di scomporre l'errore su TRE livelli cumulativi:

  Livello 1 (solo LLM+euristica):        query -> intento estratto
  Livello 2 (+ Markov):                  intento -> stato utente persistito
  Livello 3 (+ RL + KNN, sistema intero): stato -> hotel reale raccomandato

Richiede Ollama attivo per l'estrazione LLM vera e propria; se Ollama non e'
raggiungibile, parse_intent() ricade sul fallback (rete di sicurezza euristica
per price_pref, None per gli altri campi) - la valutazione resta eseguibile,
ma il Livello 1/2 rifletteranno solo l'euristica, non la comprensione LLM.

Uso:
    python -m evaluation.evaluate_llm_pipeline
"""

from __future__ import annotations

import os
import sys

sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from sklearn.metrics import accuracy_score, confusion_matrix, f1_score, precision_score, recall_score

from evaluation.llm_test_set import LABELED_QUERIES
from evaluation.metrics import discretize, BIN_LABELS
from llm.ollama_gateway import parse_intent
from app_utils.state_manager import apply_intent_update, get_user_touched
from app_utils.recommender import recommend

DIMENSIONS_TESTED = ["prezzo", "comfort"]  # le uniche con etichetta umana nel test set attuale


def _scores(y_true, y_pred):
    return {
        "n": len(y_true),
        "accuracy": float(accuracy_score(y_true, y_pred)),
        "precision_macro": float(precision_score(y_true, y_pred, labels=BIN_LABELS, average="macro", zero_division=0)),
        "recall_macro": float(recall_score(y_true, y_pred, labels=BIN_LABELS, average="macro", zero_division=0)),
        "f1_macro": float(f1_score(y_true, y_pred, labels=BIN_LABELS, average="macro", zero_division=0)),
        "confusion_matrix": confusion_matrix(y_true, y_pred, labels=BIN_LABELS).tolist(),
    }


def full_system_evaluation(df, actors, labeled_queries=None, db_path=None, verbose=True):
    """
    Esegue ogni query etichettata attraverso la pipeline reale (username isolato
    per query, per evitare contaminazione di stato tra i casi di test) e
    confronta, per ciascuna dimensione con etichetta attesa, la classe
    discretizzata a TRE livelli: intento grezzo, stato Markoviano, hotel finale.
    """
    labeled_queries = labeled_queries or LABELED_QUERIES
    if db_path is None:
        db_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "eval_users_db.json")
    if os.path.exists(db_path):
        os.remove(db_path)  # valutazione pulita, senza stato residuo da run precedenti

    y_true = {dim: [] for dim in DIMENSIONS_TESTED}
    y_pred_livello1_intento = {dim: [] for dim in DIMENSIONS_TESTED}
    y_pred_livello2_stato = {dim: [] for dim in DIMENSIONS_TESTED}
    y_pred_livello3_hotel = {dim: [] for dim in DIMENSIONS_TESTED}

    for i, case in enumerate(labeled_queries):
        username = f"eval_user_{i}"
        query = case["query"]

        intent = parse_intent(query)
        new_state = apply_intent_update(username, intent, target_city=intent.get("target_city"), db_path=db_path)
        touched = get_user_touched(username, db_path=db_path)
        best_row, action_vector, filtered = recommend(df, actors, new_state, intent.get("target_city"), touched=touched)

        for dim in DIMENSIONS_TESTED:
            expected = case.get(f"expected_{dim}")
            if expected is None:
                continue  # dimensione non testata in questa query (nessuna etichetta umana)

            intent_key = {"prezzo": "price_pref", "comfort": "comfort_pref"}[dim]
            raw_intent_val = intent.get(intent_key)

            y_true[dim].append(expected)
            # Livello 1: solo l'intento grezzo estratto (se None, l'LLM non ha detto nulla
            # su questa dimensione - trattato come "medio" perche' un campo null implica
            # nessuna preferenza espressa, il caso neutro).
            y_pred_livello1_intento[dim].append(discretize(raw_intent_val) if raw_intent_val is not None else "medio")
            y_pred_livello2_stato[dim].append(discretize(new_state[dim]))
            y_pred_livello3_hotel[dim].append(discretize(float(best_row[dim])))

        if verbose:
            print(f"[{i+1}/{len(labeled_queries)}] \"{query[:60]}...\" -> hotel: {best_row['name']} ({best_row['city']})")

    report = {}
    for dim in DIMENSIONS_TESTED:
        if len(y_true[dim]) == 0:
            continue
        report[dim] = {
            "livello1_solo_intento": _scores(y_true[dim], y_pred_livello1_intento[dim]),
            "livello2_dopo_markov": _scores(y_true[dim], y_pred_livello2_stato[dim]),
            "livello3_sistema_intero": _scores(y_true[dim], y_pred_livello3_hotel[dim]),
        }

    if os.path.exists(db_path):
        os.remove(db_path)  # non lasciare residui di valutazione nel db reale

    return report


def print_report(report):
    print()
    print("=" * 90)
    print("VALUTAZIONE END-TO-END: LLM + MARKOV + RL + KNN, contro etichette umane")
    print("=" * 90)
    for dim, levels in report.items():
        print()
        print(f"--- Dimensione: {dim} ---")
        print(f"{'Livello':<28}{'N':<6}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1'}")
        for level_name, label in [
            ("livello1_solo_intento", "1. Solo intento LLM"),
            ("livello2_dopo_markov", "2. + Markov (stato)"),
            ("livello3_sistema_intero", "3. + RL+KNN (sistema)"),
        ]:
            m = levels[level_name]
            print(f"{label:<28}{m['n']:<6}{m['accuracy']:<12.3f}{m['precision_macro']:<12.3f}{m['recall_macro']:<12.3f}{m['f1_macro']:.3f}")
    print()
    print("Nota: il Livello 1 isola la sola comprensione linguistica (LLM + rete di")
    print("sicurezza euristica); il Livello 3 e' l'intero sistema end-to-end. Un forte")
    print("calo dal Livello 1 al Livello 3 indica che l'errore si accumula soprattutto")
    print("a valle (RL/matching), non nella comprensione della richiesta; viceversa se")
    print("il calo avviene gia' al Livello 1, il problema e' nell'estrazione LLM stessa.")


if __name__ == "__main__":
    import pandas as pd
    import torch
    from env.hotel_env import FEATURE_NAMES
    from marl.networks import Actor

    MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "models")
    DATA_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "data", "hotels_dataset.csv")

    actors = {}
    for name in FEATURE_NAMES:
        a = Actor()
        path = os.path.join(MODELS_DIR, f"attore_{name}.pth")
        if os.path.exists(path):
            a.load_state_dict(torch.load(path, map_location="cpu"))
        a.eval()
        actors[name] = a

    df = pd.read_csv(DATA_PATH)
    report = full_system_evaluation(df, actors)
    print_report(report)
