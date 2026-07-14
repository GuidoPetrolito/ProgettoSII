"""
Gestione dello stato utente persistente (memoria Markoviana).

Formula di aggiornamento (combinazione convessa):
    s_t = 0.2 * s_{t-1} + 0.8 * v_intento

Dove v_intento eredita dal passo precedente s_{t-1} ogni componente nulla
restituita dall'LLM. Lo stato finale viene troncato in [0, 1] e persistito
in un file JSON locale (users_db.json), indicizzato per username.

Oltre allo stato corrente, viene persistito anche lo STORICO degli stati
(uno snapshot per ogni turno di chat) cosi' che il grafico di monitoraggio
in tempo reale nell'app possa essere ricostruito quando l'utente ricarica
un profilo gia' esistente, invece di ripartire vuoto.
"""

from __future__ import annotations

import json
import os
import threading

FEATURE_KEYS = ["reputazione", "comfort", "popolarita", "leisure", "prezzo"]
ALPHA_OLD = 0.2
ALPHA_NEW = 0.8
MAX_HISTORY_LEN = 200  # cap per evitare che users_db.json cresca senza limiti

_LOCK = threading.Lock()

DEFAULT_STATE = {
    "reputazione": 0.5,
    "comfort": 0.5,
    "popolarita": 0.5,
    "leisure": 0.5,
    "prezzo": 0.5,
}

INTENT_TO_FEATURE_KEY = {
    "rating_pref": "reputazione",
    "comfort_pref": "comfort",
    "popularity_pref": "popolarita",
    "leisure_pref": "leisure",
    "price_pref": "prezzo",
}


def _db_path(db_path: str | None = None) -> str:
    if db_path:
        return db_path
    here = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    return os.path.join(here, "users_db.json")


def load_db(db_path: str | None = None) -> dict:
    path = _db_path(db_path)
    if not os.path.exists(path):
        return {}
    with _LOCK:
        try:
            with open(path, "r", encoding="utf-8") as f:
                return json.load(f)
        except (json.JSONDecodeError, OSError):
            return {}


def _save_db(db: dict, db_path: str | None = None) -> None:
    path = _db_path(db_path)
    with _LOCK:
        tmp_path = path + ".tmp"
        with open(tmp_path, "w", encoding="utf-8") as f:
            json.dump(db, f, indent=2, ensure_ascii=False)
        os.replace(tmp_path, path)


def get_user_state(username: str, db_path: str | None = None) -> dict:
    """Ritorna solo le 5 componenti dello stato (per la sidebar/matching), senza lo storico."""
    db = load_db(db_path)
    record = db.get(username, DEFAULT_STATE)
    return {key: record.get(key, DEFAULT_STATE[key]) for key in FEATURE_KEYS}


def get_user_history(username: str, db_path: str | None = None) -> list[dict]:
    """Ritorna la lista degli snapshot di stato (uno per turno) per ricostruire il grafico."""
    db = load_db(db_path)
    record = db.get(username, {})
    return list(record.get("history", []))


def get_user_touched(username: str, db_path: str | None = None) -> set[str]:
    """Ritorna l'insieme delle dimensioni MAI esplicitamente menzionate dall'utente
    (usato per pesare il matching geometrico: una dimensione mai richiesta non deve
    competere alla pari con quelle che l'utente ha davvero specificato)."""
    db = load_db(db_path)
    record = db.get(username, {})
    return set(record.get("touched", []))


def apply_intent_update(
    username: str,
    intent: dict,
    target_city: str | None = None,
    db_path: str | None = None,
) -> dict:
    """
    Applica l'aggiornamento Markoviano allo stato persistente dell'utente.

    intent: dict con chiavi rating_pref/comfort_pref/popularity_pref/leisure_pref/price_pref
            (float in [0,1] oppure None).
    """
    db = load_db(db_path)
    prev_record = db.get(username, DEFAULT_STATE)
    prev_state = {key: prev_record.get(key, DEFAULT_STATE[key]) for key in FEATURE_KEYS}
    prev_touched = set(prev_record.get("touched", []))

    # 1. Costruisci v_intento ereditando dal passo precedente dove null
    v_intento = {}
    touched_this_turn = set()
    for intent_key, feature_key in INTENT_TO_FEATURE_KEY.items():
        val = intent.get(intent_key)
        v_intento[feature_key] = float(val) if val is not None else prev_state.get(feature_key, 0.5)
        if val is not None:
            touched_this_turn.add(feature_key)

    # 2. Combinazione convessa
    new_state = {}
    for key in FEATURE_KEYS:
        s_prev = prev_state.get(key, 0.5)
        v_new = v_intento[key]
        s_t = ALPHA_OLD * s_prev + ALPHA_NEW * v_new
        new_state[key] = float(min(max(s_t, 0.0), 1.0))  # clip [0,1]

    # 3. Aggiorna lo storico persistente (append + cap alla lunghezza massima)
    history = list(prev_record.get("history", []))
    history.append(dict(new_state))
    if len(history) > MAX_HISTORY_LEN:
        history = history[-MAX_HISTORY_LEN:]

    # 4. "touched" e' cumulativo: una volta che l'utente esprime una preferenza su
    # una dimensione, resta rilevante per il matching anche nei turni successivi
    # in cui non la ripete (viene comunque ereditata dallo stato Markoviano).
    new_touched = prev_touched | touched_this_turn

    # 5. Persisti (stato corrente + ultima citta' target + storico + touched)
    user_record = dict(new_state)
    user_record["history"] = history
    user_record["touched"] = sorted(new_touched)
    if target_city:
        user_record["last_target_city"] = target_city
    elif "last_target_city" in prev_record:
        user_record["last_target_city"] = prev_record["last_target_city"]

    db[username] = user_record
    _save_db(db, db_path)
    return new_state
