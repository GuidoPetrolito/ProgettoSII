"""
Gateway conversazionale basato su Ollama (qwen2.5:3b) in modalita' JSON rigida.

Due responsabilita':
1. parse_intent(query, previous_state) -> dict con lo schema:
     {"target_city": str|None, "rating_pref": float|None, "comfort_pref": float|None,
      "popularity_pref": float|None, "leisure_pref": float|None, "price_pref": float|None}
2. generate_response(hotel_context, user_query) -> stringa in italiano (2-3 frasi)

Nota sul fallback euristico per price_pref: un modello da 3B parametri non
generalizza in modo affidabile a formulazioni indirette che coinvolgono
esplicitamente il denaro (es. "soldi non ne mancano" estratto come prezzo
BASSO invece che ALTO, osservato empiricamente). Per questo, oltre al prompt
con esempi few-shot, applichiamo una rete di sicurezza a parole chiave che
sovrascrive price_pref SOLO quando il testo contiene un riferimento esplicito
e non ambiguo al denaro/budget (non a categoria/lusso, che restano di
competenza esclusiva del LLM per preservare l'indipendenza comfort/prezzo).
"""

from __future__ import annotations

import json
import re
import requests

OLLAMA_URL = "http://localhost:11434/api/generate"
MODEL_NAME = "qwen2.5:3b"

INTENT_SCHEMA_KEYS = [
    "target_city",
    "rating_pref",
    "comfort_pref",
    "popularity_pref",
    "leisure_pref",
    "price_pref",
]

INTENT_SYSTEM_PROMPT = """Sei un motore di estrazione di intenti per un sistema di raccomandazione di hotel.
Analizza la richiesta dell'utente e restituisci ESCLUSIVAMENTE un oggetto JSON valido,
senza alcun testo aggiuntivo, con esattamente questi campi:

{
  "target_city": "<stringa o null>",
  "rating_pref": <float 0.0-1.0 o null>,
  "comfort_pref": <float 0.0-1.0 o null>,
  "popularity_pref": <float 0.0-1.0 o null>,
  "leisure_pref": <float 0.0-1.0 o null>,
  "price_pref": <float 0.0-1.0 o null>
}

REGOLA PIU' IMPORTANTE DI TUTTE: non inventare mai un valore per un aspetto che
l'utente non ha menzionato, nemmeno implicitamente o "per coerenza" con altri
campi. Se un utente parla SOLO di categoria e prezzo, gli altri 3 campi
(rating_pref, popularity_pref, leisure_pref) DEVONO essere null, anche se ti
sembra plausibile ipotizzare un valore. Esempio corretto:
  Query: "Cerco un hotel di categoria Lusso/Resort ma con un prezzo onesto"
  Risposta corretta: {"target_city": null, "rating_pref": null, "comfort_pref": 0.9,
  "popularity_pref": null, "leisure_pref": null, "price_pref": 0.35}
  (NON: rating_pref, popularity_pref, leisure_pref con valori inventati tipo 0.6-0.8)

Altre regole:
- Ogni valore numerico rappresenta un LIVELLO su una scala, MAI quanto l'utente si preoccupa
  di quell'aspetto. Un utente molto attento al prezzo che vuole risparmiare ha price_pref BASSO,
  non alto (alto significherebbe che vuole spendere molto, il contrario).
- rating_pref alto = utente cerca hotel con ottime recensioni; basso = la reputazione non conta.
- comfort_pref alto = utente cerca lusso/comfort elevato; basso = economico/essenziale.
- popularity_pref alto = utente preferisce hotel molto recensiti/popolari; basso = indifferente.
- leisure_pref alto = vacanza/relax; basso = viaggio di lavoro/business.
- price_pref e' il LIVELLO DI PREZZO CHE L'UTENTE VUOLE PAGARE (non quanto gli importa il prezzo):
    * price_pref VICINO A 0.0 = l'utente vuole spendere POCO (economico, budget, risparmio,
      low-cost, motel, la tariffa piu' bassa possibile, "soldi contati", "al verde").
    * price_pref VICINO A 1.0 = l'utente e' disposto a spendere molto/non bada a spese/vuole
      il massimo lusso indipendentemente dal costo, incluse espressioni indirette come
      "soldi non ne mancano", "il portafoglio non e' un problema", "posso permettermelo".
  Esempi concreti (imitali esattamente per la direzione):
    "Ho un budget molto ridotto, cerco il prezzo piu' basso possibile" -> price_pref: 0.05
    "Voglio risparmiare, opzioni low-cost o motel" -> price_pref: 0.1
    "Il budget non e' un problema, voglio il massimo" -> price_pref: 0.95
    "Preferisco pagare di piu' per avere servizi eccellenti" -> price_pref: 0.9
    "Soldi non ne mancano, voglio la struttura piu' esclusiva" -> price_pref: 0.95
    "Sono disposto a pagarlo profumatamente" -> price_pref: 0.95
  comfort_pref e price_pref sono concetti distinti: un utente puo' cercare lusso (comfort alto)
  ma a un buon prezzo (price_pref basso), oppure spendere molto (price_pref alto) senza cercare
  necessariamente comfort elevato. Se l'utente parla SOLO di categoria/lusso senza mai
  menzionare denaro/spesa/budget/prezzo, price_pref resta null.
- Tutti i valori numerici DEVONO essere rigorosamente nell'intervallo [0.0, 1.0]. Non restituire
  mai un numero maggiore di 1.0 o minore di 0.0.
- Rispondi SOLO con il JSON, nessun commento, nessun markdown.
"""

RESPONSE_SYSTEM_PROMPT = """Sei un assistente di viaggio cordiale ed esperto.
Ti verra' fornito il contesto di un hotel selezionato (che e' SEMPRE disponibile e valido:
non esiste alcuno scenario di "nessun hotel disponibile", il sistema ha gia' individuato la
struttura migliore tra quelle esistenti) e la richiesta originale dell'utente.
Genera una risposta conversazionale in ITALIANO, di massimo 2-3 frasi, che consigli
QUESTO SPECIFICO hotel in modo naturale, sicuro di se' e pertinente alla richiesta.
E' VIETATO usare frasi o concetti che esprimano dubbio, scuse o indisponibilita', tra cui:
"mi dispiace", "non troviamo", "non abbiamo trovato", "potrebbe non essere adatta/o",
"non risponde esattamente", "sembra ci sia stato un errore", "purtroppo", "non e' proprio
quello che cerchi". Presenta SEMPRE l'hotel fornito con sicurezza come la scelta consigliata,
evidenziandone gli aspetti piu' in linea con la richiesta (es. se l'utente cerca risparmio,
sottolinea il prezzo contenuto; se cerca lusso senza badare a spese, sottolinea le stelle,
la categoria e l'esclusivita', NON il prezzo contenuto). Non usare markdown, non elencare
dati grezzi (es. non dire "reputazione: 0.8"), parla in modo naturale come farebbe un concierge.
"""


# --------------------------------------------------------------------------- #
# Fallback euristico a parole chiave per price_pref (rete di sicurezza)
# --------------------------------------------------------------------------- #
_CHEAP_PATTERNS = [
    r"budget (molto )?ridott", r"budget (molto )?limitat", r"al verde",
    r"risparmi", r"buon mercato", r"low.?cost", r"prezzo (il )?pi[uù] bass",
    r"spendere (il )?meno possibile", r"non voglio spendere", r"spendere poc\w*",
    r"portafoglio [eè] l'unica cosa", r"tirare la cinghia", r"conveniente",
    r"soldi contati", r"fascia di prezzo minima", r"il pi[uù] economic",
]
_EXPENSIVE_PATTERNS = [
    r"budget non [eè] un problema", r"non bado a spese", r"non guardo a spese",
    r"soldi non (ne )?manc", r"pagar\w* profumatamente", r"spendere senza limiti",
    r"non guardo al portafoglio", r"prezzo non [eè] un problema",
    r"costi quel che costi", r"posso permettermel", r"disposto a pagare (di pi[uù]|molto)",
    r"pagare di pi[uù]", r"pagando di pi[uù]",
]

_CHEAP_RE = re.compile("|".join(_CHEAP_PATTERNS), re.IGNORECASE)
_EXPENSIVE_RE = re.compile("|".join(_EXPENSIVE_PATTERNS), re.IGNORECASE)


def _heuristic_price_override(query: str) -> float | None:
    """
    Ritorna un valore forzato per price_pref se il testo contiene un riferimento
    ESPLICITO e non ambiguo al denaro/budget, altrimenti None (nessuna sovrascrittura,
    si usa cio' che ha estratto l'LLM). Deliberatamente NON include parole di
    categoria/lusso (es. "esclusivo", "lusso", "resort") per non violare
    l'indipendenza tra comfort_pref e price_pref.
    """
    is_cheap = bool(_CHEAP_RE.search(query))
    is_expensive = bool(_EXPENSIVE_RE.search(query))
    if is_cheap and not is_expensive:
        return 0.08
    if is_expensive and not is_cheap:
        return 0.95
    return None  # ambiguo o nessun segnale esplicito -> fidati dell'LLM


def _call_ollama(system_prompt: str, user_prompt: str, json_mode: bool = True) -> str:
    payload = {
        "model": MODEL_NAME,
        "system": system_prompt,
        "prompt": user_prompt,
        "stream": False,
        "options": {"temperature": 0.1},
    }
    if json_mode:
        payload["format"] = "json"

    resp = requests.post(OLLAMA_URL, json=payload, timeout=30)
    resp.raise_for_status()
    data = resp.json()
    return data.get("response", "").strip()


def parse_intent(query: str) -> dict:
    """Chiama Ollama per estrarre l'intento; ritorna sempre un dict con le 6 chiavi attese."""
    fallback = {k: None for k in INTENT_SCHEMA_KEYS}
    try:
        raw = _call_ollama(INTENT_SYSTEM_PROMPT, query, json_mode=True)
        parsed = json.loads(raw)
    except (requests.RequestException, json.JSONDecodeError, ValueError):
        parsed = None

    if parsed is None:
        result = dict(fallback)
    else:
        result = {}
        for key in INTENT_SCHEMA_KEYS:
            val = parsed.get(key, None)
            if key == "target_city":
                result[key] = str(val) if val not in (None, "", "null") else None
            else:
                try:
                    f = float(val) if val is not None else None
                    # Difesa: un LLM di piccole dimensioni puo' violare il vincolo [0,1]
                    # dichiarato nel prompt (osservato empiricamente, es. 1.124). Il
                    # clipping va fatto qui, PRIMA della combinazione convessa Markoviana.
                    result[key] = min(max(f, 0.0), 1.0) if f is not None else None
                except (TypeError, ValueError):
                    result[key] = None

    # Rete di sicurezza: sovrascrivi price_pref se il testo contiene un riferimento
    # esplicito al denaro che l'LLM potrebbe aver interpretato nella direzione sbagliata
    # (osservato empiricamente su formulazioni indirette come "soldi non ne mancano").
    override = _heuristic_price_override(query)
    if override is not None:
        result["price_pref"] = override

    return result


def generate_response(hotel_row: dict, user_query: str) -> str:
    """Genera una risposta conversazionale in italiano basata sull'hotel individuato."""
    context = (
        f"Hotel: {hotel_row.get('name')}\n"
        f"Citta': {hotel_row.get('city')}\n"
        f"Categoria: {hotel_row.get('category')}\n"
        f"Stelle: {hotel_row.get('stars')}\n"
        f"Numero recensioni: {hotel_row.get('num_reviews')}\n"
        f"Mood: {hotel_row.get('mood')}\n"
        f"Prezzo per notte: {hotel_row.get('price_per_night')} EUR\n\n"
        f"Richiesta originale dell'utente: {user_query}"
    )
    try:
        return _call_ollama(RESPONSE_SYSTEM_PROMPT, context, json_mode=False)
    except requests.RequestException:
        return (
            f"Ti consiglio {hotel_row.get('name')} a {hotel_row.get('city')}: "
            f"un'ottima scelta in linea con quello che stai cercando."
        )
