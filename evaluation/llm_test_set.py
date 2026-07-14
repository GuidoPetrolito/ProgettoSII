"""
Dataset di query etichettate manualmente, riutilizzate dalla batteria di test
validata nel corso dello sviluppo (Sezioni 1-5). A differenza delle valutazioni
in metrics.py basate su target sintetici casuali, qui la "verita' di base"
(ground truth) e' un giudizio umano su cosa una frase in linguaggio naturale
dovrebbe produrre - l'unico modo per valutare se il gateway LLM (il pezzo piu'
fragile della pipeline) traduce correttamente l'intento in un target numerico.

Ogni voce ha un'etichetta attesa (None se la dimensione non e' ambigua/testata
in quella query) in {"basso", "medio", "alto"}, coerente con BIN_LABELS in
metrics.py. Le etichette price sono quelle gia' validate empiricamente nei
round di test precedenti; le etichette comfort sono state aggiunte dove la
richiesta menziona esplicitamente una categoria (Lusso/Resort o Budget/Motel).
"""

LABELED_QUERIES = [
    # --- Prezzo basso (economico) ---
    {"query": "Devo andare a Napoli e ho un budget molto ridotto. Cerco un hotel della categoria Budget/Motel o comunque la struttura più economica disponibile. Mi basta che sia pulito e dignitoso, il prezzo deve essere il più basso possibile.",
     "expected_prezzo": "basso", "expected_comfort": "basso"},
    {"query": "Per il mio prossimo viaggio a Catania la priorità resta lo screening dei costi: voglio spendere pochissimo.",
     "expected_prezzo": "basso", "expected_comfort": None},
    {"query": "Dovrò fermarmi a Bari. Mostrami le alternative low-cost o motel della fascia di prezzo minima. Il portafoglio è l'unica cosa che conta per questo viaggio.",
     "expected_prezzo": "basso", "expected_comfort": None},
    {"query": "Cerco l'hotel più a buon mercato che c'è, non voglio spendere niente.",
     "expected_prezzo": "basso", "expected_comfort": None},
    {"query": "Sono al verde, datemi la sistemazione più conveniente possibile.",
     "expected_prezzo": "basso", "expected_comfort": None},
    {"query": "Ho un budget familiare limitato per questo viaggio, cerchiamo di risparmiare.",
     "expected_prezzo": "basso", "expected_comfort": None},

    # --- Prezzo alto (lusso / nessun limite di spesa) ---
    {"query": "Sto organizzando un viaggio a Roma. Cerco un'esperienza esclusiva: mi serve un hotel di categoria Lusso/Resort con ottime recensioni e stelle alte. Il budget non è un problema, voglio il massimo del comfort.",
     "expected_prezzo": "alto", "expected_comfort": "alto"},
    {"query": "Adesso devo prenotare a Venezia. Preferisco pagare di più ma avere servizi eccellenti.",
     "expected_prezzo": "alto", "expected_comfort": None},
    {"query": "Non bado a spese, cercami il non plus ultra.",
     "expected_prezzo": "alto", "expected_comfort": None},
    {"query": "Soldi non ne mancano, voglio la struttura più esclusiva disponibile.",
     "expected_prezzo": "alto", "expected_comfort": "alto"},
    {"query": "Va bene anche un hotel semplice, standard, ma sono disposto a pagarlo profumatamente pur di avere la location perfetta.",
     "expected_prezzo": "alto", "expected_comfort": "basso"},

    # --- Casi di indipendenza comfort/prezzo (Sezione 4 originale) ---
    {"query": "Cerco un hotel di categoria Lusso/Resort ma con un prezzo onesto, senza spendere una follia.",
     "expected_prezzo": "medio", "expected_comfort": "alto"},

    # --- Nessuna preferenza esplicita (deve ereditare, non deve inventare) ---
    {"query": "Cerco un hotel a Roma",
     "expected_prezzo": None, "expected_comfort": None},
]
