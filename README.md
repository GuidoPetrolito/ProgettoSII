# Hotel Recommender — LLM + MADDPG (Multi-Agent RL) + Streamlit

Sistema di raccomandazione ibrido per hotel:
- **LLM (Ollama, `qwen2.5:3b`)** per l'estrazione dell'intento dell'utente in JSON rigido.
- **MADDPG** (5 attori decentralizzati + 5 critici centralizzati) per apprendere a
  tradurre un profilo di preferenza in un'azione coordinata nello spazio 5D
  `[reputazione, comfort, popolarita, leisure, prezzo]`.
- **Streamlit** per l'interfaccia (layout stile Amazon: vetrina in alto, login a
  sinistra, chat in basso a sinistra, scatter plot comfort/prezzo in basso a destra).

> **Nota sulla feature prezzo**: `prezzo` e' una dimensione distinta da `comfort`.
> `comfort` misura il livello qualitativo dichiarato della struttura (categoria),
> mentre `prezzo` misura quanto l'utente e' disposto a spendere. Le due cose
> possono divergere (hotel di lusso convenienti, o hotel standard cari), e
> tracciarle separatamente permette al sistema di distinguere un utente che
> "cerca il lusso" da uno che "cerca il lusso ma senza spendere troppo".

## 1. Setup ambiente

```bash
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Installa e avvia Ollama con il modello richiesto:

```bash
brew install ollama          # oppure vedi https://ollama.com
ollama serve                 # in un terminale separato
ollama pull qwen2.5:3b
```

## 2. Generazione del dataset

Se possiedi un dataset reale, posizionalo in `data/hotels_raw.csv` con colonne
`name, city, stars, category, num_reviews, mood, price_per_night` (category in
`Lusso/Resort | Standard | Budget/Motel`, mood in `Relax/Vacanza | Business/Lavoro`,
price_per_night in EUR). Se `price_per_night` manca, viene assunta una fascia
media costante. In assenza di un file, viene generato automaticamente un
dataset sintetico plausibile (prezzo incluso, correlato a categoria e citta').

```bash
python -m data.generate_dataset
```

Produce `data/hotels_dataset.csv` con le 5 feature normalizzate in [0,1].

## 3. Addestramento MADDPG

```bash
python -m marl.train
```

- 1000 episodi, decadimento lineare del rumore di esplorazione (0.4 -> 0.02).
- Salva i pesi migliori in `models/attore_reputazione.pth`,
  `models/attore_comfort.pth`, `models/attore_popolarita.pth`,
  `models/attore_leisure.pth`, `models/attore_prezzo.pth`.

## 4. Avvio dell'app (macOS Apple Silicon — istruzioni tassative)

Per evitare Segmentation Fault e deadlock di memoria di PyTorch dentro i thread
di Streamlit su Apple Silicon, avvia **sempre** l'app in modalita' headless con
indirizzo IPv4 esplicito:

```bash
streamlit run app.py --server.headless true --server.address 127.0.0.1
```

Le variabili d'ambiente di fork-safety (`OBJC_DISABLE_INITIALIZE_FORK_SAFETY`,
`KMP_DUPLICATE_LIB_OK`, `OMP_NUM_THREADS=1`, `MKL_NUM_THREADS=1`) e il vincolo
`torch.set_num_threads(1)` sono gia' impostati come primissime righe di `app.py`,
prima di qualunque altro import. Gli attori PyTorch vengono caricati una sola
volta in `st.session_state` (mai a livello globale, mai con `@st.cache_resource`).

## Struttura del progetto

```
hotel_recsys/
├── app.py                     # entry point Streamlit
├── data/
│   ├── generate_dataset.py    # generazione/normalizzazione dataset
│   └── hotels_dataset.csv     # generato
├── env/
│   └── hotel_env.py           # MultiAgentHotelEnv (Gymnasium)
├── marl/
│   ├── networks.py            # Actor (Sigmoid) + CentralizedCritic
│   ├── maddpg.py               # algoritmo MADDPG + replay buffer
│   └── train.py                # training loop (1000 episodi)
├── llm/
│   └── ollama_gateway.py      # parsing intento + generazione risposta
├── app_utils/
│   ├── state_manager.py       # memoria Markoviana + users_db.json
│   └── recommender.py         # matching geometrico (attori + KNN L2)
├── models/                    # pesi .pth salvati dal training
└── users_db.json              # generato a runtime
```

## Note matematiche

- Ricompensa MADDPG: `R = 1.0 - (distanza_euclidea(azioni, target) * 1.5)`, identica
  per tutti e 5 gli agenti (ricompensa condivisa, densa e continua).
- Aggiornamento Markoviano dello stato utente:
  `s_t = 0.2 * s_{t-1} + 0.8 * v_intento`, con troncamento (`clip`) in `[0, 1]`
  e componenti di `v_intento` ereditate da `s_{t-1}` quando l'LLM restituisce `null`.
