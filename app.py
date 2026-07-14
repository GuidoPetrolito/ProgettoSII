# === Vincoli critici macOS Apple Silicon: DEVONO essere le primissime righe ===
import os
os.environ["OBJC_DISABLE_INITIALIZE_FORK_SAFETY"] = "YES"
os.environ["KMP_DUPLICATE_LIB_OK"] = "TRUE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
# ================================================================================

import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st
import torch

torch.set_num_threads(1)  # vincolo esplicito richiesto per stabilita' su Apple Silicon

from data.generate_dataset import build_dataset
from env.hotel_env import FEATURE_NAMES
from marl.networks import Actor
from llm.ollama_gateway import parse_intent, generate_response
from app_utils.state_manager import get_user_state, get_user_history, get_user_touched, apply_intent_update
from app_utils.recommender import recommend, FEATURE_ORDER

MODELS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "models")
DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "data")

st.set_page_config(page_title="Hotel Recommender MARL+LLM", layout="wide")


# --------------------------------------------------------------------------- #
# Caricamento dati e modelli (MAI a livello globale, MAI @st.cache_resource
# per gli attori PyTorch: i riferimenti cross-thread corrompono i puntatori
# di memoria C++ durante i rerun di Streamlit su macOS).
# --------------------------------------------------------------------------- #
@st.cache_data
def load_hotel_dataset() -> pd.DataFrame:
    out_path = os.path.join(DATA_DIR, "hotels_dataset.csv")
    if os.path.exists(out_path):
        return pd.read_csv(out_path)
    return build_dataset()


def load_actors_into_session():
    """Carica gli attori PyTorch UNA SOLA VOLTA in st.session_state."""
    if "actors" in st.session_state:
        return
    actors = {}
    for name in FEATURE_NAMES:
        path = os.path.join(MODELS_DIR, f"attore_{name}.pth")
        actor = Actor()
        if os.path.exists(path):
            state_dict = torch.load(path, map_location="cpu")
            actor.load_state_dict(state_dict)
        actor.eval()
        actors[name] = actor
    st.session_state["actors"] = actors
    st.session_state["models_loaded_from_disk"] = all(
        os.path.exists(os.path.join(MODELS_DIR, f"attore_{n}.pth")) for n in FEATURE_NAMES
    )


def init_session_state():
    load_actors_into_session()
    if "username" not in st.session_state:
        st.session_state["username"] = None
    if "chat_history" not in st.session_state:
        st.session_state["chat_history"] = []
    if "last_recommendation" not in st.session_state:
        st.session_state["last_recommendation"] = None
    if "state_history" not in st.session_state:
        st.session_state["state_history"] = []  # per lo scatter plot in tempo reale


init_session_state()
hotels_df = load_hotel_dataset()


# --------------------------------------------------------------------------- #
# Sidebar: login
# --------------------------------------------------------------------------- #
with st.sidebar:
    st.header("Accesso utente")
    username_input = st.text_input("Username", value=st.session_state.get("username") or "")
    if st.button("Login / Cambia utente"):
        new_username = username_input.strip() or "guest"
        st.session_state["username"] = new_username
        st.session_state["chat_history"] = []
        # Ricostruisce il grafico dallo storico persistito (users_db.json), invece
        # di ripartire vuoto: un profilo gia' esistente riprende la sua traiettoria.
        st.session_state["state_history"] = get_user_history(new_username)
        st.session_state["last_recommendation"] = None
        st.rerun()

    if st.session_state["username"]:
        st.success(f"Connesso come: {st.session_state['username']}")
        current_state = get_user_state(st.session_state["username"])
        st.caption("Profilo preferenze attuale")
        st.json(current_state)
    else:
        st.info("Inserisci uno username per iniziare.")

    if not st.session_state.get("models_loaded_from_disk"):
        st.warning(
            "Pesi degli attori non trovati in models/. "
            "Esegui `python -m marl.train` per addestrare il modello. "
            "Nel frattempo verranno usati pesi non addestrati (casuali)."
        )


# --------------------------------------------------------------------------- #
# Vetrina in alto: elementi raccomandati / ultima raccomandazione
# --------------------------------------------------------------------------- #
st.title("🏨 Hotel Recommender — LLM + MADDPG")

showcase = st.container()
with showcase:
    st.subheader("Vetrina")
    if st.session_state["last_recommendation"] is not None:
        row = st.session_state["last_recommendation"]
        cols = st.columns([1, 3])
        with cols[0]:
            st.metric("Distanza dal profilo", f"{row.get('distance', 0):.3f}")
        with cols[1]:
            st.markdown(
                f"**{row['name']}** — {row['city']}  \n"
                f"{row['category']} · {row['stars']}★ · {int(row['num_reviews'])} recensioni · "
                f"mood: {row['mood']} · **{row['price_per_night']:.0f}€/notte**"
            )
    else:
        st.caption("Nessuna raccomandazione ancora. Effettua il login e scrivi una richiesta in chat.")
        top_preview = hotels_df.sort_values("reputazione", ascending=False).head(6)
        st.dataframe(
            top_preview[["name", "city", "stars", "category", "num_reviews", "mood", "price_per_night"]],
            width='stretch',
            hide_index=True,
        )

st.divider()

# --------------------------------------------------------------------------- #
# Layout inferiore: chat (sx) + scatter plot stato in tempo reale (dx)
# --------------------------------------------------------------------------- #
chat_col, plot_col = st.columns(2)

with chat_col:
    st.subheader("💬 Assistente di viaggio")
    chat_box = st.container(height=380)
    with chat_box:
        for role, msg in st.session_state["chat_history"]:
            with st.chat_message(role):
                st.write(msg)

    user_query = st.chat_input(
        "Es. 'Resort di lusso a Firenze per una vacanza rilassante'",
        disabled=not st.session_state["username"],
    )

    if user_query:
        st.session_state["chat_history"].append(("user", user_query))

        # 1. Estrazione intento via LLM (Ollama, JSON rigido)
        intent = parse_intent(user_query)

        # 2. Aggiornamento Markoviano dello stato persistente
        new_state = apply_intent_update(
            username=st.session_state["username"],
            intent=intent,
            target_city=intent.get("target_city"),
        )
        st.session_state["state_history"].append(dict(new_state))

        # 3. Matching geometrico: attori MADDPG + KNN (L2 pesata) sul dataset filtrato.
        # Le dimensioni mai richieste esplicitamente (touched) pesano meno, cosi' non
        # competono alla pari con quelle che l'utente ha davvero specificato.
        touched = get_user_touched(st.session_state["username"])
        best_row, action_vector, filtered_df = recommend(
            hotels_df, st.session_state["actors"], new_state, intent.get("target_city"), touched=touched
        )
        st.session_state["last_recommendation"] = best_row

        # 4. Risposta conversazionale in linguaggio naturale
        reply = generate_response(best_row.to_dict(), user_query)

        st.session_state["chat_history"].append(("assistant", reply))
        st.rerun()

with plot_col:
    st.subheader("📈 Monitoraggio stato in tempo reale")
    history = st.session_state["state_history"]
    if len(history) == 0:
        st.caption("Il grafico mostrera' l'evoluzione del profilo utente (comfort vs leisure) man mano che chatti.")
    else:
        # Asse economico (prezzo) vs asse lusso (comfort): mostra direttamente
        # come evolve il compromesso tra scelte economiche e lussuose dell'utente.
        comfort_vals = [h["comfort"] for h in history]
        prezzo_vals = [h["prezzo"] for h in history]

        fig = go.Figure()
        fig.add_trace(
            go.Scatter(
                x=prezzo_vals,
                y=comfort_vals,
                mode="lines+markers",
                marker=dict(
                    size=10,
                    color=list(range(len(history))),
                    colorscale="Viridis",
                    showscale=False,
                ),
                text=[f"turno {i+1}" for i in range(len(history))],
                name="Traiettoria profilo",
            )
        )
        if st.session_state["last_recommendation"] is not None:
            row = st.session_state["last_recommendation"]
            fig.add_trace(
                go.Scatter(
                    x=[row["prezzo"]],
                    y=[row["comfort"]],
                    mode="markers",
                    marker=dict(size=14, color="red", symbol="star"),
                    name="Hotel raccomandato",
                )
            )
        fig.update_layout(
            xaxis_title="prezzo (0=economico, 1=costoso)",
            yaxis_title="comfort (0=essenziale, 1=lusso)",
            xaxis=dict(range=[0, 1]),
            yaxis=dict(range=[0, 1]),
            height=380,
            margin=dict(l=10, r=10, t=10, b=10),
        )
        st.plotly_chart(fig, width='stretch')

        with st.expander("Profilo utente vs hotel raccomandato (per debug)"):
            if st.session_state["last_recommendation"] is not None:
                row = st.session_state["last_recommendation"]
                from env.hotel_env import FEATURE_NAMES as _FEATS

                compare = {
                    "profilo utente (target)": {k: round(history[-1][k], 3) for k in _FEATS},
                    "hotel raccomandato (dati reali dal dataset)": {
                        k: round(float(row[k]), 3) for k in _FEATS
                    },
                    "distanza L2 pesata": round(float(row.get("distance", 0)), 3),
                }
                st.json(compare)
            else:
                st.json({"profilo utente (target)": {k: round(history[-1][k], 3) for k in ["reputazione", "comfort", "popolarita", "leisure", "prezzo"]}})
