"""
Genera (o normalizza) il dataset di hotel usato dal sistema di raccomandazione.

Se esiste un file 'hotels_raw.csv' nella stessa cartella con colonne:
    name, city, stars (1.0-5.0), category (Lusso/Resort | Standard | Budget/Motel),
    num_reviews, mood (Relax/Vacanza | Business/Lavoro), price_per_night (EUR)
viene usato quello. Altrimenti viene generato un dataset sintetico plausibile.
Se price_per_night manca dal raw, viene assunta una fascia media costante.

Produce 'hotels_dataset.csv' con le 5 feature normalizzate nello spazio [0,1]:
    reputazione, comfort, popolarita, leisure, prezzo
oltre alle colonne originali (name, city, stars, category, num_reviews, mood,
price_per_night).
"""

import os
import numpy as np
import pandas as pd

RNG_SEED = 42
N_SYNTH_HOTELS = 500

CITIES = [
    "Roma", "Milano", "Firenze", "Venezia", "Napoli", "Torino",
    "Bologna", "Palermo", "Verona", "Bari", "Genova", "Catania",
]

CATEGORY_COMFORT_MAP = {
    "Lusso/Resort": 0.95,
    "Standard": 0.50,
    "Budget/Motel": 0.10,
}

# Prezzo medio per notte (EUR) usato come base per la generazione sintetica;
# categoria e citta' introducono variazione realistica attorno a questa base.
CATEGORY_BASE_PRICE = {
    "Lusso/Resort": 260.0,
    "Standard": 110.0,
    "Budget/Motel": 45.0,
}
CITY_PRICE_MULTIPLIER = {
    "Roma": 1.15, "Milano": 1.20, "Venezia": 1.35, "Firenze": 1.15,
    "Napoli": 0.90, "Torino": 0.95, "Bologna": 1.00, "Palermo": 0.85,
    "Verona": 1.05, "Bari": 0.85, "Genova": 0.95, "Catania": 0.85,
}


def _synthesize_raw(n=N_SYNTH_HOTELS, seed=RNG_SEED) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    categories = rng.choice(
        list(CATEGORY_COMFORT_MAP.keys()), size=n, p=[0.25, 0.5, 0.25]
    )
    moods = rng.choice(["Relax/Vacanza", "Business/Lavoro"], size=n, p=[0.6, 0.4])
    cities = rng.choice(CITIES, size=n)

    # stelle correlate (debolmente) alla categoria per un dataset più realistico
    base_stars = rng.normal(3.4, 0.8, size=n)
    cat_bonus = np.array([0.6 if c == "Lusso/Resort" else (-0.4 if c == "Budget/Motel" else 0.0) for c in categories])
    stars = np.clip(base_stars + cat_bonus, 1.0, 5.0).round(1)

    num_reviews = rng.lognormal(mean=5.0, sigma=1.2, size=n).astype(int)
    num_reviews = np.clip(num_reviews, 3, 20000)

    names = [f"Hotel {chr(65 + i % 26)}{i}" for i in range(n)]

    # --- Prezzo per notte: base di categoria * moltiplicatore citta' * rumore ---
    base_price = np.array([CATEGORY_BASE_PRICE[c] for c in categories])
    city_mult = np.array([CITY_PRICE_MULTIPLIER.get(c, 1.0) for c in cities])
    price_noise = rng.normal(1.0, 0.15, size=n)
    price_per_night = np.clip(base_price * city_mult * price_noise, 20.0, 2000.0).round(0)

    return pd.DataFrame(
        {
            "name": names,
            "city": cities,
            "stars": stars,
            "category": categories,
            "num_reviews": num_reviews,
            "mood": moods,
            "price_per_night": price_per_night,
        }
    )


def build_dataset(raw_path: str = None, out_path: str = None) -> pd.DataFrame:
    here = os.path.dirname(os.path.abspath(__file__))
    raw_path = raw_path or os.path.join(here, "hotels_raw.csv")
    out_path = out_path or os.path.join(here, "hotels_dataset.csv")

    if os.path.exists(raw_path):
        df = pd.read_csv(raw_path)
    else:
        df = _synthesize_raw()

    # --- Reputazione: 1.0-5.0 stelle -> 0.0-1.0 ---
    df["reputazione"] = (df["stars"] - 1.0) / 4.0

    # --- Comfort: mappatura categoriale fissa ---
    df["comfort"] = df["category"].map(CATEGORY_COMFORT_MAP).fillna(0.5)

    # --- Popolarita: min-max scaling sul numero di recensioni ---
    min_rev, max_rev = df["num_reviews"].min(), df["num_reviews"].max()
    if max_rev > min_rev:
        df["popolarita"] = (df["num_reviews"] - min_rev) / (max_rev - min_rev)
    else:
        df["popolarita"] = 0.5

    # --- Prezzo: min-max scaling sul prezzo per notte (0=economico, 1=costoso) ---
    if "price_per_night" not in df.columns:
        # dataset raw fornito dall'utente senza prezzo: assume fascia media
        df["price_per_night"] = 120.0
    min_price, max_price = df["price_per_night"].min(), df["price_per_night"].max()
    if max_price > min_price:
        df["prezzo"] = (df["price_per_night"] - min_price) / (max_price - min_price)
    else:
        df["prezzo"] = 0.5

    # --- Leisure: profilo mood target ---
    # Relax/Vacanza -> valori alti (con lieve rumore), Business/Lavoro -> valori bassi
    rng = np.random.default_rng(RNG_SEED)
    noise = rng.normal(0.0, 0.05, size=len(df))
    base_leisure = np.where(df["mood"] == "Relax/Vacanza", 0.85, 0.15)
    df["leisure"] = np.clip(base_leisure + noise, 0.0, 1.0)

    df[["reputazione", "comfort", "popolarita", "leisure", "prezzo"]] = df[
        ["reputazione", "comfort", "popolarita", "leisure", "prezzo"]
    ].clip(0.0, 1.0)

    df.to_csv(out_path, index=False)
    return df


if __name__ == "__main__":
    dataset = build_dataset()
    print(f"Dataset generato con {len(dataset)} hotel.")
    print(dataset.head())
