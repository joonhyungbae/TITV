"""
30_imdb_panel.py
Build person-year panel for film industry from IMDb actor-credits data.
Maps: director = institution (gatekeeper), film credit = career event.
Plateau = 5-year gap without film credits.
"""

import json
from pathlib import Path

import numpy as np
import pandas as pd
from tqdm import tqdm

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
INPUT_PATH = DATA_DIR / "imdb_actor_credits.json"
OUTPUT_PATH = DATA_DIR / "imdb_panel.csv"

CENSOR_YEAR = 2023
CUTPOINT = 10
PLATEAU_WINDOW = 5


def load_imdb_data():
    """Load imdb_actor_credits.json."""
    if not INPUT_PATH.exists():
        raise FileNotFoundError(
            f"IMDb data not found at {INPUT_PATH}. Run 29_imdb_fetch.py first."
        )
    with open(INPUT_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def build_events_df(actor_data):
    """Convert actor_data dict to events DataFrame."""
    rows = []
    for actor_id, info in actor_data.items():
        for c in info["credits"]:
            rows.append({
                "actor_id": actor_id,
                "year": c["year"],
                "director": c["director"],
                "event_type": "film_credit",
            })
    return pd.DataFrame(rows)


def build_actors_df(actor_data):
    """Build actors DataFrame with career_start_year, num_events, birth_year."""
    rows = []
    for actor_id, info in actor_data.items():
        credits = info["credits"]
        if len(credits) < 5:
            continue
        years = [c["year"] for c in credits]
        career_start = min(years)
        span = max(years) - career_start
        if span < 14:
            continue

        # Use actual birth year if available, otherwise proxy
        birth_year = info.get("birth_year")
        if birth_year is None:
            birth_year = career_start - 25  # Proxy: assume career starts around age 25

        rows.append({
            "actor_id": actor_id,
            "actor_name": info.get("name", actor_id),
            "career_start_year": career_start,
            "num_events": len(credits),
            "birth_year": birth_year,
        })
    return pd.DataFrame(rows)


def detect_plateau(events_df, actor_id, career_start, window=5, censor_year=2023):
    """
    Plateau = window consecutive years with no film credits.
    Returns (plateau_occurred, plateau_year, time_to_plateau).
    """
    actor_events = events_df[
        (events_df["actor_id"] == actor_id)
        & (events_df["year"] >= career_start)
        & (events_df["year"] <= censor_year)
    ]
    credit_years = set(actor_events["year"].unique())
    end_year = min(
        int(actor_events["year"].max()) if len(actor_events) > 0 else censor_year,
        censor_year,
    )
    for start_y in range(int(career_start), end_year - window + 2):
        gap_years = set(range(start_y, start_y + window))
        if not gap_years.intersection(credit_years):
            return True, start_y, start_y - career_start
    return False, None, end_year - career_start


def compute_network_size_stability(events_up_to):
    """Compute network size and stability from cumulative events.

    director = institution (gatekeeper).
    stability = cumulative credits / unique directors.
    """
    if len(events_up_to) == 0:
        return 0, 0.0
    directors = events_up_to["director"].dropna().unique()
    network_size = len(directors)
    total_events = len(events_up_to)
    stability = total_events / network_size if network_size > 0 else 0.0
    return network_size, stability


def build_person_year_panel(df_actors, df_events, cutpoint=10, censor_year=2023):
    """Build person-year panel for Cox time-varying models."""
    # First detect plateaus for all actors
    plateau_info = {}
    for _, row in tqdm(df_actors.iterrows(), total=len(df_actors),
                       desc="Detecting plateaus", unit="actor"):
        aid = row["actor_id"]
        cs = row["career_start_year"]
        occurred, p_year, _ = detect_plateau(
            df_events, aid, int(cs), window=PLATEAU_WINDOW, censor_year=censor_year
        )
        plateau_info[aid] = (occurred, p_year)

    records = []
    for _, row in tqdm(df_actors.iterrows(), total=len(df_actors),
                       desc="Building panel", unit="actor"):
        aid = row["actor_id"]
        cs = int(row["career_start_year"])
        if aid not in plateau_info:
            continue
        occurred, p_year = plateau_info[aid]
        end_year = int(p_year) if occurred else censor_year
        birth_year = row.get("birth_year", cs - 25)

        actor_events = df_events[
            (df_events["actor_id"] == aid)
            & (df_events["year"] <= end_year)
            & (df_events["year"] <= censor_year)
        ].copy()

        for year in range(cs, end_year + 1):
            career_year = year - cs
            events_up_to = actor_events[actor_events["year"] <= year]
            net_size, net_stability = compute_network_size_stability(events_up_to)
            cum_val = float(len(events_up_to))
            is_last_year = year == end_year
            event = 1 if (is_last_year and occurred) else 0

            records.append({
                "author_id": aid,   # Keep column name for replication script compat
                "year": year,
                "career_year": career_year,
                "start": career_year,
                "stop": career_year + 1,
                "event": event,
                "network_size": max(net_size, 0.5),
                "network_stability": net_stability,
                "cumulative_validation": cum_val,
                "post_cutpoint": 1 if career_year >= cutpoint else 0,
                "birth_year": birth_year,
            })

    df = pd.DataFrame(records)
    df["birth_year"] = df["birth_year"].fillna(df["birth_year"].median())
    return df


def main():
    print("=" * 70)
    print("IMDb PANEL: Person-Year for Pilot Replication")
    print("=" * 70)

    actor_data = load_imdb_data()
    print(f"Loaded {len(actor_data)} actors")

    df_events = build_events_df(actor_data)
    df_actors = build_actors_df(actor_data)
    print(f"Events: {len(df_events):,}, Actors: {len(df_actors)}")

    panel = build_person_year_panel(
        df_actors, df_events, cutpoint=CUTPOINT, censor_year=CENSOR_YEAR
    )
    print(f"\nPanel: {len(panel):,} person-years, {panel['author_id'].nunique()} actors")
    print(f"  Events (plateaus): {panel['event'].sum()}")
    print(f"  Pre-decade events: {panel[panel['post_cutpoint'] == 0]['event'].sum()}")
    print(f"  Post-decade events: {panel[panel['post_cutpoint'] == 1]['event'].sum()}")

    panel.to_csv(OUTPUT_PATH, index=False)
    print(f"\nSaved to {OUTPUT_PATH}")
    print("DONE.")


if __name__ == "__main__":
    main()
