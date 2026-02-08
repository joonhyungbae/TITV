"""
29_imdb_fetch.py  –  Download & parse IMDb TSV dumps → actor × director × year

Pipeline (4-pass streaming):
  1. title.basics   → valid movie tconsts + year
  2. title.crew     → tconst → primary director
  3. title.principals → actor–title links (actors/actresses only)
  4. name.basics    → actor birth years

Output: data/imdb_actor_credits.json
  { actor_nconst: [ {"year": int, "director": director_nconst}, ... ], ... }
"""

import csv
import gzip
import json
import os
import time
import urllib.request
from collections import defaultdict
from pathlib import Path

from tqdm import tqdm

# ── paths ─────────────────────────────────────────────────────────────────
DATA_DIR = Path(__file__).resolve().parent.parent / "data"
IMDB_DIR = DATA_DIR / "imdb"
OUTPUT_PATH = DATA_DIR / "imdb_actor_credits.json"

BASE_URL = "https://datasets.imdbws.com"
NEEDED_FILES = [
    "title.basics.tsv.gz",
    "title.crew.tsv.gz",
    "title.principals.tsv.gz",
    "name.basics.tsv.gz",
]

# ── selection criteria ────────────────────────────────────────────────────
MIN_YEAR = 1950
MAX_YEAR = 2024
VALID_TITLE_TYPES = {"movie"}
VALID_CATEGORIES = {"actor", "actress"}
MIN_CREDITS = 20
MIN_CAREER_SPAN = 15
TARGET_ACTORS = 500


# ── download ──────────────────────────────────────────────────────────────

def download_file(filename: str, dest_dir: Path) -> Path:
    """Download an IMDb TSV file if not already present."""
    dest = dest_dir / filename
    if dest.exists():
        sz = dest.stat().st_size
        print(f"  {filename} already exists ({sz / 1e6:.0f} MB), skipping download")
        return dest

    url = f"{BASE_URL}/{filename}"
    print(f"  Downloading {url} …")

    # Get file size for progress bar
    req = urllib.request.Request(url, method="HEAD")
    with urllib.request.urlopen(req) as resp:
        total = int(resp.headers.get("Content-Length", 0))

    pbar = tqdm(total=total, unit="B", unit_scale=True, desc=f"  {filename}")

    def _reporthook(block_num, block_size, _total):
        pbar.update(block_size)

    urllib.request.urlretrieve(url, dest, reporthook=_reporthook)
    pbar.close()
    return dest


def download_all():
    """Download all needed IMDb TSV files."""
    IMDB_DIR.mkdir(parents=True, exist_ok=True)
    paths = {}
    for fn in NEEDED_FILES:
        paths[fn] = download_file(fn, IMDB_DIR)
    return paths


# ── streaming TSV reader ──────────────────────────────────────────────────

def iter_tsv_gz(path: Path, desc: str = ""):
    """Stream rows from a gzip TSV file as dicts."""
    file_size = os.path.getsize(path)
    raw_fh = open(path, "rb")
    gz_fh = gzip.open(raw_fh, "rt", encoding="utf-8")

    pbar = tqdm(total=file_size, unit="B", unit_scale=True, unit_divisor=1024,
                desc=desc, miniters=1)
    last_pos = 0

    reader = csv.DictReader(gz_fh, delimiter="\t", quoting=csv.QUOTE_NONE)
    for row in reader:
        pos = raw_fh.tell()
        if pos != last_pos:
            pbar.update(pos - last_pos)
            last_pos = pos
        yield row

    pbar.update(file_size - last_pos)
    pbar.close()
    gz_fh.close()
    raw_fh.close()


# ── main pipeline ─────────────────────────────────────────────────────────

def main():
    print("=" * 70)
    print("IMDb FETCH: Download & Parse → actor × director × year")
    print("=" * 70)
    print(f"\n  Criteria: movies only, {MIN_YEAR}–{MAX_YEAR}, "
          f"≥{MIN_CREDITS} credits, ≥{MIN_CAREER_SPAN}-yr span")
    print(f"  Target : up to {TARGET_ACTORS} actors\n")

    t0 = time.time()

    # ── Step 0: Download ──
    print("Step 0: Downloading IMDb data files …")
    paths = download_all()

    # ── Step 1: title.basics → valid movies + year ──
    print("\nStep 1: Parsing title.basics (movies, non-adult) …")
    movie_year: dict[str, int] = {}
    for row in iter_tsv_gz(paths["title.basics.tsv.gz"], "title.basics"):
        if row["titleType"] not in VALID_TITLE_TYPES:
            continue
        if row.get("isAdult", "0") == "1":
            continue
        yr = row.get("startYear", "\\N")
        if yr == "\\N":
            continue
        try:
            year = int(yr)
        except ValueError:
            continue
        if year < MIN_YEAR or year > MAX_YEAR:
            continue
        movie_year[row["tconst"]] = year

    print(f"  Valid movies: {len(movie_year):,}")

    # ── Step 2: title.crew → primary director per movie ──
    print("\nStep 2: Parsing title.crew (directors) …")
    movie_director: dict[str, str] = {}
    for row in iter_tsv_gz(paths["title.crew.tsv.gz"], "title.crew"):
        tc = row["tconst"]
        if tc not in movie_year:
            continue
        dirs_str = row.get("directors", "\\N")
        if dirs_str == "\\N" or not dirs_str:
            continue
        # Take the first (primary) director
        primary = dirs_str.split(",")[0].strip()
        if primary and primary != "\\N":
            movie_director[tc] = primary

    print(f"  Movies with director: {len(movie_director):,}")

    # ── Step 3: title.principals → actor credits ──
    print("\nStep 3: Parsing title.principals (actor/actress credits) …")
    actor_credits: dict[str, list] = defaultdict(list)
    n_credits = 0
    for row in iter_tsv_gz(paths["title.principals.tsv.gz"], "title.principals"):
        tc = row["tconst"]
        if tc not in movie_director:
            continue
        cat = row.get("category", "")
        if cat not in VALID_CATEGORIES:
            continue
        nc = row["nconst"]
        actor_credits[nc].append({
            "year": movie_year[tc],
            "director": movie_director[tc],
        })
        n_credits += 1

    print(f"  Total actor credits: {n_credits:,}")
    print(f"  Unique actors: {len(actor_credits):,}")

    # ── Step 4: name.basics → birth years ──
    print("\nStep 4: Parsing name.basics (birth years) …")
    relevant_nconsts = set(actor_credits.keys())
    actor_info: dict[str, dict] = {}
    for row in iter_tsv_gz(paths["name.basics.tsv.gz"], "name.basics"):
        nc = row["nconst"]
        if nc not in relevant_nconsts:
            continue
        by = row.get("birthYear", "\\N")
        birth_year = None
        if by != "\\N":
            try:
                birth_year = int(by)
            except ValueError:
                pass
        actor_info[nc] = {
            "name": row.get("primaryName", nc),
            "birth_year": birth_year,
        }

    print(f"  Actors with info: {len(actor_info):,}")

    # ── Step 5: Filter actors ──
    print(f"\nStep 5: Filtering (≥{MIN_CREDITS} credits, ≥{MIN_CAREER_SPAN}-yr span) …")
    valid: dict[str, list] = {}
    valid_meta: dict[str, dict] = {}

    # Sort actors by credit count (descending) for stable selection
    sorted_actors = sorted(actor_credits.items(), key=lambda x: -len(x[1]))

    for nc, credits in tqdm(sorted_actors, desc="Filtering", unit="actor"):
        if len(credits) < MIN_CREDITS:
            break  # Sorted descending, no more can pass
        years = [c["year"] for c in credits]
        span = max(years) - min(years)
        if span < MIN_CAREER_SPAN:
            continue
        valid[nc] = credits
        valid_meta[nc] = actor_info.get(nc, {"name": nc, "birth_year": None})
        if len(valid) >= TARGET_ACTORS:
            break

    print(f"  Valid actors: {len(valid)}")

    if valid:
        tot = sum(len(c) for c in valid.values())
        spans = [max(c["year"] for c in cs) - min(c["year"] for c in cs)
                 for cs in valid.values()]
        n_with_by = sum(1 for m in valid_meta.values() if m["birth_year"] is not None)
        print(f"  Total credits: {tot:,}")
        print(f"  Career span: mean={sum(spans)/len(spans):.1f}, "
              f"min={min(spans)}, max={max(spans)}")
        print(f"  Credits/actor: mean={tot/len(valid):.1f}")
        print(f"  Actors with birth year: {n_with_by}/{len(valid)}")

        # Show a few example actors
        print(f"\n  Sample actors:")
        for i, (nc, cs) in enumerate(valid.items()):
            if i >= 5:
                break
            meta = valid_meta[nc]
            yrs = [c["year"] for c in cs]
            dirs = set(c["director"] for c in cs)
            print(f"    {meta['name']}: {len(cs)} credits, "
                  f"{min(yrs)}–{max(yrs)}, {len(dirs)} unique directors")

    # ── Save ──
    # Include birth_year in the output for panel building
    output = {}
    for nc, credits in valid.items():
        meta = valid_meta.get(nc, {})
        output[nc] = {
            "name": meta.get("name", nc),
            "birth_year": meta.get("birth_year"),
            "credits": credits,
        }

    with open(OUTPUT_PATH, "w", encoding="utf-8") as f:
        json.dump(output, f, ensure_ascii=False)
    print(f"\n  Saved to {OUTPUT_PATH}")
    print(f"  Total time: {time.time() - t0:.1f} s")
    print("DONE.")


if __name__ == "__main__":
    main()
