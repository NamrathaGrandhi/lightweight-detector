"""Dataset preparation workflow (thesis §3.4, §4.3).

Six steps, in the exact order of proposal §7.8 (Dataset Preparation
Workflow):

    1. Acquisition   — raw CSVs from two companion Kaggle releases,
                       'arielzilber/prompt-injection-in-the-wild' (malicious)
                       and '...-benign-evaluation-framework' (benign),
                       placed in archive/ (via `kaggle datasets download`).
    2. Sampling      — stratified random sample of SAMPLE_SIZE rows,
                       preserving the benign/malicious ratio.
    3. De-duplication— cosine similarity >= 0.97 between MiniLM embeddings;
                       the later occurrence in each near-duplicate pair is
                       dropped.
    4. Cleaning      — whitespace normalisation, encoding repair, length
                       filter (5..5000 chars), empty/corrupt/exact-duplicate
                       rows dropped.
    5. Splitting     — stratified 70/15/15 train/val/test, seed 42.
    6. Caching       — cleaned splits written to data/processed/ as parquet.

Sampling precedes de-duplication and cleaning (as the proposal specifies),
so the final corpus lands slightly under SAMPLE_SIZE after steps 3-4; the
per-step row accounting is printed and reported in thesis Chapter 4.

The raw Kaggle release aggregates several upstream sources into separate
CSV files. Because column layouts differ per file, ingestion is driven by
the FILE_LABEL_MAP below: each known filename is mapped to a label and the
name of its prompt column. Files not listed are skipped with a warning, so
a new dataset version fails loudly rather than silently mislabelling.
"""

from __future__ import annotations

import re
import sys
import unicodedata
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split

from pidetect.config import (
    BALANCE_SOURCES,
    DATA_PROCESSED,
    DATA_RAW,
    DEDUP_BLOCK_SIZE,
    DEDUP_COSINE_THRESHOLD,
    MAX_PROMPT_CHARS,
    MIN_PROMPT_CHARS,
    SAMPLE_SIZE,
    SEED,
    TEST_FRAC,
    TRAIN_FRAC,
    VAL_FRAC,
)

# filename -> (label, prompt-column candidates); label 1 = malicious, 0 = benign.
#
# Verified against the actual Kaggle download on 12 Aug 2026 (see
# docs/dataset_validation.md). Two points that cost real debugging time:
#   * the prompt column is 'Prompt' (capitalised), not 'prompt';
#   * 'malicous_deepset.csv' is misspelled in the upstream release.
# The attack files carry NO label column - the label is implied by the file,
# which is why this table exists at all.
# Candidate prompt-column names, matched CASE-INSENSITIVELY. Case matters in
# practice: this release uses 'Prompt' in the attack files but 'Text' in
# tapir.csv, and an exact-match lookup silently skipped 116,862 benign rows.
_P = ["prompt", "text", "question", "instruction", "input"]

FILE_LABEL_MAP: dict[str, tuple[int, list[str]]] = {
    # --- malicious: 'prompt-injection-in-the-wild' (Zilber, 2024a) -----------
    "forbidden_question_set_with_prompts.csv": (1, _P),
    # Mixed file - per-row labels come from ROW_RANGE_LABELS, not this entry.
    "predictionguard_df.csv": (1, _P),
    "jailbreak_prompts.csv": (1, _P),
    "malicous_deepset.csv": (1, _P),  # upstream misspelling, kept verbatim
    # 45,504 rows holding just 22 distinct jailbreak templates (a template x
    # forbidden-question cross product where only the index varies). Ingested
    # via COLLAPSE_ON_READ so its 5 genuinely unique prompts are not lost -
    # an earlier revision excluded the file on the false assumption that all
    # 22 appeared elsewhere; only 17 do.
    "forbidden_question_set_df.csv": (1, _P),
    # --- benign: 'prompt-injection-benign-evaluation-framework' (2024b) -----
    "tapir.csv": (0, _P),
    "platypus.csv": (0, _P),
    "super_glue_squad_v2.csv": (0, _P),
    "code.csv": (0, _P),
    "puffin.csv": (0, _P),
    "boolq.csv": (0, _P),
    "docRED.csv": (0, _P),
    "benign_deepset.csv": (0, _P),
}

# Files read with immediate exact de-duplication. Only for sources that are
# a cross product of a small template set, where loading every row costs
# hundreds of megabytes to yield a handful of distinct prompts.
COLLAPSE_ON_READ = {"forbidden_question_set_df.csv"}

# Files that are NOT a single class, despite having no label column.
#
# predictionguard_df.csv is two sources concatenated: an ordinary-prompt
# block (news classification, technical questions, general queries) followed
# by a block of generated injection strings. An earlier revision labelled the
# whole file malicious, which mislabelled 8,878 benign prompts as attacks and
# corrupted 84% of the attack class.
#
# The boundary is unambiguous and was located by scanning for the first
# sustained run of short imperative override strings: rows before it match
# that signature 0.0% of the time, rows after it 99.2%. The split is a
# property of how the file was assembled, not a judgement about individual
# prompts. `scripts/audit_homogeneity.py` re-derives it, and confirms this is
# the only file in the corpus affected.
ROW_RANGE_LABELS: dict[str, list[tuple[int, int, int]]] = {
    # filename: [(start_row_inclusive, end_row_exclusive, label), ...]
    "predictionguard_df.csv": [(0, 8878, 0), (8878, 10**9, 1)],
}

LABEL_COLUMN_CANDIDATES = ["label", "is_injection", "malicious", "target"]

_WS = re.compile(r"\s+")


# ---------------------------------------------------------------------------
# Step 1 — acquisition / ingestion
# ---------------------------------------------------------------------------
def load_raw(raw_dir: Path = DATA_RAW) -> pd.DataFrame:
    frames = []
    csvs = sorted(raw_dir.glob("*.csv"))
    skipped: list[str] = []
    if not csvs:
        sys.exit(
            f"No CSVs found in {raw_dir}. Download the dataset first:\n"
            "  kaggle datasets download -d arielzilber/prompt-injection-in-the-wild"
            f" -p {raw_dir} --unzip"
        )
    for path in csvs:
        spec = FILE_LABEL_MAP.get(path.name)
        if spec is None:
            print(f"  SKIPPED (not in FILE_LABEL_MAP): {path.name}")
            skipped.append(path.name)
            continue
        label, prompt_cols = spec

        # Read the header alone first, so only the columns actually needed are
        # loaded. These CSVs carry a stringified 384-dimensional 'embedding'
        # column that accounts for nearly all of their ~2 GB on disk; reading
        # it would cost gigabytes of RAM for data this pipeline never uses.
        header = pd.read_csv(path, nrows=0).columns.tolist()
        lookup = {c.lower(): c for c in header}
        prompt_col = next((lookup[c] for c in prompt_cols if c in lookup), None)
        if prompt_col is None:
            raise ValueError(
                f"{path.name} is in FILE_LABEL_MAP but has no recognisable "
                f"prompt column. Columns present: {header}. Add the right name "
                f"to _P rather than letting the file be dropped."
            )
        wanted = [prompt_col]
        label_col = None
        if label is None:
            label_col = next(
                (lookup[c] for c in LABEL_COLUMN_CANDIDATES if c in lookup), None
            )
            if label_col is None:
                raise ValueError(
                    f"{path.name} is mapped to label=None (expecting its own "
                    f"label column) but none was found in {header}."
                )
            wanted.append(label_col)

        df = pd.read_csv(path, usecols=wanted, on_bad_lines="skip",
                         encoding_errors="replace", low_memory=False)
        out = pd.DataFrame({"prompt": df[prompt_col].astype(str)})

        if path.name in ROW_RANGE_LABELS:
            # Mixed-class file: label by position, per ROW_RANGE_LABELS.
            out["label"] = -1
            for start, stop, value in ROW_RANGE_LABELS[path.name]:
                out.iloc[start:stop, out.columns.get_loc("label")] = value
            if (out["label"] < 0).any():
                raise ValueError(
                    f"{path.name}: ROW_RANGE_LABELS leaves "
                    f"{(out['label'] < 0).sum()} rows unlabelled"
                )
            counts = out["label"].value_counts().to_dict()
            print(f"  MIXED FILE {path.name}: "
                  f"{counts.get(0, 0):,} benign + {counts.get(1, 0):,} malicious "
                  f"(split by row range)")
        else:
            out["label"] = df[label_col].astype(int) if label_col else label
        out["source_file"] = path.name
        if path.name in COLLAPSE_ON_READ:
            before = len(out)
            out = out.drop_duplicates(subset=["prompt"]).reset_index(drop=True)
            print(f"  loaded {len(out):>7,} rows from {path.name} "
                  f"(collapsed from {before:,} exact duplicates)")
        else:
            print(f"  loaded {len(out):>7,} rows from {path.name}")
        frames.append(out)
    # Every mapped file that exists on disk must have been ingested. Losing one
    # silently changes the corpus composition and every number downstream.
    expected = {n for n in FILE_LABEL_MAP if (raw_dir / n).exists()}
    ingested = {p.name for p in csvs} - set(skipped)
    if expected - ingested:
        sys.exit(f"FATAL: mapped files present but not ingested: "
                 f"{sorted(expected - ingested)}")

    raw = pd.concat(frames, ignore_index=True)
    n_mal = int((raw.label == 1).sum())
    n_ben = int((raw.label == 0).sum())
    print(f"Raw corpus: {len(raw):,} rows ({n_mal:,} malicious / {n_ben:,} benign)")

    # A binary classifier cannot be trained on one class. The Kaggle release
    # 'prompt-injection-in-the-wild' contains attack prompts ONLY; the benign
    # class comes from the companion release
    # 'prompt-injection-benign-evaluation-framework'. Fail here rather than
    # producing a meaningless model.
    if n_ben == 0 or n_mal == 0:
        sys.exit(
            f"\nFATAL: corpus has only one class (malicious={n_mal}, benign={n_ben}).\n"
            "The benign class is missing. Download the companion dataset:\n"
            "  kaggle datasets download -d "
            "arielzilber/prompt-injection-benign-evaluation-framework "
            f"-p {raw_dir} --unzip\n"
            "then add its filenames to FILE_LABEL_MAP with label 0."
        )
    return raw


# ---------------------------------------------------------------------------
# Step 4 — cleaning
# ---------------------------------------------------------------------------
def clean(df: pd.DataFrame) -> pd.DataFrame:
    n0 = len(df)
    df = df.dropna(subset=["prompt"])
    df["prompt"] = df["prompt"].map(
        lambda t: _WS.sub(" ", unicodedata.normalize("NFKC", t)).strip()
    )
    df = df[df["prompt"].str.len().between(MIN_PROMPT_CHARS, MAX_PROMPT_CHARS)]
    df = df[df["prompt"] != ""]
    df = df.drop_duplicates(subset=["prompt"])  # exact duplicates
    print(f"Cleaning: {n0:,} -> {len(df):,} rows "
          f"(dropped {n0 - len(df):,} empty/short/long/exact-duplicate)")
    return df.reset_index(drop=True)


# ---------------------------------------------------------------------------
# Step 2 — stratified sampling
# ---------------------------------------------------------------------------
def _equal_share_quotas(counts: dict[str, int], total: int) -> dict[str, int]:
    """Split `total` across sources as evenly as their supply allows.

    Every source is offered total/k. Any source with fewer rows than that
    contributes all of them, and the shortfall is redistributed among the
    sources that still have spare rows. Repeats until nothing changes, so a
    single large source cannot dominate while smaller ones go unused.
    """
    quotas = {s: 0 for s in counts}
    remaining, active = total, set(counts)
    while remaining > 0 and active:
        share = max(remaining // len(active), 1)
        progressed = False
        for s in sorted(active):
            if remaining <= 0:
                break
            take = min(share, counts[s] - quotas[s], remaining)
            if take > 0:
                quotas[s] += take
                remaining -= take
                progressed = True
            if quotas[s] >= counts[s]:
                active.discard(s)
        if not progressed:
            break
    return quotas


def sample(df: pd.DataFrame, n: int = SAMPLE_SIZE) -> pd.DataFrame:
    if len(df) <= n:
        print(f"Sampling skipped: corpus ({len(df):,}) <= target ({n:,})")
        return df

    if not BALANCE_SOURCES:
        frac = n / len(df)
        sampled = (df.groupby("label", group_keys=False)
                     .sample(frac=frac, random_state=SEED)
                     .reset_index(drop=True))
        print(f"Sampled {len(sampled):,} rows "
              f"({(sampled.label == 1).mean():.1%} malicious), ratio preserved")
        return sampled

    # Preserve the class ratio, then balance across sources within each class.
    parts = []
    print("Sampling with source balancing:")
    for label in sorted(df["label"].unique()):
        cls = df[df["label"] == label]
        quota = round(n * len(cls) / len(df))
        counts = cls.groupby("source_file").size().to_dict()
        quotas = _equal_share_quotas(counts, quota)
        for src, take in sorted(quotas.items()):
            if take == 0:
                continue
            rows = cls[cls["source_file"] == src]
            parts.append(rows.sample(n=take, random_state=SEED))
            print(f"    label {label}  {src:<42} {take:>5,} of {counts[src]:>7,} "
                  f"({take / quota * 100:4.1f}% of class)")

    sampled = (pd.concat(parts, ignore_index=True)
                 .sample(frac=1.0, random_state=SEED)
                 .reset_index(drop=True))
    print(f"Sampled {len(sampled):,} rows "
          f"({(sampled.label == 1).mean():.1%} malicious), "
          f"class ratio preserved and sources balanced")
    return sampled


# ---------------------------------------------------------------------------
# Step 3 — near-duplicate removal on MiniLM embeddings
# ---------------------------------------------------------------------------
def dedup_near(df: pd.DataFrame, threshold: float = DEDUP_COSINE_THRESHOLD) -> pd.DataFrame:
    from pidetect.features.embeddings import EmbeddingEncoder

    print("Near-duplicate removal: encoding prompts with MiniLM ...")
    emb = EmbeddingEncoder().extract_batch(df["prompt"].tolist())

    # Embeddings are L2-normalised, so cosine similarity = dot product.
    # Blockwise comparison keeps memory bounded at ~block x n floats.
    n = len(df)
    keep = np.ones(n, dtype=bool)
    block = DEDUP_BLOCK_SIZE
    for start in range(0, n, block):
        stop = min(start + block, n)
        sims = emb[start:stop] @ emb.T          # (block, n)
        for i in range(start, stop):
            if not keep[i]:
                continue
            row = sims[i - start]
            dup = np.where(row >= threshold)[0]
            dup = dup[dup > i]                   # drop the LATER occurrence
            keep[dup] = False
    out = df[keep].reset_index(drop=True)
    print(f"Near-duplicates: {n:,} -> {len(out):,} rows "
          f"(removed {n - len(out):,} pairs at cosine >= {threshold})")
    return out


# ---------------------------------------------------------------------------
# Step 5 + 6 — split and cache
# ---------------------------------------------------------------------------
def split_and_cache(df: pd.DataFrame, out_dir: Path = DATA_PROCESSED) -> None:
    train, rest = train_test_split(
        df, test_size=(VAL_FRAC + TEST_FRAC), stratify=df["label"], random_state=SEED
    )
    val, test = train_test_split(
        rest,
        test_size=TEST_FRAC / (VAL_FRAC + TEST_FRAC),
        stratify=rest["label"],
        random_state=SEED,
    )
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, part in [("train", train), ("val", val), ("test", test)]:
        part = part.reset_index(drop=True)
        part.to_parquet(out_dir / f"{name}.parquet")
        print(f"  {name}: {len(part):,} rows "
              f"({(part.label == 1).mean():.1%} malicious) -> {name}.parquet")


def main() -> None:
    raw = load_raw()                 # Step 1  Acquisition
    sampled = sample(raw)            # Step 2  Sampling
    deduped = dedup_near(sampled)    # Step 3  De-duplication
    cleaned = clean(deduped)         # Step 4  Cleaning
    split_and_cache(cleaned)         # Steps 5-6  Splitting + Caching
    print("Dataset preparation complete.")


if __name__ == "__main__":
    main()
