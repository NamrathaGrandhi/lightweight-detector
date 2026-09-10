# lightweight black-box detector for prompt-injection detection

Reference implementation for the MSc thesis *"A lightweight, automated and scalable
prompt-security framework for detecting prompt-injection attacks against large
language models"* (Liverpool John Moores University, 2026).

The detector sits in front of an LLM and decides, per prompt, whether it is an
injection attack. It treats the protected model as a black box: no weights, no
activations, no sight of the system prompt, no second model call. Everything runs
CPU-only, and a full decision costs ~214 ms.

Three cheap signals are fused into one 401-dimensional vector:

| Signal | What it captures | Dims |
|---|---|---:|
| Text-surface | length, casing, punctuation, entropy, base64 runs, ~40 curated override phrases | 12 |
| Perplexity | how surprising the prompt is to a small scoring LM  aggregate, mean, variance, max, and *where* the max falls | 5 |
| Sentence embedding | `all-MiniLM-L6-v2`, L2-normalised | 384 |

Logistic Regression is the transparent baseline; XGBoost is the main model.

## Results

Held-out test set: 1,640 prompts (178 attacks, 1,462 legitimate), opened once.
Every metric carries a 95% percentile-bootstrap interval over 2,000 resamples.

| Model | Attacks caught | Legitimate blocked | F1 | FPR | ROC-AUC |
|---|---:|---:|---:|---:|---:|
| TF-IDF + LogReg (baseline) | 170 / 178 | 26 / 1,462 | 0.909 | 1.8% | 0.996 |
| Fusion + Qwen2.5-0.5B scorer | 169 | 31 | 0.894 | 2.1% | 0.995 |
| Fusion + GPT-2 scorer | 174 | 114 | 0.747 | 7.8% | 0.994 |
| Surface features only | 169 | 806 | 0.293 | 55.1% | 0.881 |
| Perplexity only | 161 | 1,032 | 0.235 | 70.6% | 0.756 |
| Length only *(diagnostic)* | 176 | 1,451 | 0.195 | 99.3% | 0.432 |

Three things are worth pulling out of that table.

**The scoring model is the limiting component, not the fusion design.** Swapping
GPT-2 (124M) for Qwen2.5-0.5B  changing nothing else  moves F1 by +0.148
[+0.113, +0.186] and cuts the false-positive rate from 7.8% to 2.1%. Of the 114
legitimate prompts the GPT-2 configuration blocked, the Qwen configuration blocks
31, and all 31 are a subset of the original 114: 83 false positives removed, none
introduced. Treat scorer selection as a first-order design decision.

**Fusion and a tuned lexical baseline are statistically indistinguishable.** With
an adequate scorer the paired difference is 0.015 [-0.019, +0.048]  an interval
crossing zero  and it stays that way at every catch rate from 80% to 98%. Most
published comparisons report lexical baselines without giving them the same tuning
and threshold selection as the proposed method. This one does (see
[Protocol](#protocol)), and the gap disappears.

**Detection is the wrong axis of comparison.** Every model here catches between
161 and 176 of 178 attacks  too narrow a range to choose between them. The number
of legitimate requests each blocks spans 26 to 1,451. The deployment decision lives
entirely in the second number.

The last row is a deliberate dud. A model given nothing but `char_len` catches more
attacks than anything else in the study, by blocking 99.3% of legitimate traffic.
It exists to test whether prompt length explains the headline result. Its ROC-AUC
of 0.432 is *below chance*, so it does not.

## Architecture

```
raw Kaggle CSVs
  -> prepare     clean, source-balanced sample of 12k, dedup >= 0.97 cosine, split 70/15/15
  -> featurize   12 + 5 + 384 -> 401-d, cached to parquet
  -> train       LogReg + XGBoost, 5-fold CV grid search, tau chosen on val at 95% recall
  -> baselines   perplexity-only, surface-only, TF-IDF, length-only  identical protocol
  -> evaluate    test set, once: Acc/P/R/F1/FPR/AUC + bootstrap CIs
  -> analysis    UMAP / t-SNE / silhouette over the embedding space
  -> latency     per-prompt wall-clock, batch size 1
  -> alt_scorer  re-run the perplexity block with a modern small LM
```

See [docs/architecture.md](docs/architecture.md) for the design decisions behind
each stage.

## Setup

Python 3.11 is required (torch and xgboost wheel availability).

```bash
python -m venv .venv
.venv\Scripts\activate            # Windows
source .venv/bin/activate         # Linux / macOS

pip install -r requirements-lock.txt   # exact environment that produced the results
pip install -e .
```

Use `requirements.txt` instead if you want compatible ranges rather than pins.

## Data

Two companion Kaggle releases are needed  one supplies the malicious class, the
other the benign class. Neither alone is sufficient, and the pipeline aborts with
an explicit message if only one class is present.

```bash
kaggle datasets download -d arielzilber/prompt-injection-in-the-wild -p archive --unzip
kaggle datasets download -d arielzilber/prompt-injection-benign-evaluation-framework -p archive --unzip
```

Requires a free Kaggle account and API token.

Neither release carries a label column  the class of a prompt is implied by the
file it sits in. That assumption is encoded explicitly in
`src/pidetect/data/prepare.py::FILE_LABEL_MAP`, and files absent from that table
are skipped with a warning rather than silently ingested.

**Audit the raw data before trusting it.** The assumption above held for twelve of
the thirteen files and failed for one:

```bash
python scripts/validate_raw.py archive     # rows, columns, labels, sample prompts
python scripts/validate_raw2.py            # duplication, cross-file overlap
python scripts/validate_benign.py          # per-source lengths, length confound
python scripts/audit_homogeneity.py        # does each file really hold one class?
python scripts/audit_deep.py               # same question, 20 slices per file
python scripts/audit_labels.py             # does each row's content match its label?
```

[docs/dataset_validation.md](docs/dataset_validation.md) records what these found,
including a 390 MB file holding 22 distinct prompts and a file that turned out to
be two sources concatenated with nothing marking the join.

## Reproduce

```bash
python scripts/run_all.py               # all eight stages, ~3-4 hours CPU-only
python scripts/run_all.py --from train  # resume from a later stage
python scripts/run_all.py --only evaluate
pytest
```

A failing stage is logged and the run continues to the next one  the stages are
only loosely coupled, and every stage caches its own artefacts, so a failure can be
re-run alone with `--only`. The exit code is non-zero if anything failed.

Every figure and table under `results/` is generated, never hand-edited. The seed is
fixed at 42 at every stochastic call site (`src/pidetect/config.py`), so a re-run
reproduces the published numbers rather than numbers like them.

`results/models/` and the per-prompt `misclassified_*.csv` dumps are not committed 
the first are large regenerable binaries, and the second reproduce attack prompts
verbatim. Both are written locally by `run_all.py`.

## Protocol

The same seven steps govern every model in the study, main models and baselines
alike, so that all comparisons are like-for-like:

1. Split 70/15/15, stratified by label. De-duplication and cleaning happen **before**
   the split, so no near-duplicate pair can straddle a boundary.
2. Extract features once, cache to parquet. Every model sees byte-identical inputs.
3. Tune by stratified 5-fold cross-validated grid search on the **training split only**,
   optimising F1. Class imbalance handled by inverse-frequency weighting inside the loss.
4. Choose the decision threshold on the **validation split**: the largest tau whose
   recall still meets the 95% target, which is also the tau minimising false positives
   at that recall.
5. Evaluate **once** on the held-out test set. No tuning follows.
6. Apply steps 1-5 identically to every baseline.
7. Run two validity diagnostics designed so that a *good* score would invalidate the
   study's own results.

That last point is the one worth stealing. The `length_only` model asks whether the
result is explained by prompt length; the `prefix_diagnostic` asks whether it is
explained by template openings that identify which dataset a prompt came from. Both
were specified before the results were known. The first came back clean. The second
did not  84% of full performance is still available from the first 20 characters
after source balancing, down from 94% before it  and that is reported as the
study's most serious threat to validity rather than quietly dropped.

## Repository layout

| Path | Purpose |
|---|---|
| `src/pidetect/config.py` | Every experiment constant  single source of truth |
| `src/pidetect/features/` | The three feature extractors, one file each |
| `src/pidetect/data/` | Six-step dataset preparation, feature caching |
| `src/pidetect/models/` | Main models and baselines, identical protocol |
| `src/pidetect/eval/` | One-shot test evaluation, bootstrap CIs, figures |
| `src/pidetect/analysis/` | Embedding-space analysis (UMAP, t-SNE, silhouette) |
| `scripts/` | End-to-end runner, data audits, validity diagnostics, ablations |
| `tests/` | Unit tests for the feature maths and the threshold rule |
| `results/` | Generated figures and tables |
| `docs/` | Architecture notes and the dataset validation report |

## Limitations

- **Template confound.** 84% of full performance remains available from the opening
  20 characters, so absolute figures are partly a property of this benchmark. What
  survives is the *relative ordering* of models, since all faced the identical corpus.
- **Machine-generated attacks.** A substantial share of the attack class consists of
  combinatorial expansions of a small template, which favours lexical methods and
  limits generalisation to human-written attacks.
- **Implied labels.** Classes are inferred from source files rather than independently
  annotated, and that assumption demonstrably failed once.
- **Sample size.** 178 test attacks cannot resolve differences of a few prompts, which
  is why several comparisons are reported as indistinguishable rather than as wins.
- **English-only, text-only, binary.** No multilingual, multimodal or attack-type
  classification.
- **Non-adaptive adversary.** Every figure measures performance against attacks
  collected before this detector existed. Publishing the feature set tells an attacker
  what to avoid. No detector of this class should be deployed as a sole defence.

## Ethics

The corpus contains no personal data and no human participants were involved. What
this repository produces is a defensive classifier  it adds no new attack technique,
and every attack it examines was already publicly available. Attack prompts are
characterised structurally rather than reproduced, and the per-prompt error dumps are
excluded from version control for that reason.

## Citation

```bibtex
@mastersthesis{grandhi2026pidetector,
  author = {Grandhi, Namratha},
  title  = {A Lightweight, Automated and Scalable Prompt-Security Framework for
            Detecting Prompt-Injection Attacks Against Large Language Models},
  school = {Liverpool John Moores University},
  year   = {2026}
}
```

