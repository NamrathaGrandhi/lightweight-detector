# Dataset Validation Report

**Date:** 12 August 2026
**Source validated:** the contents of `archive/` — the Kaggle release
*prompt-injection-in-the-wild* (Zilber, 2024), and later its benign companion.
**Scripts:** `scripts/validate_raw.py`, `scripts/validate_raw2.py` (re-runnable).

This is the working record of what the raw data turned out to contain. It is kept in
the repository because the findings changed the pipeline, and because every claim
below is reproducible from the scripts named.

## Verdict

**The dataset is usable, but it is not what the proposal describes, and on its own it cannot support the study.** Three findings, in order of severity.

### Finding 1 — There is no benign class, and no label column (blocking)

The proposal (§8.3) states the dataset holds "over 200,000 labelled prompts … each row carrying a prompt string and a binary label ∈ {benign, malicious}". The actual download contains **no label column in any file**, and every file is an attack source:

| File | Rows | Unique prompts | Content |
|---|---:|---:|---|
| `forbidden_question_set_df.csv` | 45,504 | **22** | DAN/jailbreak templates |
| `forbidden_question_set_with_prompts.csv` | 21,060 | 21,060 | jailbreak templates × forbidden questions |
| `jailbreak_prompts.csv` | 2,071 | 1,558 | jailbreak personas |
| `malicous_deepset.csv` *(sic)* | 263 | 263 | deepset injection benchmark, malicious half |
| `predictionguard_df.csv` | 17,678 | 17,671 | injection/jailbreak prompts |
| **Total** | **86,576** | **40,479** | **all malicious** |

Columns are `Prompt`, `Length`, `Perplexity`, `embedding`, plus index columns. Note that `Perplexity` and `embedding` are *already computed* by the dataset author — this study computes its own from the raw text, so those columns are ignored.

A binary classifier cannot be trained on one class. `prepare.py` now aborts with an explicit message if only one class is present, rather than silently producing a meaningless model.

**Root cause:** `prompt-injection-in-the-wild` is the *adversarial* half of the author's data. The author's GitHub project (`ariel-zilber/prompt-security`) uses a **separate benign dataset**, published on Kaggle as **`arielzilber/prompt-injection-benign-evaluation-framework`**. That companion release was not downloaded. This is a download gap, not a flaw in the proposal's choice of source.

### Finding 2 — Severe duplication inflates the apparent corpus size (fixed)

`forbidden_question_set_df.csv` is 390 MB and 45,504 rows, but contains **only 22 unique prompts**, each repeated ~2,068 times — it is a template × forbidden-question cross product where only the index varies.

An earlier revision of this report claimed all 22 templates appear in the other files and excluded the file entirely. **That was wrong: only 17 of the 22 appear elsewhere**, so excluding the file silently discarded five unique prompts. The file is now ingested with exact de-duplication applied at read time (`COLLAPSE_ON_READ` in `prepare.py`), which keeps all 22 distinct prompts while loading 22 rows instead of 45,504.

Across the malicious release, 86,576 rows reduce to **40,479 unique prompts** before any semantic de-duplication. The proposal's "over 200,000 labelled prompts" figure is therefore not supported by the data; Chapter 4 reports the true numbers.

### Finding 3 — Length is a moderate, measured confound (quantified)

`char_len` is feature #1 of the text-surface block, so any systematic length difference between the classes is a route to a high score for the wrong reason. Both classes turn out to be **internally bimodal**, which is what keeps the confound moderate rather than fatal.

Median prompt length by source (measured, `scripts/validate_benign.py`):

| Source | Class | Rows | Median chars |
|---|---|---:|---:|
| `forbidden_question_set_with_prompts.csv` | malicious | 21,060 | 2,460 |
| `jailbreak_prompts.csv` | malicious | 2,071 | 1,723 |
| `malicous_deepset.csv` | malicious | 263 | 126 |
| `predictionguard_df.csv` | malicious | 17,678 | **52** |
| `docRED.csv` | benign | 998 | 956 |
| `super_glue_squad_v2.csv` | benign | 11,873 | 877 |
| `boolq.csv` | benign | 3,270 | 618 |
| `platypus.csv` | benign | 24,926 | 242 |
| `code.csv` | benign | 10,001 | 226 |
| `tapir.csv` | benign | 116,862 | 163 |
| `puffin.csv` | benign | 6,994 | 115 |
| `benign_deepset.csv` | benign | 399 | 42 |

Crucially, **44% of malicious rows come from short sources** (PredictionGuard at 52 and deepset at 126 characters), and several benign sources are long (DocRED 956, SQuAD 877, BoolQ 618). Pooled: malicious median 1,354, benign median 173.

The decisive measurement is the separability of the classes on length alone, which equals the AUC of a length-only ranker:

> **Length-only AUC = 0.697** (1.00 = length alone separates the classes perfectly; 0.50 = length carries no information).

Length is therefore informative but far from sufficient. If the fused detector reaches the target F1 ≥ 0.90, it cannot be explained by length. The length-only baseline (`baselines.py`) reports the full metric set at the same operating point so this argument rests on a measured number rather than an assertion.

## Final corpus

Both releases are now present in `archive/`:

| Class | Sources | Rows |
|---|---:|---:|
| Malicious | 5 files (Zilber, 2024a) | 41,094 after collapsing the duplicate-heavy file |
| Benign | 8 files (Zilber, 2024b) | 175,323 |

The benign class is drawn from eight distinct collections — instruction-following (Platypus, Puffin, TAPIR), reading comprehension (SQuAD v2, BoolQ, DocRED), code (`code.csv`), and the benign half of the deepset benchmark — which gives the negative class genuine stylistic breadth rather than a single register.

## Actions taken

- `FILE_LABEL_MAP` rewritten against the real filenames, the real column name (`Prompt`, capitalised) and the upstream misspelling `malicous_deepset.csv`; all eight benign files added with label 0.
- `forbidden_question_set_df.csv` ingested with read-time exact de-duplication rather than excluded, after the "all 22 appear elsewhere" claim was found to be false.
- Hard failure added when the corpus has fewer than two classes.
- `source_file` carried through the feature cache so Chapter 5 can report recall and false-positive rate per attack family and per benign source.
- Validation scripts committed so every check above is reproducible and citable in Chapter 4.
