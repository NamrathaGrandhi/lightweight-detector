"""Central configuration for the prompt-injection detection pipeline.

Every experiment-level constant lives here so that the thesis can cite a
single source of truth and every run is reproducible.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]
# The Kaggle downloads live in archive/ (as delivered with the proposal).
DATA_RAW = PROJECT_ROOT / "archive"
DATA_PROCESSED = PROJECT_ROOT / "data" / "processed"
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = RESULTS_DIR / "figures"
TABLES_DIR = RESULTS_DIR / "tables"
MODELS_DIR = RESULTS_DIR / "models"

# ---------------------------------------------------------------------------
# Reproducibility
# ---------------------------------------------------------------------------
SEED = 42

# ---------------------------------------------------------------------------
# Dataset preparation (thesis §3.4 / §4.3)
# ---------------------------------------------------------------------------
SAMPLE_SIZE = 12_000          # stratified sample drawn from the raw corpus

# Balance the sample across SOURCE FILES as well as across labels.
#
# Without this, tapir.csv alone supplies 63% of the benign class and every
# one of its rows opens with the same twelve words, so a classifier can
# separate the classes by recognising which dataset a prompt came from. The
# prefix diagnostic measured the damage: the first 20 characters alone reach
# CV F1 0.911 against 0.970 for the whole prompt.
#
# Allocation is equal-share with redistribution: each source in a class is
# offered the same quota, sources holding fewer rows contribute everything
# they have, and the shortfall is redistributed among the sources that can
# still supply more. No arbitrary cut-off, and no data is wasted.
BALANCE_SOURCES = True
MIN_PROMPT_CHARS = 5          # rows shorter than this are dropped
MAX_PROMPT_CHARS = 5_000      # rows longer than this are dropped
DEDUP_COSINE_THRESHOLD = 0.97 # near-duplicate removal on MiniLM embeddings
TRAIN_FRAC, VAL_FRAC, TEST_FRAC = 0.70, 0.15, 0.15

# ---------------------------------------------------------------------------
# Feature extraction (thesis §3.5)
# ---------------------------------------------------------------------------
PERPLEXITY_MODEL = "gpt2"                     # 124M parameter scorer
ALT_PERPLEXITY_MODEL = "Qwen/Qwen2.5-0.5B"    # robustness check scorer
EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
PERPLEXITY_MAX_TOKENS = 1024                  # GPT-2 context limit safeguard
EMBEDDING_DIM = 384                           # all-MiniLM-L6-v2 output width
EMBEDDING_BATCH_SIZE = 256

# Minimum run length of base64-alphabet characters treated as an encoded
# payload (text-surface feature 11). Twenty characters is long enough that
# ordinary words and identifiers do not trigger it.
BASE64_RUN_MIN = 20

# Curated instruction-trigger phrases for the text-surface block.
# Sources: OWASP LLM01 examples, deepset injection benchmark patterns,
# and common jailbreak personas documented in the literature.
TRIGGER_PHRASES = [
    "ignore previous", "ignore all previous", "ignore the above",
    "disregard previous", "disregard the above", "forget previous",
    "forget all previous", "forget your instructions",
    "you are now", "you are no longer", "pretend to be", "pretend you are",
    "act as", "roleplay as", "role play as",
    "system prompt", "system message", "hidden prompt", "initial prompt",
    "reveal your instructions", "reveal your prompt", "repeat your instructions",
    "print your instructions", "output your instructions",
    "developer mode", "debug mode", "admin mode", "god mode",
    "dan", "do anything now", "jailbreak", "jailbroken",
    "no restrictions", "without restrictions", "bypass", "override",
    "new instructions", "your new task",
]

# ---------------------------------------------------------------------------
# Models (thesis §3.6)
# ---------------------------------------------------------------------------
LOGREG_MAX_ITER = 5_000
LOGREG_GRID = {"C": [0.01, 0.1, 1, 10]}

# TF-IDF baseline specification (thesis §3.6 step 6) - a citable design
# choice, so it belongs here rather than inline in the baseline module.
TFIDF_NGRAM_RANGE = (1, 2)
TFIDF_MAX_FEATURES = 20_000
TFIDF_SUBLINEAR_TF = True
XGB_GRID = {
    "n_estimators": [300, 500, 800],
    "max_depth": [4, 6, 8],
    "learning_rate": [0.03, 0.05, 0.1],
    "subsample": [0.7, 0.8, 1.0],
}
CV_FOLDS = 5

# Grid-search parallelism. On a small feature matrix (~6k rows x 401 cols) a
# single XGBoost fit does not scale across many threads, so running several
# fits concurrently with a few threads each is far faster than one fit using
# every core. Purely a scheduling choice: the seed is fixed, so results are
# identical either way.
GRID_N_JOBS = 6      # concurrent CV fits
MODEL_N_THREADS = 2  # threads per individual fit

# ---------------------------------------------------------------------------
# Classification protocol (thesis §3.7)
# ---------------------------------------------------------------------------
TARGET_RECALL = 0.95   # threshold tau = smallest value achieving this recall
BOOTSTRAP_ITERATIONS = 2_000  # for test-set confidence intervals

# ---------------------------------------------------------------------------
# Analysis and instrumentation
# ---------------------------------------------------------------------------
TSNE_PERPLEXITY = 30          # t-SNE neighbourhood size (embedding analysis)
DEDUP_BLOCK_SIZE = 1_000      # rows per similarity block during de-duplication
LATENCY_SAMPLE_SIZE = 200     # prompts timed for the latency measurement
