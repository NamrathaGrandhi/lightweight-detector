"""Final evaluation on the held-out test set (thesis §3.7 step 5, Chapter 5).

Protocol: each trained model is loaded together with its validation-selected
threshold tau and evaluated on the test set EXACTLY ONCE. No tuning of any
kind happens after this script runs.

Reported per model:
    accuracy, precision, recall, F1, false-positive rate, ROC-AUC,
    each with a bootstrap 95% confidence interval (2,000 resamples),
    plus per-prompt inference latency (feature extraction excluded /
    included, measured separately in scripts/measure_latency.py).

Artefacts written for the thesis:
    results/tables/test_metrics.csv       one row per (model, metric)
    results/tables/test_metrics_wide.csv  models x metrics matrix
    results/figures/roc_curves.png        all models, one axis
    results/figures/pr_curves.png         precision-recall
    results/figures/confusion_<model>.png confusion matrix at tau
    results/tables/misclassified_<model>.csv   FP/FN prompts for error analysis
    results/tables/per_source_<model>.csv      recall / false-positive rate
                                               broken down by attack family
                                               and benign source
"""

from __future__ import annotations

import json

import joblib
import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    ConfusionMatrixDisplay,
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_recall_curve,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
)

from pidetect.config import (
    BOOTSTRAP_ITERATIONS,
    DATA_PROCESSED,
    FIGURES_DIR,
    MODELS_DIR,
    SEED,
    TABLES_DIR,
)
from pidetect.models.train import load_features

# (tag, model-file, feature groups or None for text input)
EVALUATED_MODELS = [
    ("combined_xgboost", "combined_xgboost.joblib", ["surface", "perplexity", "embedding"]),
    ("combined_logreg", "combined_logreg.joblib", ["surface", "perplexity", "embedding"]),
    ("ppl_only", "ppl_only_logreg.joblib", ["perplexity"]),
    ("surface_only", "surface_only_logreg.joblib", ["surface"]),
    ("length_only", "length_only_logreg.joblib", ["surface"]),  # validity diagnostic
    ("tfidf", "tfidf_logreg.joblib", None),
]

# Baselines restricted to a subset of their group's columns.
RESTRICTED_COLUMNS = {"length_only": ["ts_0"]}


def fpr_score(y_true: np.ndarray, y_pred: np.ndarray) -> float:
    tn, fp, _, _ = confusion_matrix(y_true, y_pred).ravel()
    return fp / (fp + tn) if (fp + tn) else 0.0


METRICS = {
    "accuracy": lambda y, p, s: accuracy_score(y, p),
    "precision": lambda y, p, s: precision_score(y, p, zero_division=0),
    "recall": lambda y, p, s: recall_score(y, p, zero_division=0),
    "f1": lambda y, p, s: f1_score(y, p, zero_division=0),
    "fpr": lambda y, p, s: fpr_score(y, p),
    "roc_auc": lambda y, p, s: roc_auc_score(y, s),
}


def bootstrap_ci(y, pred, scores, metric_fn, n_iter=BOOTSTRAP_ITERATIONS):
    """Percentile bootstrap 95% CI over test-set resamples."""
    rng = np.random.default_rng(SEED)
    n = len(y)
    stats = np.empty(n_iter)
    for b in range(n_iter):
        idx = rng.integers(0, n, n)
        # A resample without both classes cannot be scored; redraw.
        while len(np.unique(y[idx])) < 2:
            idx = rng.integers(0, n, n)
        stats[b] = metric_fn(y[idx], pred[idx], scores[idx])
    return float(np.percentile(stats, 2.5)), float(np.percentile(stats, 97.5))


def get_scores(tag: str, model, threshold: float):
    """Return (y_true, scores, preds, prompts) on the test split."""
    if tag == "tfidf":
        test = pd.read_parquet(DATA_PROCESSED / "test.parquet")
        y = test["label"].values
        scores = model.predict_proba(test["prompt"])[:, 1]
        prompts = test["prompt"]
    else:
        groups = dict((t, g) for t, _, g in EVALUATED_MODELS)[tag]
        X, y, names = load_features("test", groups)
        if tag in RESTRICTED_COLUMNS:
            X = X[:, [names.index(c) for c in RESTRICTED_COLUMNS[tag]]]
        scores = model.predict_proba(X)[:, 1]
        prompts = pd.read_parquet(DATA_PROCESSED / "test.parquet")["prompt"]
    preds = (scores >= threshold).astype(int)
    return y, scores, preds, prompts


def main() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)

    rows = []
    roc_data, pr_data = {}, {}

    for tag, fname, _groups in EVALUATED_MODELS:
        model_path = MODELS_DIR / fname
        if not model_path.exists():
            print(f"[skip] {tag}: {fname} not found")
            continue
        model = joblib.load(model_path)
        summary_file = MODELS_DIR / f"{tag.replace('_xgboost', '').replace('_logreg', '')}_training_summary.json"
        # combined models share one summary file keyed by model name
        if tag.startswith("combined"):
            with open(MODELS_DIR / "combined_training_summary.json") as f:
                tau = json.load(f)[tag.split("_")[1]]["threshold"]
        else:
            with open(summary_file) as f:
                tau = json.load(f)["threshold"]

        y, scores, preds, prompts = get_scores(tag, model, tau)
        print(f"[{tag}] tau={tau:.4f}  n_test={len(y):,}")

        for mname, fn in METRICS.items():
            point = fn(y, preds, scores)
            lo, hi = bootstrap_ci(y, preds, scores, fn)
            rows.append({"model": tag, "metric": mname,
                         "value": point, "ci_low": lo, "ci_high": hi})
            print(f"    {mname:10s} {point:.4f}  [{lo:.4f}, {hi:.4f}]")

        # curves
        fpr_c, tpr_c, _ = roc_curve(y, scores)
        roc_data[tag] = (fpr_c, tpr_c, roc_auc_score(y, scores))
        prec_c, rec_c, _ = precision_recall_curve(y, scores)
        pr_data[tag] = (rec_c, prec_c)

        # confusion matrix figure
        fig, ax = plt.subplots(figsize=(4, 4))
        ConfusionMatrixDisplay(
            confusion_matrix(y, preds),
            display_labels=["benign", "malicious"],
        ).plot(ax=ax, colorbar=False, values_format="d")
        ax.set_title(f"{tag} (tau={tau:.3f})")
        fig.tight_layout()
        fig.savefig(FIGURES_DIR / f"confusion_{tag}.png", dpi=200)
        plt.close(fig)

        # Misclassified prompts, and per-source recall/FPR, for the
        # error-analysis section (thesis §5.6).
        test_meta = pd.read_parquet(DATA_PROCESSED / "test.parquet")
        detail = pd.DataFrame({
            "prompt": prompts.values, "label": y,
            "score": scores, "pred": preds,
            "source_file": test_meta["source_file"].values,
        })
        detail[detail.label != detail.pred].to_csv(
            TABLES_DIR / f"misclassified_{tag}.csv", index=False)

        # Straight aggregation rather than groupby.apply: the grouping keys
        # are needed inside the calculation, and apply(include_groups=False)
        # removes them.
        detail["flagged"] = (detail["pred"] == 1).astype(int)
        by_source = (
            detail.groupby(["source_file", "label"])
            .agg(n=("label", "size"), flagged=("flagged", "sum"))
            .reset_index()
        )
        # For malicious sources the rate is recall (share correctly flagged);
        # for benign sources it is the false-positive rate (share wrongly
        # flagged). Both are simply flagged/n, but they mean opposite things,
        # so the metric is named explicitly.
        by_source["rate"] = by_source["flagged"] / by_source["n"]
        by_source["metric"] = np.where(by_source["label"] == 1,
                                       "recall", "false_positive_rate")
        by_source["model"] = tag
        by_source.sort_values(["label", "rate"]).to_csv(
            TABLES_DIR / f"per_source_{tag}.csv", index=False)

    # metric tables
    metrics_df = pd.DataFrame(rows)
    metrics_df.to_csv(TABLES_DIR / "test_metrics.csv", index=False)
    wide = metrics_df.pivot(index="model", columns="metric", values="value").round(4)
    wide.to_csv(TABLES_DIR / "test_metrics_wide.csv")
    print("\nModel x metric matrix:\n", wide)

    # ROC figure
    fig, ax = plt.subplots(figsize=(6, 5))
    for tag, (f, t, auc) in roc_data.items():
        ax.plot(f, t, label=f"{tag} (AUC={auc:.3f})")
    ax.plot([0, 1], [0, 1], "k--", lw=0.8)
    ax.set_xlabel("False positive rate")
    ax.set_ylabel("True positive rate")
    ax.legend(loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "roc_curves.png", dpi=200)
    plt.close(fig)

    # PR figure
    fig, ax = plt.subplots(figsize=(6, 5))
    for tag, (r, p) in pr_data.items():
        ax.plot(r, p, label=tag)
    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.legend(loc="lower left", fontsize=8)
    fig.tight_layout()
    fig.savefig(FIGURES_DIR / "pr_curves.png", dpi=200)
    plt.close(fig)

    print(f"\nEvaluation artefacts written to {TABLES_DIR} and {FIGURES_DIR}")


if __name__ == "__main__":
    main()
