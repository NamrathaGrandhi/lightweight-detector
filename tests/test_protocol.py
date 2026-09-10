"""Tests for protocol-critical logic that does not need heavy models:
threshold selection and dataset-preparation invariants.
"""

import numpy as np
import pandas as pd
import pytest

from pidetect.data import prepare
from pidetect.models.train import select_threshold


class TestThresholdSelection:
    def test_perfect_separation(self):
        y = np.array([0, 0, 0, 1, 1, 1])
        s = np.array([0.1, 0.2, 0.3, 0.7, 0.8, 0.9])
        tau = select_threshold(y, s, target_recall=0.95)
        # All positives above tau, all negatives below.
        assert ((s >= tau).astype(int) == y).all()

    def test_recall_target_met(self):
        rng = np.random.default_rng(0)
        y = (rng.random(1000) < 0.3).astype(int)
        s = np.clip(y * 0.6 + rng.random(1000) * 0.5, 0, 1)
        tau = select_threshold(y, s, target_recall=0.95)
        recall = ((s >= tau) & (y == 1)).sum() / y.sum()
        assert recall >= 0.95

    def test_tau_is_maximal_for_target(self):
        # Choosing any higher threshold must break the recall target,
        # otherwise we are paying false positives for nothing.
        rng = np.random.default_rng(1)
        y = (rng.random(500) < 0.4).astype(int)
        s = np.clip(y * 0.5 + rng.random(500) * 0.6, 0, 1)
        tau = select_threshold(y, s, target_recall=0.95)
        higher = s[s > tau]
        if len(higher):
            tau2 = float(higher.min())
            recall2 = ((s >= tau2) & (y == 1)).sum() / y.sum()
            assert recall2 < 0.95


class TestCleaning:
    def _df(self, prompts, labels=None):
        return pd.DataFrame({
            "prompt": prompts,
            "label": labels or [0] * len(prompts),
            "source_file": "test.csv",
        })

    def test_length_filter(self):
        df = self._df(["ok prompt here", "abc", "x" * 6000])
        out = prepare.clean(df)
        assert len(out) == 1

    def test_whitespace_normalisation(self):
        df = self._df(["hello\t\tworld\n\nagain  end"])
        out = prepare.clean(df)
        assert out.loc[0, "prompt"] == "hello world again end"

    def test_exact_duplicates_dropped(self):
        df = self._df(["same prompt text", "same prompt text", "different one!"])
        out = prepare.clean(df)
        assert len(out) == 2

    def test_stratified_sample_preserves_ratio(self):
        df = self._df(
            ["prompt number %d" % i for i in range(1000)],
            labels=[1 if i < 300 else 0 for i in range(1000)],
        )
        out = prepare.sample(df, n=500)
        ratio = (out.label == 1).mean()
        assert abs(ratio - 0.3) < 0.02
