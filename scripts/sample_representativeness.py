"""Is the 12,000-row sample representative of the corpus it was drawn from?

Source balancing deliberately changes how much each file contributes, so the
mix of sources in the sample is not the mix in the corpus, and is not meant
to be. What must still hold is that the prompts drawn from each source are a
fair draw from that source. If they are not, the sample is biased in a way no
amount of balancing would justify. This compares the sampled prompts against
the full population of each source on character length, which is the property
most likely to expose a skewed draw.
"""

from __future__ import annotations

import glob
import os

import pandas as pd
from scipy import stats

from pidetect.config import DATA_PROCESSED, PROJECT_ROOT

COLS = ("prompt", "text", "question", "instruction", "input")


def load_source(path: str) -> pd.Series:
    d = pd.read_csv(path, low_memory=False)
    col = next((c for c in d.columns if c.lower() in COLS), None)
    if col is None:
        return pd.Series(dtype=str)
    return d[col].dropna().astype(str).str.strip()


def main() -> None:
    splits = [pd.read_parquet(DATA_PROCESSED / f"{s}.parquet")
              for s in ("train", "val", "test")]
    sample = pd.concat(splits, ignore_index=True)
    print(f"prepared corpus: {len(sample):,} prompts\n")

    print(f"{'source':<40}{'population':>11}{'sampled':>9}"
          f"{'pop med':>9}{'smp med':>9}{'shift':>8}{'KS p':>8}")
    print("-" * 94)

    flagged = []
    for path in sorted(glob.glob(str(PROJECT_ROOT / "archive" / "*.csv"))):
        name = os.path.basename(path)
        pop = load_source(path)
        if pop.empty:
            continue
        smp = sample.loc[sample["source_file"] == name, "prompt"]
        if len(smp) < 20:
            continue
        if name == "predictionguard_df.csv":
            # This file holds both classes, so its population median mixes
            # two very different distributions. Comparing against it would
            # measure the concatenation, not the draw.
            print(f"{name:<40}{len(pop):>11,}{len(smp):>9,}"
                  f"{'-':>9}{'-':>9}{'-':>8}{'  mixed file, skipped':>8}")
            continue
        pop_len, smp_len = pop.str.len(), smp.str.len()
        p = stats.ks_2samp(pop_len, smp_len).pvalue
        # A p-value alone is not enough. With populations in the tens of
        # thousands, KS reports significance for differences far too small
        # to matter, so the median shift is reported alongside it and both
        # must be large before a source is called biased.
        shift = abs(smp_len.median() - pop_len.median()) / pop_len.median()
        mark = ""
        if p < 0.01 and shift > 0.10:
            mark = "  <-- differs"
            flagged.append((name, shift))
        print(f"{name:<40}{len(pop):>11,}{len(smp):>9,}"
              f"{pop_len.median():>9.0f}{smp_len.median():>9.0f}"
              f"{shift * 100:>7.0f}%{p:>8.3f}{mark}")

    print()
    if flagged:
        print("Sources whose sampled prompts are both statistically and materially "
              "shorter or longer than their population:")
        for name, shift in flagged:
            print(f"  {name}  ({shift * 100:.0f}% shift in median length)")
        print("\nDe-duplication and cleaning run after sampling, so sources that are "
              "\nheavily duplicated lose their repeated long templates and shift "
              "\nshorter. That is the mechanism to check before calling it bias.")
    else:
        print("No source shifts by more than 10% in median length. The draw within "
              "each source is consistent with a fair sample.")


if __name__ == "__main__":
    main()
