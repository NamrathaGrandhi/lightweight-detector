"""One-command reproduction of the entire study.

    python scripts/run_all.py                 # every stage
    python scripts/run_all.py --from train    # resume from a later stage
    python scripts/run_all.py --only evaluate # a single stage
    python scripts/run_all.py --fail-fast     # stop at the first failure

Stages: prepare -> featurize -> train -> baselines -> evaluate -> analysis
        -> latency -> alt_scorer
Prerequisite: raw Kaggle CSVs in archive/ (see README).

Resilience: by default a failing stage is logged and the run CONTINUES to the
next one, because the stages are only loosely coupled - a failure in the
embedding analysis, say, should not cost you the latency measurement. Every
stage caches its artefacts to disk, so a failed stage can be re-run on its own
afterwards with --only. A summary table prints at the end and the exit code is
non-zero if anything failed, so a failure is never mistaken for success.
"""

from __future__ import annotations

import argparse
import faulthandler
import sys
import time
import traceback
from pathlib import Path

# The latency and ablation stages live in this directory, not in the package.
sys.path.insert(0, str(Path(__file__).resolve().parent))

# Dump tracebacks for all threads if the process is killed or hangs, so a
# stall can be diagnosed rather than guessed at.
faulthandler.enable()

STAGES = ["prepare", "featurize", "train", "baselines", "evaluate", "analysis",
          "latency", "alt_scorer"]


def _run(name: str) -> None:
    if name == "prepare":
        from pidetect.data import prepare
        prepare.main()
    elif name == "featurize":
        from pidetect.data import featurize
        featurize.main()
    elif name == "train":
        from pidetect.models import train
        train.main()
    elif name == "baselines":
        from pidetect.models import baselines
        baselines.main()
    elif name == "evaluate":
        from pidetect.eval import evaluate
        evaluate.main()
    elif name == "analysis":
        from pidetect.analysis import embedding_space
        embedding_space.main()
    elif name == "latency":
        import measure_latency
        measure_latency.main()
    elif name == "alt_scorer":
        import alt_scorer_ablation
        alt_scorer_ablation.main()
    else:
        raise ValueError(f"unknown stage: {name}")


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--from", dest="start", choices=STAGES, default="prepare",
                    help="resume from this stage (earlier stages must be cached)")
    ap.add_argument("--only", choices=STAGES, help="run exactly one stage")
    ap.add_argument("--fail-fast", action="store_true",
                    help="abort at the first failure instead of continuing")
    args = ap.parse_args()

    todo = [args.only] if args.only else STAGES[STAGES.index(args.start):]
    results: list[tuple[str, str, float, str]] = []

    for stage in todo:
        print(f"\n{'=' * 70}\nSTAGE: {stage}\n{'=' * 70}", flush=True)
        t0 = time.time()
        try:
            _run(stage)
            elapsed = time.time() - t0
            results.append((stage, "OK", elapsed, ""))
            print(f"\nSTAGE {stage} OK in {elapsed:.0f}s", flush=True)
        except BaseException as exc:  # noqa: BLE001 - also catches SystemExit
            elapsed = time.time() - t0
            if isinstance(exc, KeyboardInterrupt):
                results.append((stage, "INTERRUPTED", elapsed, "user"))
                print(f"\nSTAGE {stage} INTERRUPTED after {elapsed:.0f}s",
                      flush=True)
                break
            reason = f"{type(exc).__name__}: {exc}".strip()
            results.append((stage, "FAILED", elapsed, reason))
            print(f"\nSTAGE {stage} FAILED after {elapsed:.0f}s -- {reason}",
                  file=sys.stderr, flush=True)
            traceback.print_exc()
            if args.fail_fast:
                print("--fail-fast set; aborting.", file=sys.stderr, flush=True)
                break
            print("Continuing to the next stage; re-run this one later with "
                  f"--only {stage}", flush=True)

    print(f"\n{'=' * 70}\nSUMMARY\n{'=' * 70}")
    for stage, status, elapsed, reason in results:
        line = f"  {stage:<12} {status:<12} {elapsed:>7.0f}s"
        print(f"{line}  {reason}" if reason else line)
    skipped = [s for s in todo if s not in {r[0] for r in results}]
    if skipped:
        print(f"  not reached: {', '.join(skipped)}")

    failures = [r[0] for r in results if r[1] != "OK"]
    if failures:
        print(f"\n{len(failures)} stage(s) did not succeed: {', '.join(failures)}")
        return 1
    print("\nAll stages complete. Artefacts in results/.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
