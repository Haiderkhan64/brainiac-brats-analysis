#!/usr/bin/env python3
"""Split each mean paired Dice difference into a failed-case part and the rest.

WHY
---
A mean Dice difference can come from many small gains, or from a few cases that
one model segments completely wrong (Dice near 0). This script separates the
two, so the paper can say which of them drives a difference.

DEFINITIONS
-----------
* A (seed, case) cell is a FAILURE for a model when its Dice is below
  --fail-thresh.
* For a contrast A - B, a cell counts as "failed" if EITHER model fails on it.
* The mean difference then splits exactly as
      total = share_failed * mean_diff_on_failed
              + (1 - share_failed) * mean_diff_on_non_failed
  and the two terms are reported as contrib_failed / contrib_non_failed.

USAGE
-----
  python scripts/failure_decomposition.py --per-case results/per_case_metrics.csv \\
      --exclude splits/test_excluded.csv --fail-thresh 0.1 --out tables

IMPORTANT
---------
The default threshold (0.1) matches the GLI_lt10 column of the original analysis.
Use the definition of "complete failure" the paper states, and state it there.
"""
import argparse
import os

import pandas as pd

# Same contrasts as clean_tables.py (the "control" contrast is not decomposed). Keep the two lists identical.
CONTRASTS = [("frozen", "rand_frozen"), ("full", "rand_full_matched"), ("lora", "frozen"), ("full", "frozen")]

# The matched random-init control was logged as rand_full_matched (5 and 20 cases) and
# rand_full_tuned (88 and 876 cases): one arm, merged here (same as clean_tables.py).
RENAME = {"rand_full_tuned": "rand_full_matched"}


def matrix(d, strategy, n, dataset):
    """Seed x case table of Dice values; None if that arm has no runs."""
    s = d[(d.strategy == strategy) & (d.n_train == n) & (d.dataset == dataset)]
    return None if s.empty else s.pivot_table(index="seed", columns="case_id", values="dice", aggfunc="first")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", default="results/per_case_metrics.csv")
    ap.add_argument("--exclude", default=None, help="CSV with a case_id column: GLI test cases to drop")
    ap.add_argument("--fail-thresh", type=float, default=0.1, help="Dice below this counts as a failure")
    ap.add_argument("--out", default="tables")
    a = ap.parse_args()

    d = pd.read_csv(a.per_case)
    d["strategy"] = d["strategy"].replace(RENAME)
    if a.exclude:
        ex = set(pd.read_csv(a.exclude)["case_id"])
        d = d[~((d.dataset == "GLI") & d.case_id.isin(ex))]   # same clean-221 filter as clean_tables.py

    rows = []
    for ds in sorted(d.dataset.unique()):
        for n in sorted(d.n_train.unique()):
            for A, B in CONTRASTS:
                MA, MB = matrix(d, A, n, ds), matrix(d, B, n, ds)
                if MA is None or MB is None:
                    continue                      # one arm missing at this size
                seeds, cases = MA.index.intersection(MB.index), MA.columns.intersection(MB.columns)

                # Flatten the seed x case grid into one vector of "cells"; the pairing is kept
                # because both matrices are indexed identically.
                xa, xb = MA.loc[seeds, cases].to_numpy().ravel(), MB.loc[seeds, cases].to_numpy().ravel()
                fa, fb = xa < a.fail_thresh, xb < a.fail_thresh   # failure flags per model
                fail = fa | fb                                    # failed for either model
                diff = xa - xb                                    # paired difference per cell
                share = fail.mean()                               # fraction of cells that failed

                # Mean difference inside each group (0 if a group happens to be empty).
                on_f = diff[fail].mean() if fail.any() else 0.0
                on_n = diff[~fail].mean() if (~fail).any() else 0.0

                # Persistence of failures: cases that fail in ALL seeds / in AT LEAST ONE seed, per arm.
                FA, FB = (MA.loc[seeds, cases] < a.fail_thresh), (MB.loc[seeds, cases] < a.fail_thresh)
                # Dice on non-failed cells: mean over runs of the mean Dice of that run's non-failed cases.
                nfA = float(pd.Series([r[r >= a.fail_thresh].mean() for _, r in MA.loc[seeds, cases].iterrows()]).mean())
                nfB = float(pd.Series([r[r >= a.fail_thresh].mean() for _, r in MB.loc[seeds, cases].iterrows()]).mean())

                rows.append({"dataset": ds, "n_train": n, "A": A, "B": B, "thresh": a.fail_thresh,
                             "fail_all_seeds_A": int(FA.all(axis=0).sum()), "fail_any_seed_A": int(FA.any(axis=0).sum()),
                             "fail_all_seeds_B": int(FB.all(axis=0).sum()), "fail_any_seed_B": int(FB.any(axis=0).sum()),
                             "nonfailed_dice_A": nfA, "nonfailed_dice_B": nfB,
                             "cells": len(diff), "fail_A": int(fa.sum()), "fail_B": int(fb.sum()),
                             "fail_either": int(fail.sum()), "mean_diff_total": diff.mean(),
                             # The two contributions add up to mean_diff_total.
                             "contrib_failed": share * on_f, "contrib_non_failed": (1 - share) * on_n,
                             "mean_diff_non_failed": on_n})

    r = pd.DataFrame(rows)
    os.makedirs(a.out, exist_ok=True)
    fn = os.path.join(a.out, f"failure_decomposition_thr{a.fail_thresh:g}.csv")
    r.to_csv(fn, index=False, float_format="%.6f")
    print(r.round(4).to_string(index=False))
    print("wrote", fn)


if __name__ == "__main__":
    main()
