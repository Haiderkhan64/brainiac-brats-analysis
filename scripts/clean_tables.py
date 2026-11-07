#!/usr/bin/env python3
"""Statistical tables for the BrainIAC / BraTS study.

WHAT THIS SCRIPT DOES
---------------------
Reads results/per_case_metrics.csv (one row per run x test case) and, for every
contrast in CONTRASTS, writes:

  * the mean paired difference (A - B), averaged over seeds and test cases
  * a 95% confidence interval from a TWO-LEVEL bootstrap (seeds, then cases)
  * a per-seed paired t-test (one-sample t-test on the per-seed mean differences)
  * Holm-adjusted p-values (family = the 24 pretraining contrasts of one metric)

It also writes a means table (mean and SD over seeds for every strategy).

PRIMARY ANALYSIS
----------------
GLI test cases listed in splits/test_excluded.csv are dropped before anything is
computed (the remaining 221 cases are the "clean" test set). PEDs and Africa are
never filtered.

USAGE
-----
  python scripts/clean_tables.py --per-case results/per_case_metrics.csv \\
      --exclude splits/test_excluded.csv --B 100000 --out tables

IMPORTANT
---------
The contrast list, the bootstrap definition and the Holm family below are the
defaults of this script. They must match what the paper states. If the paper
differs, edit CONTRASTS / HOLM_KIND here and re-run.
"""
import argparse
import os

import numpy as np
import pandas as pd
from scipy import stats

# ---------------------------------------------------------------------------
# Analysis definition (edit here if the paper's definition changes)
# ---------------------------------------------------------------------------

# Each pair is (A, B) and the reported difference is A - B.
# Differences are PAIRED: the same seed uses the same training subset for every
# strategy (see make_subset in the private training code), so for a given seed
# and test case we can subtract B's Dice from A's Dice directly.
# The third entry is the contrast KIND. Only "pretraining" contrasts form the
# Holm family (paper, Section 3.7: 3 cohorts x 4 sizes x 2 pairs = 24 tests).
CONTRASTS = [
    ("frozen", "rand_frozen", "pretraining"),      # pretraining effect, frozen encoder
    ("full", "rand_full_matched", "pretraining"),  # pretraining effect, full fine-tuning (matched random-init control)
    ("lora", "frozen", "strategy"),                # adaptation effect: LoRA vs frozen
    ("full", "frozen", "strategy"),                # adaptation effect: full fine-tuning vs frozen
    ("rand_full_matched", "rand_full_UNTUNED_mult1", "control"),  # tuning sensitivity (only at 88 and 876 cases)
]

# The matched random-init full fine-tuning control was logged under two names:
# "rand_full_matched" (5 and 20 cases) and "rand_full_tuned" (88 and 876 cases). Both use the
# same encoder-LR multiplier as pretrained full fine-tuning, so they are ONE arm.
# Without this merge the full-vs-matched contrast would be missing at 5 and 20 cases.
RENAME = {"rand_full_tuned": "rand_full_matched"}

# Holm's correction: ONE family containing all "pretraining" contrasts of a metric,
# across all datasets and training sizes (24 tests for Dice). Other kinds get no Holm value.
HOLM_KIND = "pretraining"


# ---------------------------------------------------------------------------
# Statistics helpers
# ---------------------------------------------------------------------------

def holm(p):
    """Holm step-down adjusted p-values.

    Sort the m p-values ascending; the i-th smallest (0-based) is multiplied by
    (m - i); then enforce monotonicity with a running maximum and cap at 1.
    NaNs are not part of the family: they are left out of m and returned as NaN.
    """
    p = np.asarray(p, float)
    out = np.full(p.shape, np.nan)        # NaN everywhere by default
    ok = ~np.isnan(p)                     # only real p-values take part
    q = p[ok]
    m = len(q)                            # family size
    order = np.argsort(q)                 # indices from smallest to largest p
    adj = np.empty(m)
    run = 0.0                             # running maximum (keeps adjusted p non-decreasing)
    for rank, i in enumerate(order):
        run = max(run, (m - rank) * q[i])
        adj[i] = min(1.0, run)            # a probability can never exceed 1
    out[ok] = adj
    return out


def seed_case_matrix(d, strategy, n, dataset, metric):
    """Return a table indexed by seed with one column per test case.

    Returns None when that (strategy, training size, dataset) has no runs, so the
    caller can skip contrasts whose arms do not exist (for example
    rand_full_tuned at 5 and 20 training cases).
    """
    s = d[(d.strategy == strategy) & (d.n_train == n) & (d.dataset == dataset)]
    if s.empty:
        return None
    # "first" is safe: there is exactly one value per (seed, case) in a clean export.
    return s.pivot_table(index="seed", columns="case_id", values=metric, aggfunc="first")


def two_level_bootstrap(D, B, rng, chunk=1000):
    """Two-level bootstrap of the mean paired difference.

    D is an array [seeds, cases] of paired differences. Each replicate
      1. resamples the seeds with replacement,
      2. resamples the test cases with replacement,
      3. averages D over the resampled seeds x cases.
    Returns the B replicate means; the 2.5th and 97.5th percentiles are the CI.

    NOTE: with very few seeds (3 here) resampling seeds cannot capture the full
    seed-to-seed variability, so this CI can be optimistic. The per-seed t-test
    is the more conservative check on the seed dimension.

    The loop runs in chunks so the memory stays small: one chunk needs
    chunk x seeds x cases numbers.
    """
    S, C = D.shape
    out = np.empty(B)
    done = 0
    while done < B:
        b = min(chunk, B - done)                       # size of this chunk
        si = rng.integers(0, S, (b, S))                # resampled seed indices, one row per replicate
        ci = rng.integers(0, C, (b, C))                # resampled case indices, one row per replicate
        # Fancy indexing builds [b, S, C] with D[seed_i, case_j]; the mean over S and C is the replicate mean.
        out[done:done + b] = D[si[:, :, None], ci[:, None, :]].mean(axis=(1, 2))
        done += b
    return out


def case_only_bootstrap(dm, B, rng, chunk=1000):
    """Bootstrap of the mean of dm (seed-averaged paired difference per case), resampling cases only."""
    C = len(dm)
    out = np.empty(B)
    done = 0
    while done < B:
        b = min(chunk, B - done)
        out[done:done + b] = dm[rng.integers(0, C, (b, C))].mean(axis=1)
        done += b
    return out


# ---------------------------------------------------------------------------
# Main program
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--per-case", default="results/per_case_metrics.csv")
    ap.add_argument("--exclude", default=None, help="CSV with a case_id column: GLI test cases to drop")
    ap.add_argument("--metrics", nargs="+", default=["dice"], choices=["dice", "hd95_mm"])
    ap.add_argument("--B", type=int, default=100000, help="bootstrap replicates")
    ap.add_argument("--rng-seed", type=int, default=0, help="seed of the bootstrap random generator (repeatable output)")
    ap.add_argument("--B-case", type=int, default=5000, help="replicates of the case-only sensitivity bootstrap (Table A1)")
    ap.add_argument("--out", default="tables")
    ap.add_argument("--tag", default=None, help="file-name tag (default: clean<N> or all<N> GLI cases)")
    a = ap.parse_args()

    d = pd.read_csv(a.per_case)
    d["strategy"] = d["strategy"].replace(RENAME)     # merge the two names of the matched control

    # ---- optional exclusion of GLI test cases (the "clean 221" analysis) ----
    n_gli = d.loc[d.dataset == "GLI", "case_id"].nunique()
    if a.exclude:
        ex = set(pd.read_csv(a.exclude)["case_id"])
        # Only GLI rows are filtered; PEDs and Africa keep all their cases.
        d = d[~((d.dataset == "GLI") & d.case_id.isin(ex))]
        n_gli = d.loc[d.dataset == "GLI", "case_id"].nunique()
        tag = a.tag or f"clean{n_gli}"
    else:
        tag = a.tag or f"all{n_gli}"
    print(f"GLI test cases used: {n_gli}")       # sanity check: must read 221 for the primary analysis
    os.makedirs(a.out, exist_ok=True)
    rng = np.random.default_rng(a.rng_seed)

    # ---- table 1: means (per dataset, training size, strategy) ----
    # For each seed take the mean over test cases, then report mean and SD over seeds.
    rows = []
    for (ds, n, st), g in d.groupby(["dataset", "n_train", "strategy"]):
        for m in a.metrics:
            per_seed = g.groupby("seed")[m].mean()
            rows.append({"dataset": ds, "n_train": n, "strategy": st, "metric": m,
                         "n_seeds": len(per_seed), "n_cases": g.case_id.nunique(),
                         "mean": per_seed.mean(),
                         "sd_over_seeds": per_seed.std(ddof=1) if len(per_seed) > 1 else np.nan})
    pd.DataFrame(rows).to_csv(os.path.join(a.out, f"means_primary_{tag}.csv"), index=False, float_format="%.6f")

    # ---- table 2: paired contrasts ----
    res = []
    for ds in sorted(d.dataset.unique()):
        for m in a.metrics:
            for n in sorted(d.n_train.unique()):
                for A, Bn, kind in CONTRASTS:
                    MA = seed_case_matrix(d, A, n, ds, m)
                    MB = seed_case_matrix(d, Bn, n, ds, m)
                    if MA is None or MB is None:
                        continue                  # one arm does not exist at this size: no contrast
                    # Pair on seeds and cases that both arms have.
                    seeds = MA.index.intersection(MB.index)
                    cases = MA.columns.intersection(MB.columns)
                    if len(seeds) < 2 or len(cases) < 2:
                        continue                  # too little data for a variance estimate
                    D = (MA.loc[seeds, cases] - MB.loc[seeds, cases]).to_numpy()   # [seeds, cases]
                    per_seed = D.mean(axis=1)     # one mean difference per seed (input of the t-test)
                    boot = two_level_bootstrap(D, a.B, rng)
                    lo, hi = np.percentile(boot, [2.5, 97.5])
                    # Sensitivity intervals for Appendix Table A1 (do not touch the main random stream):
                    # (a) case-only bootstrap of the seed-averaged difference (ignores seed variance, too narrow);
                    dm = D.mean(axis=0)
                    ci_case = np.percentile(case_only_bootstrap(dm, a.B_case, np.random.default_rng(a.rng_seed + 1)), [2.5, 97.5])
                    # (b) per-seed t interval, df = seeds - 1 (conservative with 3 seeds).
                    S_ = len(per_seed)
                    half = stats.t.ppf(0.975, S_ - 1) * per_seed.std(ddof=1) / np.sqrt(S_) if S_ > 1 else np.nan
                    t_lo, t_hi = per_seed.mean() - half, per_seed.mean() + half
                    # A zero-variance sample would make the t statistic undefined, so guard it.
                    t = stats.ttest_1samp(per_seed, 0.0) if per_seed.std(ddof=1) > 0 else None
                    res.append({"dataset": ds, "metric": m, "n_train": n, "A": A, "B": Bn, "kind": kind,
                                "n_seeds": len(seeds), "n_cases": len(cases),
                                "mean_diff": D.mean(), "ci95_lo": lo, "ci95_hi": hi,
                                "ci_case_only_lo": ci_case[0], "ci_case_only_hi": ci_case[1],
                                "ci_seed_t_lo": t_lo, "ci_seed_t_hi": t_hi,
                                "per_seed_diffs": ";".join(f"{x:.4f}" for x in per_seed),
                                "t": t.statistic if t else np.nan, "df": len(seeds) - 1,
                                "p_seed_ttest": t.pvalue if t else np.nan})
    r = pd.DataFrame(res)
    if r.empty:
        raise SystemExit("no contrast could be computed (check strategy names / seeds / cases)")

    # ---- Holm correction: one family of pretraining contrasts per metric ----
    r["p_holm"] = np.nan
    for _, idx in r[r.kind == HOLM_KIND].groupby("metric").groups.items():
        r.loc[idx, "p_holm"] = holm(r.loc[idx, "p_seed_ttest"].to_numpy())
    print("Holm family sizes:", r[r.kind == HOLM_KIND].groupby("metric").size().to_dict())

    fn = os.path.join(a.out, f"contrasts_primary_{tag}_B{a.B}.csv")
    r.to_csv(fn, index=False, float_format="%.6f")
    print(r.drop(columns=["per_seed_diffs"]).round(4).to_string(index=False))
    print("wrote", fn)


if __name__ == "__main__":
    main()
