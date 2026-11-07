#!/usr/bin/env python3
"""Turn the private run logs into the two public result files.

INPUT  (private, written by the training notebook)
  results_*.jsonl   one JSON record per finished run, with per-case Dice and
                    HD95 lists for the GLI test set, PEDs and Africa

OUTPUT (public, committed to the repository)
  results/all_66_runs_exact.csv   one row per run (exact Dice / HD95 / calibration summaries)
  results/per_case_metrics.csv    one row per run x test case (Dice and HD95 in mm)

Run this ONCE, privately, next to the notebook outputs, then commit the two CSVs.
It reads only the case-ID lists from the notebook cache (split.json, ped_ids.json,
afr_ids.json). No image data is read.

USAGE
  python scripts/export_results.py \\
      --results results_final.jsonl results_fewshot.jsonl \\
      --cache /path/to/cache --out results --expect 66
"""
import argparse
import json
import os

import pandas as pd

# Key used inside each run record -> dataset name used in the public CSVs.
# "test" is the GLI held-out test split; "ped" and "afr" are the external sets.
SETS = {"test": "GLI", "ped": "PEDs", "afr": "Africa"}

# The notebook stores the training size as a string for percentages ("10%" and
# "100%") and as a number for few-shot runs ("5", "20"). 10% and 100% of the
# 876 training cases are 88 and 876 cases.
SPEC_N = {"100%": 876, "10%": 88}


def n_train(spec):
    """Number of training cases for a run's `spec` field."""
    return SPEC_N.get(spec) or int(spec)


def norm_strategy(r):
    """Public strategy label for a run record.

    The random-init full fine-tuning baseline exists in three versions that the
    raw logs do not tell apart by the `strategy` field:
      * rfT_* runs                 -> rand_full_tuned (corrected control: same encoder LR
                                      multiplier as pretrained full fine-tuning)
      * rand_full at 5 / 20 cases  -> rand_full_matched (same multiplier as full)
      * rand_full at 88 / 876      -> rand_full_UNTUNED_mult1 (encoder LR multiplier 1)
    """
    if r["name"].startswith("rfT_"):
        return "rand_full_tuned"
    if r["strategy"] == "rand_full":      # logged before the corrected control existed
        return "rand_full_matched" if n_train(r["spec"]) in (5, 20) else "rand_full_UNTUNED_mult1"
    return r["strategy"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", nargs="+", required=True, help="one or more results_*.jsonl files")
    ap.add_argument("--cache", required=True, help="notebook cache folder with split.json and *_ids.json")
    ap.add_argument("--out", default="results")
    ap.add_argument("--expect", type=int, default=None, help="fail if the number of runs differs")
    a = ap.parse_args()

    # ---- read all run records; a later record with the same run name replaces an earlier one ----
    runs = {}
    for fn in a.results:
        for line in open(fn):
            if line.strip():
                r = json.loads(line)
                runs[r["name"]] = r
    runs = list(runs.values())
    if a.expect is not None and len(runs) != a.expect:
        raise SystemExit(f"expected {a.expect} runs, found {len(runs)}")

    # ---- case IDs, in the same order as the per-case score lists ----
    split = json.load(open(os.path.join(a.cache, "split.json")))
    ids = {
        # The notebook evaluates the GLI test set in the order of the sorted test indices.
        "test": [split["ids"][i] for i in split["test"]],
        "ped": json.load(open(os.path.join(a.cache, "ped_ids.json"))),
        "afr": json.load(open(os.path.join(a.cache, "afr_ids.json"))),
    }

    runs_rows, case_rows = [], []
    for r in runs:
        st, n = norm_strategy(r), n_train(r["spec"])
        row = {"run": r["name"], "strategy": st, "spec": r["spec"], "n_train": n, "seed": r["seed"],
               "lr": r.get("lr"), "iters": r.get("iters"), "best_val_dice": r.get("best_val_dice")}
        for k, nm in SETS.items():
            m = r[k]
            # Per-run summaries (one row per run).
            row[f"{nm}_dice"] = m["dice_mean"]
            row[f"{nm}_hd95_mm"] = m.get("hd95_mean")
            row[f"{nm}_hd95_capped"] = m.get("hd95_capped")   # cases with an empty prediction (HD95 set to the grid diagonal)
            for extra in ("ece", "brier", "brier_fg", "brier_bg"):
                if extra in m:
                    row[f"{nm}_{extra}"] = m[extra]
            # Per-case scores (one row per run x case).
            dl, hl = m["dice_list"], m.get("hd95_list", [None] * len(m["dice_list"]))
            if len(dl) != len(ids[k]):
                # A length mismatch would silently attach scores to the wrong case IDs.
                raise SystemExit(f"{r['name']}: {len(dl)} {nm} scores but {len(ids[k])} case IDs")
            for cid, dv, hv in zip(ids[k], dl, hl):
                case_rows.append({"run": r["name"], "strategy": st, "n_train": n, "seed": r["seed"],
                                  "dataset": nm, "case_id": cid, "dice": dv, "hd95_mm": hv})
        runs_rows.append(row)

    os.makedirs(a.out, exist_ok=True)
    pd.DataFrame(runs_rows).sort_values(["strategy", "n_train", "seed"]).to_csv(
        os.path.join(a.out, "all_66_runs_exact.csv"), index=False, float_format="%.10g")
    pd.DataFrame(case_rows).to_csv(os.path.join(a.out, "per_case_metrics.csv"), index=False, float_format="%.10g")
    print(f"wrote {len(runs_rows)} runs and {len(case_rows)} per-case rows to {a.out}/")


if __name__ == "__main__":
    main()
