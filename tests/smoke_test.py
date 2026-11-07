#!/usr/bin/env python3
"""End-to-end check on SYNTHETIC data (no real results are used).

It builds a small fake per-case table with a known planted effect, runs
clean_tables.py and failure_decomposition.py on it, and checks that

  * a planted +0.01 Dice effect is recovered, with the mean inside its CI,
  * a contrast with no planted effect is not reported as an improvement,
  * Holm-adjusted p-values are never smaller than the raw p-values,
  * the exclusion list removes exactly the listed GLI cases (60 - 5 = 55).

Run it from the repository root:   python tests/smoke_test.py
"""
import os
import subprocess
import sys
import tempfile

import numpy as np
import pandas as pd

rng = np.random.default_rng(1)          # fixed seed: the test is deterministic
rows = []
cases = [f"BraTS-GLI-{i:05d}-000" for i in range(60)]   # 60 fake test cases

for n in (88, 876):                     # two training sizes
    for seed in (0, 1, 2):              # three seeds, as in the study
        # Per-case "difficulty": all strategies share it, which makes differences paired.
        base = np.clip(rng.normal(0.8, 0.1, len(cases)), 0, 1)
        # (strategy, planted shift): frozen is +0.01 better than rand_frozen, full +0.011 better than
        # rand_full_tuned; lora has no effect relative to its own base.
        for st, shift in [("frozen", 0.01), ("rand_frozen", 0.0), ("full", 0.011), ("rand_full_tuned", 0.0), ("lora", 0.0)]:
            dice = np.clip(base + shift + rng.normal(0, 0.005, len(cases)), 0, 1)
            if st == "rand_frozen":
                dice[0] = 0.0           # plant one complete failure for the decomposition script
            for c, v in zip(cases, dice):
                rows.append({"run": f"{st}_{n}_s{seed}", "strategy": st, "n_train": n, "seed": seed,
                             "dataset": "GLI", "case_id": c, "dice": v, "hd95_mm": 5.0})

tmp = tempfile.mkdtemp()                # scratch folder: nothing is written into the repository
pc = os.path.join(tmp, "per_case.csv")
pd.DataFrame(rows).to_csv(pc, index=False)
pd.DataFrame({"case_id": cases[:5]}).to_csv(os.path.join(tmp, "ex.csv"), index=False)   # exclude 5 cases

# Run both scripts exactly as a user would, with a small bootstrap to keep the test fast.
for cmd in (["scripts/clean_tables.py", "--per-case", pc, "--exclude", os.path.join(tmp, "ex.csv"), "--B", "2000", "--out", tmp],
            ["scripts/failure_decomposition.py", "--per-case", pc, "--exclude", os.path.join(tmp, "ex.csv"), "--out", tmp]):
    subprocess.run([sys.executable] + cmd, check=True, stdout=subprocess.DEVNULL)

c = pd.read_csv(os.path.join(tmp, "contrasts_primary_clean55_B2000.csv"))

# 1. The planted +0.01 effect must be recovered and lie inside its own CI.
eff = c[(c.A == "frozen") & (c.B == "rand_frozen") & (c.n_train == 876)].iloc[0]
assert abs(eff.mean_diff - 0.01) < 0.004, eff.mean_diff
assert eff.ci95_lo < eff.mean_diff < eff.ci95_hi

# 2. The no-effect contrast must be negative (lora has no +0.01 shift, frozen does) and Holm must not lower p.
null = c[(c.A == "lora") & (c.B == "frozen") & (c.n_train == 876)].iloc[0]
assert null.mean_diff < 0 and np.isnan(null.p_holm)          # strategy contrasts are outside the Holm family
assert eff.p_holm >= eff.p_seed_ttest - 1e-12                # Holm never lowers a p-value
full = c[(c.A == "full") & (c.B == "rand_full_matched") & (c.n_train == 876)]
assert len(full) == 1                                        # rand_full_tuned was merged into rand_full_matched

# 3. The exclusion list must have removed exactly 5 of the 60 cases.
assert eff.n_cases == 55
print("smoke test passed")
