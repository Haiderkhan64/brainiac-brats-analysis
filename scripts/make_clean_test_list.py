#!/usr/bin/env python3
"""List the GLI test cases that share a patient with a training case.

BACKGROUND
  BraTS-GLI IDs look like BraTS-GLI-00005-101: the five digits are the PATIENT,
  the last three digits the time point. Because the split permutes case IDs, two
  time points of one patient can end up on different sides of the split. A test
  case whose patient also has a TRAINING case could be easier than a truly
  unseen patient, so those cases are excluded from the primary analysis. The
  remaining cases are the "clean" test set (221 of 250).

NOTE ON VALIDATION
  Test cases that share a patient only with a VALIDATION case are kept, which is
  the rule the paper uses. (Four such cases exist; mention this in the paper.)

CHECK
  The output must read: excluded 29 | clean 221. If it does not, the rule here
  differs from the paper's; do not use this script, write your own list to
  splits/test_excluded.csv (column: case_id).

USAGE
  python scripts/make_clean_test_list.py --split splits/case_level_split.csv --out splits/test_excluded.csv
"""
import argparse

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--split", default="splits/case_level_split.csv")
ap.add_argument("--out", default="splits/test_excluded.csv")
a = ap.parse_args()

d = pd.read_csv(a.split)

# "BraTS-GLI-00005-101".split("-") -> ["BraTS", "GLI", "00005", "101"]; field 2 is the patient.
d["patient"] = d.case_id.str.split("-").str[2]

# Patients that have at least one TRAINING case.
other = set(d.loc[d.split == "train", "patient"])

# Test cases whose patient is in that set are the excluded ones.
test = d[d.split == "test"]
ex = test[test.patient.isin(other)][["case_id", "patient"]].sort_values("case_id")
ex.to_csv(a.out, index=False)
print(f"test cases: {len(test)} | excluded (patient also in train): {len(ex)} | clean: {len(test) - len(ex)}")
