#!/usr/bin/env python3
"""Write the case-level split (case IDs only, no image data) as a CSV.

INPUT   split.json from the notebook cache. It holds the list of all GLI case
        IDs ("ids") and three lists of positions into it ("train", "val",
        "test"), plus the seed that produced the split.
OUTPUT  a CSV with one row per case: case_id, split, index.

The split is CASE-level: the 1251 BraTS-GLI case IDs were randomly permuted
(seed 42) and cut 70 / 10 / 20 percent. Two time points of the same patient
can therefore land on different sides; make_clean_test_list.py finds those.

USAGE
  python scripts/export_split.py --split /path/to/cache/split.json --out splits/case_level_split.csv
"""
import argparse
import json

import pandas as pd

ap = argparse.ArgumentParser()
ap.add_argument("--split", required=True)
ap.add_argument("--out", default="splits/case_level_split.csv")
a = ap.parse_args()

sp = json.load(open(a.split))

# Look every position up in the ID list so the CSV carries case IDs, not indices.
rows = [{"case_id": sp["ids"][i], "split": k, "index": i} for k in ("train", "val", "test") for i in sp[k]]
df = pd.DataFrame(rows).sort_values("index")     # back to the original case order
df.to_csv(a.out, index=False)

# Sanity print: expect {'train': 876, 'val': 125, 'test': 250} and split seed 42.
print(df.split.value_counts().to_dict(), "| split seed", sp.get("seed"))
