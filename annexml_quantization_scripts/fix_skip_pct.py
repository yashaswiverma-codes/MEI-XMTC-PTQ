#!/usr/bin/env python3
"""
fix_skip_pct.py

The run_annexml_hitrate_skip.sh script's grep pattern for extracting
skip_pct from each run's timing log was wrong (expected "NN.NN% below"
but the actual C++ output is "(NN.NN%) below" -- percent sign
followed by a closing paren before " below"), so every row in the
master CSV has skip_pct=0.0 regardless of the actual threshold used.

This does NOT affect accuracy (P@1/P@3/P@5) or timing correctness --
those were parsed and computed correctly. This script only fixes the
skip_pct column, by re-reading it from the raw timing logs (which do
contain the correct value) still sitting on disk.

Usage:
    python3 fix_skip_pct.py \
        --master_csv /DATA2/.../results/hitrate_skip_all_datasets.csv \
        --results_base /DATA2/.../results \
        --out /DATA2/.../results/hitrate_skip_all_datasets_fixed.csv
"""
import argparse
import csv
import os
import re

SKIP_PCT_RE = re.compile(r'\(([0-9.]+)%\)\s+below')

def get_correct_skip_pct(timing_log_path):
    if not os.path.exists(timing_log_path):
        return None
    with open(timing_log_path) as f:
        content = f.read()
    m = SKIP_PCT_RE.search(content)
    return float(m.group(1)) if m else 0.0  # 0.0 is correct for noskip runs

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--master_csv", required=True)
    ap.add_argument("--results_base", required=True,
                     help="e.g. /DATA2/rudra1/annexml_quantization/annexml/results")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    with open(args.master_csv) as f:
        rows = list(csv.DictReader(f))

    fixed = 0
    for row in rows:
        ds = row["dataset"]
        label = row["threshold"]  # "noskip" or "thresh_0.001" etc.
        timing_log = os.path.join(
            args.results_base, ds, "hitrate_skip", "timing", f"{label}_timing.log"
        )
        pct = get_correct_skip_pct(timing_log)
        if pct is not None:
            row["skip_pct"] = f"{pct:.4f}"
            fixed += 1
        else:
            print(f"[WARN] Could not find/parse {timing_log} for {ds}/{label}")

    with open(args.out, "w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=rows[0].keys())
        writer.writeheader()
        writer.writerows(rows)

    print(f"Fixed {fixed}/{len(rows)} rows.")
    print(f"Wrote corrected CSV -> {args.out}")

if __name__ == "__main__":
    main()
